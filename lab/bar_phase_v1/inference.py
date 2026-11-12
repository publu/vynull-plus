"""Conservative preview-cue phase shifts; no stored beat-grid modifications."""
from pathlib import Path
import os,sys,json,hashlib,copy,urllib.request
import numpy as np,joblib
from features import descriptors,rotations,probabilities
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN/'semantic_pair_v1'))
from audio_context import extract

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def api(route):
    with urllib.request.urlopen(os.environ.get('VYNULL_API','http://127.0.0.1:9443/api/')+route,timeout=30) as r:return json.load(r)

def phase_prediction(model,values,times,labels,start):
    center=int(np.argmin(abs(times-start)))
    if abs(times[center]-start)>.02:raise ValueError('Planned cue no longer matches the beat grid')
    if labels[center]!=1:return dict(eligible=False,reason='Cue is an intentional non-downbeat position',center=center)
    if len(times)<32:return dict(eligible=False,reason='Insufficient phase context',center=center)
    begin=max(0,min(center-16,len(times)-32));window=values[begin:begin+32];bar_labels=labels[begin:begin+32]
    if len(window)!=32 or not np.array_equal(bar_labels,(np.arange(32)+bar_labels[0]-1)%4+1):
        return dict(eligible=False,reason='Non-four-beat bar labels',center=center)
    steps=np.diff(times[begin:begin+32]);period=np.median(steps)
    if period<=0 or not np.all((steps>=.8*period)&(steps<=1.2*period)):
        return dict(eligible=False,reason='Irregular beat spacing',center=center)
    x=rotations(window);p=np.clip(model['scorer'].predict_proba(x[:,model['columns']])[:,1],1e-7,1-1e-7)
    distribution=probabilities((np.log(p)-np.log1p(-p))[None,:],model['temperature'])[0]
    return dict(eligible=True,center=center,window_start_seconds=float(times[begin]),current_grid_phase=int(np.flatnonzero(bar_labels[:4]==1)[0]),
        model_phase=int(np.argmax(distribution)),confidence=float(max(distribution)),probabilities=distribution.tolist())

def supported_shift(diagnostic,times,before_beats,after_beats,duration,threshold):
    result=dict(diagnostic,applied=False,shift_beats=0,shift_seconds=0.)
    if not diagnostic['eligible'] or threshold is None:return result
    if diagnostic['confidence']<threshold:
        result['reason']='No confident alternative bar phase';return result
    phase=(diagnostic['model_phase']-diagnostic['current_grid_phase'])%4
    if phase==0:result['reason']='Supplied bar phase retained';return result
    center=diagnostic['center']
    options=sorted([n for n in [-2,-1,1,2] if n%4==phase],key=lambda n:(abs(n),n))
    for shift in options:
        moved=center+shift;end=moved+after_beats
        if moved-before_beats<0 or end>=len(times) or times[end]>=duration-.01:continue
        result.update(applied=True,shift_beats=shift,shift_seconds=float(times[moved]-times[center]),reason='Confident supported preview-cue shift')
        return result
    result['reason']='Alternative phase lacks surrounding audio support';return result

class PreviewPhaseAligner:
    def __init__(self):
        lock=read(HERE/'selection-lock.json');gate=read(HERE/'test-evaluation.json');selection=read(HERE/'selection.json');path=HERE/'bar_phase.joblib'
        if not gate['offline_correction_gate_passed']:raise ValueError('Bar-phase model has not passed its offline gate')
        assert sha(path)==lock['model_sha256']==gate['model_sha256']
        assert sha(HERE/'features.py')==lock['feature_code_sha256']
        assert sha(TRAIN/'semantic_pair_v1/audio_context.py')==selection['extractor_sha256']
        self.model=joblib.load(path);self.model_sha256=lock['model_sha256'];self.tracks={t['id']:t for t in api('tracks')};self.cache={}
        self.measured=read(TRAIN.parent/'review/measured-audio.json')

    def context(self,tid,expected_sha):
        if tid in self.cache:
            if self.cache[tid]['audio_sha256']!=expected_sha:raise ValueError('Cached audio identity does not match the plan')
            return self.cache[tid]
        track=self.tracks[tid];a=api(f'analysis/{tid}');bpm=float(a['bpm'])
        if not 80<=bpm<=180:return None
        data=np.load(TRAIN/'semantic_pair_v1/halo_features'/f'{tid}.npz');config=json.loads(str(data['config']))
        if sha(track['file_path'])!=expected_sha or config['audio_sha256']!=expected_sha:raise ValueError('Audio changed after part selection')
        times=np.asarray([b['time_ms']/1000 for b in a['beats']]);labels=np.asarray([b['beat_in_bar'] for b in a['beats']])
        keep=(times>=0)&(times<config['duration']);times,labels=times[keep],labels[keep]
        fp=hashlib.sha256((expected_sha+str(bpm)+sha(HERE/'features.py')+sha(TRAIN/'semantic_pair_v1/audio_context.py')).encode()+times.tobytes()).hexdigest()
        directory=HERE/'halo_features';directory.mkdir(exist_ok=True);path=directory/f'{tid}.npz'
        if path.exists() and str(np.load(path)['fingerprint'])==fp:values=np.load(path)['descriptors']
        else:
            values=descriptors(extract(track['file_path'],times,bpm*4,config['duration']));temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,descriptors=values,fingerprint=np.array(fp));temp.replace(path)
        self.cache[tid]=dict(values=values,times=times,labels=labels,duration=config['duration'],audio_sha256=expected_sha,bpm=bpm);return self.cache[tid]

    def selection(self,selection,plan):
        changed=copy.deepcopy(selection);diagnostics={};overlap=selection['overlap_bars'];loop=plan.get('requested_loop_bars',0)
        for side in ['source','target']:
            d=self.context(selection[side+'_track'],plan[side+'_audio_sha256'])
            if d is None:
                diagnostics[side]=dict(eligible=False,reason='Tempo outside phase-model domain',applied=False,shift_beats=0,shift_seconds=0.);continue
            if abs(d['bpm']-plan[side+'_bpm'])>1e-6:raise ValueError('Beat tempo changed after part selection')
            prediction=phase_prediction(self.model,d['values'],d['times'],d['labels'],selection[side+'_start_seconds'])
            before=0 if side=='target' or loop else 16;after=4*(loop or overlap) if side=='source' else 4*(overlap+4)
            delta=supported_shift(prediction,d['times'],before,after,d['duration'],self.model['threshold']);diagnostics[side]=delta
            if delta['applied']:
                changed[side+'_start_seconds']+=delta['shift_seconds']
                for key in ['start','end']:changed[side+'_section'][key]+=delta['shift_seconds']
        # Small cue shifts must still satisfy the planner's harmonic restriction.
        if plan.get('harmonic_filter_applied') and any(d['applied'] for d in diagnostics.values()):
            sys.path.insert(0,str(TRAIN.parent/'review'));from section_matches import key_relation
            keys=[]
            for side in ['source','target']:
                rows=self.measured.get(str(changed[side+'_track']),{}).get('sections',[])
                keys.append(min(rows,key=lambda r:abs(r['start_seconds']-changed[side+'_start_seconds'])).get('key') if rows else None)
            if key_relation(*keys)!='compatible':
                changed=copy.deepcopy(selection)
                for delta in diagnostics.values():
                    if delta['applied']:delta.update(applied=False,shift_beats=0,shift_seconds=0.,reason='Shift withheld by harmonic-support guard')
        changed['bar_phase_alignment']=dict(model_sha256=self.model_sha256,threshold=self.model['threshold'],source=diagnostics['source'],target=diagnostics['target'],
            original_source_start_seconds=selection['source_start_seconds'],original_target_start_seconds=selection['target_start_seconds'],library_grid_changed=False)
        return changed

    def plan(self,plan):
        result=copy.deepcopy(plan);result['selection']=self.selection(plan['selection'],plan)
        if result['selection']['bar_phase_alignment']['source']['applied']:result['loop_candidate']=None
        for entry in result.get('alternatives',[]):
            original=copy.deepcopy(entry['selection']);entry['selection']=self.selection(original,plan)
            if entry['selection']['bar_phase_alignment']['source']['applied']:entry['loop_candidate']=None
        result['bar_phase_model_sha256']=self.model_sha256;result['bar_phase_preview_only']=True
        return result
