"""Render a planned library pair with learned repair gating and separate deterministic DSP."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,argparse,urllib.request,subprocess,hashlib,wave,fcntl,time,uuid,zipfile,importlib.metadata
from pathlib import Path
import numpy as np,joblib
from scipy.signal import butter,sosfilt,resample_poly
from beatwarp import warp,repeat_loop
from refine_beats import refine
from train_repair_gate import features as repair_features
from loop_boundaries import metrics as loop_boundary_metrics

HERE=Path(__file__).parent;TRAINING=HERE.parent;ROOT=TRAINING/'section_pair_v1/renders';ROOT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(TRAINING.parent/'review'))
from preview_fades import gains as fade_gains
from render_preferences import select as select_render_preference
sys.path.insert(0,str(TRAINING/'semantic_pair_v1'))
from audio_context import extract
SR=48000

def api(route):
    with urllib.request.urlopen(os.environ.get('VYNULL_API','http://127.0.0.1:9443/api/')+route,timeout=60) as r:return json.load(r)

def decode(path,start,end):
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(path),'-t',str(end-start),
        '-vn','-ac','2','-ar',str(SR),'-f','f32le','pipe:1'],timeout=120)
    y=np.frombuffer(raw,'<f4').reshape(-1,2).copy()
    if not len(y) or not np.isfinite(y).all():raise ValueError('Audio decode failed')
    return y

def _aligned_stem_uncached(track,start,before_bars,bars,mix_bpm,gate,output_frames=None,cue_nudge_frames=0,_analysis=None):
    if type(cue_nudge_frames)!=int or abs(cue_nudge_frames)>round(2*60/mix_bpm*SR):raise ValueError('Incoming cue adjustment exceeds two beats')
    cue_shift=cue_nudge_frames/SR
    analysis=_analysis if _analysis is not None else api(f"analysis/{track['id']}");all_beats=np.array([b['time_ms']/1000 for b in analysis['beats']])
    center=int(np.argmin(abs(all_beats-start)))
    if abs(all_beats[center]-start)>.02:raise ValueError('The planned part no longer matches the song beat grid')
    begin=center-before_bars*4;end=begin+bars*4
    if begin<0 or end>=len(all_beats):raise ValueError('Not enough beat-grid context for the overlap')
    original=all_beats[begin:end+1];query=all_beats[center:center+32]
    origin=max(0,float(original[0])-1)
    finish=max(float(original[-1]),float(query[-1]))+1
    y=decode(track['file_path'],origin,finish)
    query=query[query-origin<len(y)/SR-.01]
    correction={'correction_applied':False,'reason':'Insufficient 32-beat analysis context','predicted_worthwhile_score':None}
    offsets=np.zeros(len(original))
    if len(query)==32:
        # The confidence model was calibrated with 22.05kHz analysis. Keep that
        # feature resolution while rendering the actual music at 48kHz.
        repair_audio=resample_poly(y,147,320,axis=0)
        _,d=refine(repair_audio,22050,query-origin,onset_bias=gate['onset_bias_seconds'])
        score=float(gate['model'].predict_proba([repair_features(d)])[0,1]) if d['correction_applied'] else 0.
        applied=bool(d['correction_applied'] and gate['offline_gate_passed'] and score>=gate['threshold'])
        if applied:offsets=np.interp(original,query,np.array(d['proposed_offsets_ms'])/1000)
        correction=dict(d,correction_applied=applied,predicted_worthwhile_score=score,threshold=gate['threshold'],
            overlap_beats_measured=32,context_offset_policy='Use the first/last estimated offset outside the measured overlap window')
    repaired=np.maximum(0,original+offsets+cue_shift)
    start_clamped=bool(original[0]+offsets[0]+cue_shift<0)
    first,last=np.rint((repaired[[0,-1]]-origin)*SR).astype(int)
    if cue_nudge_frames and (first<0 or last>len(y)):
        # Keep the onset-repair evidence identical to the reviewed unshifted
        # context. Decode additional REAL audio only after computing the map.
        origin=max(0,min(origin,float(repaired[0])-.1));finish=max(finish,float(repaired[-1])+.1)
        y=decode(track['file_path'],origin,finish);first,last=np.rint((repaired[[0,-1]]-origin)*SR).astype(int)
    if first<0 or last>len(y) or last<=first:raise ValueError('Corrected anchors outside decoded audio')
    unit=y[first:last].copy();source=repaired-repaired[0];source[-1]=len(unit)/SR
    target=np.arange(len(source))*60/mix_bpm
    target[-1]=(output_frames if output_frames is not None else round(target[-1]*SR))/SR
    result,info=warp(unit,SR,source,target,float(target[-1]))
    info.update({'beat_repair':correction,'original_start_seconds':float(original[0]),
        'corrected_start_seconds':float(repaired[0]),'corrected_end_seconds':float(repaired[-1]),
        'clamped_at_file_start':start_clamped,
        'unverified_downbeat_phase':True})
    if cue_nudge_frames:info['cue_nudge']=dict(shift_frames=cue_nudge_frames,shift_beats=cue_shift*mix_bpm/60,method='Previously approved controlled-preview setting; grid unchanged')
    return result,info

STEM_CACHE_STATS={'hits':0,'misses':0,'invalid':0}
_STEM_RUNTIME=None

def aligned_stem(track,start,before_bars,bars,mix_bpm,gate,output_frames=None,cue_nudge_frames=0):
    """Reuse exact beat warps only for identical audio, grid, gate and DSP code."""
    global _STEM_RUNTIME
    analysis=api(f"analysis/{track['id']}")
    if _STEM_RUNTIME is None:
        _STEM_RUNTIME={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','joblib']}
        for tool in ['rubberband','ffmpeg']:
            result=subprocess.run([tool,'--version' if tool=='rubberband' else '-version'],capture_output=True,text=True,timeout=10)
            _STEM_RUNTIME[tool]=(result.stdout+result.stderr).splitlines()[0]
    implementation={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ['render_live.py','beatwarp.py','refine_beats.py','train_repair_gate.py']}
    config={'version':'exact_aligned_stem_v1','track':track['id'],
        'audio_sha256':hashlib.sha256(Path(track['file_path']).read_bytes()).hexdigest(),
        'grid':{'bpm':analysis.get('bpm'),'beats':[[v['time_ms'],v.get('beat_in_bar')] for v in analysis['beats']]},
        'start':float(start),'before_bars':before_bars,'bars':bars,'mix_bpm':float(mix_bpm),
        'frames':output_frames,'nudge_frames':cue_nudge_frames,'gate_hash':joblib.hash(gate),'implementation':implementation,'runtime':_STEM_RUNTIME}
    key=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()
    folder=TRAINING/'analysis_cache/aligned_stems_v1';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.npz')
    with (folder/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as cached:
                    y=cached['audio'];info=json.loads(str(cached['info']))
                    if str(cached['key'])!=key or y.shape!=(info['target_frames'],2) or not np.isfinite(y).all():raise ValueError('Invalid stem cache')
                STEM_CACHE_STATS['hits']+=1
                return y,info
            except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):
                STEM_CACHE_STATS['invalid']+=1
        STEM_CACHE_STATS['misses']+=1
        y,info=_aligned_stem_uncached(track,start,before_bars,bars,mix_bpm,gate,output_frames,cue_nudge_frames,_analysis=analysis)
        temp=folder/(key+'.'+uuid.uuid4().hex+'.tmp.npz')
        with temp.open('wb') as file:np.savez(file,audio=y,info=np.array(json.dumps(info,allow_nan=False)),key=np.array(key))
        temp.replace(path)
        return y,info

def loop_check(track,start,bars,bpm,duration):
    _,values=extract(track['file_path'],[start],bpm,duration,loops=[(start,bars)])
    model=joblib.load(TRAINING/'raveform_audio_v1/artifacts/loop_region_continuity.joblib')
    probability=float(model.predict_proba(values)[0,1]);begin=values[0,2:50];finish=values[0,50:98]
    a,b=begin[-12:],finish[-12:];tonal=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-10))
    return {'structural_continuity_probability':probability,'boundary_chroma_cosine':tonal,
        'model_target':'Same annotated functional region; not human-rated loop quality',
        'accepted':bool(probability>=.8 and tonal>=.5)}

def append_preview(preview):
    lock=(ROOT/'manifest.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'manifest.json';data=json.load(open(path)) if path.exists() else {'renders':[]}
        data['renders'].append(preview);temp=ROOT/('manifest.'+uuid.uuid4().hex+'.tmp.json')
        temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def _render(plan,job_id,loop_bars=0,eq=True,publish=True):
    if not isinstance(job_id,str) or not job_id or any(c not in '0123456789abcdef_' for c in job_id):raise ValueError('Invalid job identifier')
    selection=dict(plan['selection']);tracks={t['id']:t for t in api('tracks')}
    source,target=tracks[selection['source_track']],tracks[selection['target_track']]
    for track,name in [(source,'source'),(target,'target')]:
        digest=hashlib.sha256(Path(track['file_path']).read_bytes()).hexdigest()
        if digest!=plan[name+'_audio_sha256']:raise ValueError('Song audio changed after analysis; refresh its music features')
    render_preference=select_render_preference(TRAINING.parent/'review/review.sqlite',selection,plan['source_audio_sha256'],plan['target_audio_sha256'],loop_bars,
        enabled=bool(eq and plan.get('listening_policy',{}).get('enabled',False)))
    cue_nudge=render_preference.get('target_cue_nudge_frames',0) if render_preference['applied'] else 0
    handoff=render_preference.get('handoff_bars') if render_preference['applied'] else None
    gate=joblib.load(HERE/'artifacts/beat_repair_gate.joblib')
    bpm=selection['mix_bpm'];bar=240/bpm;overlap=selection['overlap_bars'];npre=round(4*bar*SR);nover=round(overlap*bar*SR);npost=round(4*bar*SR)
    loop=None
    if loop_bars:
        if loop_bars not in [4,8]:raise ValueError('Choose a four- or eight-bar loop')
        cached=plan.get('loop_candidate')
        model_hash=hashlib.sha256((TRAINING/'raveform_audio_v1/artifacts/loop_region_continuity.joblib').read_bytes()).hexdigest()
        if cached and cached.get('continuity_model_sha256')==model_hash and cached.get('bars')==loop_bars and abs(cached['start']-selection['source_start_seconds'])<1e-5:
            check=dict(cached,pre_screened_boundary_level_difference_db=cached['boundary_level_difference_db'])
        else:check=loop_check(source,selection['source_start_seconds'],loop_bars,plan['source_bpm'],plan['source_duration_seconds'])
        if not check['accepted']:raise ValueError('This part lacks enough continuity evidence for a loop; try a blend')
        expected=round(loop_bars*bar*SR)
        unit,source_info=aligned_stem(source,selection['source_start_seconds'],0,loop_bars,bpm,gate,expected)
        if len(unit)!=expected:raise ValueError('Loop length changed during alignment')
        boundary=loop_boundary_metrics(unit,SR,bar);level_jump=boundary['boundary_level_difference_db']
        if level_jump>8:raise ValueError('The loop has an abrupt boundary-level change; try a blend')
        repeated=repeat_loop(unit,int(np.ceil((npre+nover)/len(unit)))+2,sample_rate=SR)
        # End the pre-roll at unit zero, so the advertised overlap starts at the
        # actual chosen loop start for both four- and eight-bar units.
        prefix_boundary=int(np.ceil(npre/len(unit)))*len(unit)
        a=repeated[prefix_boundary-npre:prefix_boundary+nover]
        loop=dict(check,bars=loop_bars,source_start=source_info['corrected_start_seconds'],source_end=source_info['corrected_end_seconds'],
            period_frames=len(unit),**boundary,edge_taper_ms=3,musical_review='pending')
    else:a,source_info=aligned_stem(source,selection['source_start_seconds'],4,4+overlap,bpm,gate,npre+nover)
    b,target_info=aligned_stem(target,selection['target_start_seconds'],0,overlap+4,bpm,gate,nover+npost,cue_nudge_frames=cue_nudge)
    if len(a)!=npre+nover or len(b)!=nover+npost:raise ValueError('Aligned mix duration is inconsistent')
    outgoing_rms=float(np.sqrt(np.mean(a[npre:]**2)+1e-12));incoming_rms=float(np.sqrt(np.mean(b[:nover]**2)+1e-12))
    incoming_gain=float(np.clip(outgoing_rms/incoming_rms,.5,2))
    if render_preference['applied'] and render_preference.get('incoming_level_gain') is not None:incoming_gain=float(render_preference['incoming_level_gain'])
    b*=incoming_gain
    t=np.linspace(0,1,nover);smooth=t*t*(3-2*t);ga,gb=fade_gains(nover,overlap,handoff)
    if eq:
        lowa=sosfilt(butter(3,180,fs=SR,output='sos'),a,axis=0);lowb=sosfilt(butter(3,180,fs=SR,output='sos'),b,axis=0)
        blend=lowa[npre:]*(1-smooth[:,None])+lowb[:nover]*smooth[:,None]+(a[npre:]-lowa[npre:])*ga+(b[:nover]-lowb[:nover])*gb
    else:blend=a[npre:]*ga+b[:nover]*gb
    result=np.concatenate([a[:npre],blend,b[nover:]])
    if not np.isfinite(result).all() or len(result)!=npre+nover+npost:raise ValueError('Invalid rendered audio')
    peak=float(abs(result).max());gain=min(1,.95/max(peak,1e-10))
    if render_preference['applied'] and render_preference.get('output_gain') is not None:gain=min(gain,float(render_preference['output_gain']))
    result*=gain;nfade=round(.01*SR)
    result[:nfade]*=np.linspace(0,1,nfade)[:,None];result[-nfade:]*=np.linspace(1,0,nfade)[:,None]
    output=ROOT/f'live_{job_id}.wav'
    if output.exists():raise ValueError('This immutable audition output already exists')
    with wave.open(str(output),'wb') as f:
        f.setnchannels(2);f.setsampwidth(2);f.setframerate(SR);f.writeframes((result*32767).astype('<i2').tobytes())
    with wave.open(str(output)) as f:assert f.getnframes()==len(result) and f.getnchannels()==2 and f.getframerate()==SR
    method=selection['selector'];label=('Learned parts' if method=='semantic' else 'Measured parts')+(' · selected loop' if loop and plan.get('planning_version')=='loop_conditioned_v1' else ' · loop-assisted v2' if loop else ' · aligned blend v2')
    if selection.get('bar_phase_alignment'):label+=' · bar-checked'
    if selection.get('listening_preference'):label+=' · liked parts'
    if render_preference['applied']:
        selection['base_target_start_seconds']=selection['target_start_seconds'];selection['target_start_seconds']+=cue_nudge/SR
        selection['target_section']=dict(selection['target_section'])
        for field in ['start','end']:selection['target_section'][field]+=cue_nudge/SR
        selection['render_preference']=render_preference;label+=' · approved settings'
    preview={'selection':selection,'engine':{'pre_bars':4,'post_bars':4,'loop':loop,'source_alignment':source_info,'target_alignment':target_info,
        'incoming_level_gain':incoming_gain,'level_matching':'Overlap RMS ratio, capped at ±6 dB; not perceptual loudness normalization',
        'bass_handoff':'Smoothstep low-band handoff at 180 Hz','mid_high_fade':'Equal-power smoothstep'},
        'eq_enabled':eq,'audio':{'path':str(output),'seconds':len(result)/SR,'frames':len(result),'sample_rate':SR,
            'peak_before_gain':peak,'gain_applied':gain,'peak_after_gain':float(abs(result).max()),
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},
        'comparison_label':label,'job_id':job_id,'created':time.time(),'selector_model_sha256':plan['model_sha256'],
        'selector_version':plan['selector_version'],'source_audio_sha256':plan['source_audio_sha256'],'target_audio_sha256':plan['target_audio_sha256'],
        'planning_version':plan.get('planning_version','blend_v1'),
        'algorithm':'Part scorer → confidence-gated beat repair → bounded pitch-preserving time maps → optional whole-bar loop → RMS gain match → deterministic EQ/fade',
        'listener_review':'pending','deck_commands_sent':False}
    if render_preference['applied']:
        preview['engine']['fade_envelope']=dict(version='feedback_fade_exit_v1',handoff_bars=handoff,held_bars=overlap-handoff,bass_handoff_unchanged=True)
    if publish:append_preview(preview)
    return preview

def render(plan,job_id,loop_bars=0,eq=True,publish=True):
    if plan.get('requested_loop_bars') and plan['requested_loop_bars']!=loop_bars:
        raise ValueError('The requested loop length does not match its plan')
    candidates=[plan]
    if loop_bars and plan.get('planning_version')=='loop_conditioned_v1':
        candidates += [dict(plan,selection=r['selection'],loop_candidate=r['loop_candidate'],alternatives=[]) for r in plan.get('alternatives',[])[:2]]
    attempts=[]
    for candidate in candidates:
        start=candidate['selection']['source_start_seconds'];target=candidate['selection']['target_start_seconds']
        try:preview=_render(candidate,job_id,loop_bars,eq,publish=False)
        except ValueError as e:
            retryable=any(reason in str(e) for reason in ['lacks enough continuity evidence','abrupt boundary-level change','Local tempo correction exceeds bounds'])
            if not loop_bars or not retryable:raise
            attempts.append({'source_start':start,'target_start':target,'result':'rejected','reason':str(e)});continue
        attempts.append({'source_start':start,'target_start':target,'result':'selected'})
        if loop_bars:preview['engine']['loop_search']={'candidate_attempts':attempts,'planned_loop_regions':len(candidates)}
        if publish:append_preview(preview)
        return preview
    raise ValueError('No candidate loop passed the final rendered boundary checks; try another length or a blend')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--job-id',required=True)
    parser.add_argument('--loop-bars',type=int,default=0);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=render(json.load(open(args.plan)),args.job_id,args.loop_bars)
    temp=args.output.with_suffix('.tmp.json');temp.write_text(json.dumps(result,indent=2));temp.replace(args.output)
    print(json.dumps({'titles':[result['selection']['source_title'],result['selection']['target_title']],'audio':result['audio'],'variant':result['comparison_label']}),flush=True)

if __name__=='__main__':main()
