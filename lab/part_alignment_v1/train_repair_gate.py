"""Learn when an onset-based repair helps; confirm on previously unused test tracks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,hashlib,csv,subprocess
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from refine_beats import refine

HERE=Path(__file__).parent;DATA=Path(os.environ.get('RAVEFORM_DIR','raveform/data/raveform'));SR=22050

def features(d):
    shift=np.array(d['proposed_offsets_ms'])
    return [abs(np.median(shift)),np.mean(abs(shift)),np.std(shift),np.mean(abs(np.diff(shift))),
        d['evidence_ratio'],d['supported_fraction'],d['original_strength_ratio'],
        d['beat_period_ms'],d['mean_candidate_strength'],d['mean_original_strength']]

def summarize(rows,threshold=None,model=None):
    result={}
    for kind in ['control','corrupted']:
        chosen=[r for r in rows if (r['case']=='control')==(kind=='control')];errors=[];applied=[]
        for r in chosen:
            use=r['applied'] and (threshold is None or model.predict_proba([r['features']])[0,1]>=threshold)
            errors.append(r['proposal_error_ms'] if use else r['before_error_ms'])
            if use:applied.append(r)
        result[kind]={'spans':len(chosen),'applied':len(applied),
            'before_mae_ms':float(np.mean([r['before_error_ms'] for r in chosen])),
            'after_mae_ms':float(np.mean(errors)),
            'worsened_over_10ms_when_applied':float(np.mean([r['proposal_error_ms']-r['before_error_ms']>10 for r in applied])) if applied else 0}
    return result

def main():
    original=json.load(open(HERE.parent/'acoustic_pair_v1/artifacts/eligible_partitions.json'))
    audio=json.load(open(HERE.parent/'acoustic_pair_v1/audio-manifest.json'))
    tracks={t['id']:t for t in json.load(open(DATA/'structures/segments.json'))}
    earlier=json.load(open(HERE/'beat-refinement-evaluation.json'));bias=earlier['onset_bias_ms']/1000
    def ordered(split):
        ids={r[k] for r in original[split] for k in ['outgoing_id','incoming_id']}
        return sorted(ids,key=lambda t:hashlib.sha256(('phase-benchmark:'+t).encode()).hexdigest())
    sets={'train':ordered('train')[:40],'validation':ordered('train')[40:60],'test':ordered('test')[30:60]}
    assert not set(sets['test'])&set(earlier['held_out_tracks'])
    assert not set(sets['train'])&set(sets['validation'])
    def collect(ids):
        results=[]
        for tid in ids:
            with (DATA/'structures/beats'/(tracks[tid]['key']+'.beat.csv')).open() as f:times=np.array([float(r['time']) for r in csv.DictReader(f)])
            for fraction in [.3,.65]:
                i=min(max(4,int(len(times)*fraction)),len(times)-37);absolute=times[i:i+32]
                start=max(0,float(absolute[0])-1);end=float(absolute[-1])+1
                raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',audio[tid]['path'],'-t',str(end-start),'-ac','1','-ar',str(SR),'-f','f32le','pipe:1'],timeout=90)
                y=np.frombuffer(raw,'<f4');truth=absolute-start
                for kind,phase,drift in [('control',0,0),('early',-.045,-.0015),('late',.045,.0015)]:
                    damaged=truth+phase+(truth-truth[0])*drift
                    _,d=refine(y,SR,damaged,onset_bias=bias);proposal=damaged+np.array(d['proposed_offsets_ms'])/1000
                    before=float(np.mean(abs(damaged-truth))*1000);after=float(np.mean(abs(proposal-truth))*1000)
                    results.append({'track':tid,'fraction':fraction,'case':kind,'before_error_ms':before,'proposal_error_ms':after,
                        'applied':d['correction_applied'],'worthwhile':bool(before-after>=10 and after<=25),'features':features(d)})
            print('REPAIR_GATE_FEATURES',tid,len(results),flush=True)
        return results
    train=collect(sets['train']);val=collect(sets['validation'])
    eligible=[r for r in train if r['applied']]
    x=np.array([r['features'] for r in eligible]);y=np.array([r['worthwhile'] for r in eligible])
    model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,
        l2_regularization=10,random_state=20261106,early_stopping=False).fit(x,y)
    choices=[]
    for threshold in np.r_[np.arange(.1,1,.05),1.01]:
        metrics=summarize(val,float(threshold),model);c=metrics['control'];b=metrics['corrupted']
        if c['after_mae_ms']<=2.5 and c['applied']<=1 and b['worsened_over_10ms_when_applied']<=.05:
            choices.append((b['after_mae_ms'],float(threshold),metrics))
    if not choices:raise RuntimeError('No confidence threshold satisfies validation controls')
    _,threshold,valmetrics=min(choices,key=lambda t:t[0]);print('REPAIR_GATE_THRESHOLD',threshold,valmetrics,flush=True)
    test=collect(sets['test']);ungated=summarize(test);gated=summarize(test,threshold,model)
    c=gated['control'];b=gated['corrupted']
    gate_passed=bool(c['after_mae_ms']<=5 and b['after_mae_ms']<b['before_mae_ms'] and b['applied']>=20 and b['worsened_over_10ms_when_applied']<=.1)
    artifact={'model':model,'threshold':threshold,'onset_bias_seconds':bias,'feature_version':'onset_repair_confidence_10_v1','offline_gate_passed':gate_passed}
    joblib.dump(artifact,HERE/'artifacts/beat_repair_gate.joblib')
    report={'partitions':sets,'training_eligible_spans':len(eligible),'threshold':threshold,'onset_bias_ms':bias*1000,
        'validation':valmetrics,'fresh_test':{'ungated':ungated,'gated':gated},'offline_gate_passed':gate_passed,
        'heldout_disjoint_from_first_benchmark':True,'live_grid_changes_enabled':False,
        'scope':'Supervised confidence for synthetic beat-grid damage on real music; not a downbeat detector or listening-quality model',
        'test_examples':test}
    (HERE/'beat-repair-gate-evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='test_examples'},indent=2));print('BEAT_REPAIR_GATE_COMPLETE',flush=True)

if __name__=='__main__':main()
