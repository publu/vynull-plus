"""Frozen music features -> functional sections -> small paired-location scorer."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,fcntl,collections,warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

HERE=Path(__file__).parent;OUT=HERE/'artifacts';OUT.mkdir(exist_ok=True)
SEED=20261105
KINDS=['intro','altintro','buildup','breakdown','drop','cooldown','bridge','outro','altoutro','end','unknown']

def functional_probabilities(model,x):
    p=model.predict_proba(x);result=np.zeros((len(x),len(KINDS)),np.float32)
    result[:,model[-1].classes_.astype(int)]=p
    return result

def pair_features(a,b,ea,eb):
    # Include ALL outgoing/incoming functional combinations, not just equal labels.
    outer=(a[:,4:15,None]*b[:,None,4:15]).reshape(len(a),-1)
    cosine=np.sum(ea*eb,axis=1,keepdims=True)
    return np.c_[a,b,abs(a-b),a*b,outer,cosine].astype(np.float32)

def diverse_picks(points,scores,bars,n=3):
    selected=[]
    for i in np.argsort(-scores,kind='stable'):
        if all(np.max(abs(points[i]-points[j])/bars)>=8 for j in selected):selected.append(int(i))
        if len(selected)==n:break
    return selected

def main():
    lock=(HERE/'train.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    while not (HERE/'embeddings-complete.json').exists():
        print('WAITING_FOR_FROZEN_MUSIC_FEATURES',flush=True);time.sleep(30)
    started=time.time();encoder=json.load(open(HERE/'encoder.json'))
    manifest=json.load(open(HERE.parent/'acoustic_pair_v1/audio-manifest.json'))
    original=json.load(open(HERE.parent/'acoustic_pair_v1/artifacts/eligible_partitions.json'))
    refined=json.load(open(HERE.parent/'part_alignment_v1/artifacts/refined_partitions.json'))
    tracksets={p:{r[k] for r in rows for k in ('outgoing_id','incoming_id')} for p,rows in original.items()}
    cache={}
    for tid in set.union(*tracksets.values()):
        p=HERE/'embeddings'/f'{tid}.npz'
        if not p.exists():continue
        e=np.load(p);d=np.load(HERE.parent/'raveform_audio_v1/features'/f'{tid}.npz')
        assert str(e['audio_sha256'])==manifest[tid]['sha256']
        assert str(e['model_revision'])==encoder['revision']
        assert np.array_equal(e['times'],d['times'][::4])
        cache[tid]={'times':e['times'],'embedding':e['embeddings'],'raw':d['x'][::4],'label':d['section'][::4]}
    parts={p:[r for r in rows if r['outgoing_id'] in cache and r['incoming_id'] in cache] for p,rows in refined.items()}
    for a,b in [('train','test'),('train','validation'),('test','validation')]:
        assert not tracksets[a]&tracksets[b]
        assert not {r['mix_id'] for r in original[a]}&{r['mix_id'] for r in original[b]}
        assert not {manifest[t]['sha256'] for t in tracksets[a]}&{manifest[t]['sha256'] for t in tracksets[b]}
    if len(parts['train'])<100 or min(len(parts['validation']),len(parts['test']))<20:raise RuntimeError('Insufficient paired features')
    ids={p:sorted(ts&cache.keys()) for p,ts in tracksets.items()}
    data={}
    for p,ts in ids.items():
        data[p]={'embedding':np.concatenate([cache[t]['embedding'] for t in ts]),
            'raw':np.concatenate([cache[t]['raw'] for t in ts]),
            'y':np.concatenate([cache[t]['label'] for t in ts]),
            'groups':np.concatenate([np.repeat(t,len(cache[t]['times'])) for t in ts])}
    pca=PCA(n_components=32,svd_solver='randomized',random_state=SEED).fit(data['train']['embedding'])
    for p,d in data.items():d['x']=np.c_[d['raw'],pca.transform(d['embedding'])].astype(np.float32)
    def functional_fit(x,y,c=1,groups=None):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=400,class_weight='balanced',random_state=SEED))
        weight=None
        if groups is not None:
            counts=collections.Counter(groups);weight=np.array([1/counts[g] for g in groups]);weight*=len(weight)/sum(weight)
        model.fit(x,y,logisticregression__sample_weight=weight);return model
    def functional_metrics(model,x,y):
        return float(f1_score(y,model.predict(x),labels=list(range(len(KINDS))),average='macro',zero_division=0))
    trials=[]
    for c in [.01,.1,1]:
        m=functional_fit(data['train']['x'],data['train']['y'],c,data['train']['groups'])
        f=functional_metrics(m,data['validation']['x'],data['validation']['y']);trials.append((f,c,m))
        print('FUNCTIONAL_VALIDATION',c,f,flush=True)
    _,c,functional=max(trials,key=lambda t:t[0])
    baselines={name:functional_fit(data['train']['raw'][:,:size],data['train']['y'],.1,data['train']['groups']) for name,size in [('position',4),('raw_acoustics',244)]}
    # Cross-fit on track identities to keep the paired scorer's training section
    # predictions out-of-fold, matching its inference situation on unseen tracks.
    oof=np.zeros((len(data['train']['x']),len(KINDS)),np.float32)
    for i,(train_idx,held_idx) in enumerate(GroupKFold(n_splits=3).split(data['train']['x'],data['train']['y'],data['train']['groups'])):
        m=functional_fit(data['train']['x'][train_idx],data['train']['y'][train_idx],c,data['train']['groups'][train_idx])
        oof[held_idx]=functional_probabilities(m,data['train']['x'][held_idx]);print('FUNCTIONAL_CROSSFIT',i,flush=True)
    # Compact node: 4 position/tempo + 11 inferred functional probabilities +
    # 12 acoustic energy/texture measurements + 16 frozen semantic components.
    core=np.r_[np.arange(4,8),np.arange(52,56),np.arange(100,104)]
    for split,ts in ids.items():
        cursor=0
        for tid in ts:
            d=cache[tid];n=len(d['times']);compressed=pca.transform(d['embedding'])
            probabilities=oof[cursor:cursor+n] if split=='train' else functional_probabilities(functional,np.c_[d['raw'],compressed])
            d['node']=np.c_[d['raw'][:,:4],probabilities,d['raw'][:,core],compressed[:,:16]].astype(np.float32)
            cursor+=n
    known=collections.defaultdict(list)
    for rows in parts.values():
        for r in rows:known[(r['outgoing_id'],r['incoming_id'])].append((r['outgoing_start'],r['incoming_start']))
    rng=np.random.default_rng(SEED);train_x=[];train_y=[];train_w=[]
    for r in parts['train']:
        a,b=cache[r['outgoing_id']],cache[r['incoming_id']]
        centers=[int(np.argmin(abs(d['times']-r[k+'_start']))) for d,k in [(a,'outgoing'),(b,'incoming')]]
        ap,bp=[np.unique(np.clip(center+np.array([-1,0,1]),0,len(d['times'])-1)) for d,center in [(a,centers[0]),(b,centers[1])]]
        # Neighboring four-bar candidates count as positives only within four bars.
        ap=ap[abs(a['times'][ap]-r['outgoing_start'])<=4*240/r['outgoing_bpm']]
        bp=bp[abs(b['times'][bp]-r['incoming_start'])<=4*240/r['incoming_bpm']]
        if not len(ap) or not len(bp):continue
        positive=[(int(i),int(j)) for i in ap for j in bp];negative=set();attempts=0
        bars=np.array([240/r['outgoing_bpm'],240/r['incoming_bpm']]);targets=np.array(known[(r['outgoing_id'],r['incoming_id'])])
        while len(negative)<64 and attempts<2000:
            attempts+=1
            i=int(rng.choice(ap)) if attempts%4==0 else int(rng.integers(len(a['times'])))
            j=int(rng.choice(bp)) if attempts%4==1 else int(rng.integers(len(b['times'])))
            point=np.array([a['times'][i],b['times'][j]])
            if np.min(np.max(abs(targets-point)/bars,axis=1))<8:continue
            negative.add((i,j))
        ia,ib=np.array(positive+sorted(negative)).T
        train_x.append(pair_features(a['node'][ia],b['node'][ib],a['embedding'][ia],b['embedding'][ib]))
        train_y.extend([1]*len(positive)+[0]*len(negative));train_w.extend([.5/len(positive)]*len(positive)+[.5/len(negative)]*len(negative))
    x=np.concatenate(train_x);y=np.array(train_y);w=np.array(train_w);w*=len(w)/sum(w)
    print('PAIR_HEAD_ROWS',x.shape,flush=True)
    def groups(split):
        out=[]
        for r in parts[split]:
            a,b=cache[r['outgoing_id']],cache[r['incoming_id']];ia,ib=np.meshgrid(np.arange(len(a['times'])),np.arange(len(b['times'])),indexing='ij');ia=ia.ravel();ib=ib.ravel()
            points=np.c_[a['times'][ia],b['times'][ib]];bars=np.array([240/r['outgoing_bpm'],240/r['incoming_bpm']]);targets=np.array(known[(r['outgoing_id'],r['incoming_id'])])
            error=np.min(np.max(abs(points[:,None,:]-targets[None,:,:])/bars,axis=2),axis=1)
            features=pair_features(a['node'][ia],b['node'][ib],a['embedding'][ia],b['embedding'][ib])
            out.append({'row':r,'x':features,'position':np.c_[a['node'][ia,:4],b['node'][ib,:4]],'points':points,'bars':bars,'error':error})
        return out
    val=groups('validation')
    def evaluate(model,gs,position=False):
        rows=[]
        for g in gs:
            score=model.predict_proba(g['position'] if position else g['x'])[:,1]
            picks=diverse_picks(g['points'],score,g['bars']);errors=g['error'][picks]
            rows.append({'mix_id':g['row']['mix_id'],'outgoing_id':g['row']['outgoing_id'],'incoming_id':g['row']['incoming_id'],
                'top1_error_bars':float(errors[0]),'top3_error_bars':float(min(errors)),'selected_times':g['points'][picks].tolist()})
        metrics={'queries':len(rows),'median_top1_error_bars':float(np.median([r['top1_error_bars'] for r in rows]))}
        for n in [1,3]:
            for tolerance in [4,8,16]:metrics[f'top{n}_hit_within_{tolerance}_bars_both_tracks']=float(np.mean([r[f'top{n}_error_bars']<=tolerance for r in rows]))
        return metrics,rows
    def position_rows(features):return np.c_[features[:,:4],features[:,43:47]]
    position=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=30,learning_rate=.06,early_stopping=False,random_state=SEED)
    position.fit(position_rows(x),y,sample_weight=w)
    trials=[]
    for strength in [.001,.01,.1,1]:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=400,random_state=SEED))
        m.fit(x,y,logisticregression__sample_weight=w);metrics,_=evaluate(m,val)
        trials.append((metrics['top3_hit_within_4_bars_both_tracks'],-metrics['median_top1_error_bars'],strength,m,metrics))
        print('PAIR_HEAD_VALIDATION',strength,metrics,flush=True)
    _,_,strength,head,valmetrics=max(trials,key=lambda t:t[:2])
    test=groups('test');pairmetrics,predictions=evaluate(head,test);positionmetrics,_=evaluate(position,test,True)
    functional_report={p:{'semantic_and_acoustic_macro_f1':functional_metrics(functional,data[p]['x'],data[p]['y']),
        'raw_acoustic_macro_f1':functional_metrics(baselines['raw_acoustics'],data[p]['raw'],data[p]['y']),
        'position_macro_f1':functional_metrics(baselines['position'],data[p]['raw'][:,:4],data[p]['y'])} for p in ['validation','test']}
    artifacts={'functional':functional,'semantic_pca':pca,'pair_head':head,'position_baseline':position,
        'node_core_indices':core,'encoder':encoder,'kinds':KINDS,'node_dimensions':43,'pair_dimensions':x.shape[1],
        'feature_version':'semantic_function_pair_v1'}
    joblib.dump(artifacts,OUT/'semantic_part_scorer.joblib')
    report={'encoder':encoder,'functional_section_metrics':functional_report,
        'functional_classifier_regularization':c,'paired_scorer_regularization':strength,
        'pair_scorer_parameters':int(head[-1].coef_.size+head[-1].intercept_.size),'pair_feature_dimensions':x.shape[1],
        'paired_partitions':{p:len(v) for p,v in parts.items()},'functional_tracks':{p:len(v) for p,v in ids.items()},
        'validation':{'semantic_pair':valmetrics,'position_baseline':evaluate(position,val,True)[0]},
        'test':{'semantic_pair':pairmetrics,'position_baseline':positionmetrics},'seconds':time.time()-started,
        'split_verification':'No downstream track/session/compressed-audio-hash overlap. PCA and scalers fit on training only. Functional predictions cross-fit by training track for pair-head training.',
        'supervision':'Human functional labels for section classifier; observed DJ overlap locations versus weak unobserved alternatives for pair scorer.',
        'deployment_gate':{'status':'offline_experiment','promoted':False,'requirement':'Held-out improvement, fresh confirmatory split and real listening validation'},
        'limitations':['Pretrained corpus overlap is unknown.','This downstream holdout was also inspected in previous model experiments; comparisons are exploratory, not a fresh final confirmation.',
            'Observed overlap location is a weak proxy for good mixing; many unobserved locations may also work.',
            'Grid uses manually checked dataset beats. Live beat-grid inference needs its own checks.',
            'No personal listening feedback or fader curves are learned in this run.']}
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2));(OUT/'test_predictions.json').write_text(json.dumps(predictions,indent=2))
    print(json.dumps(report),flush=True);print('SEMANTIC_PART_SCORER_COMPLETE',flush=True)

if __name__=='__main__':main()
