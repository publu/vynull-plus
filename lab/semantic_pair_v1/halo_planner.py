"""Choose feasible parts of a fixed library pair; audio rendering is separate."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib,urllib.request
from pathlib import Path
import numpy as np,joblib
from train_scorer import functional_probabilities,pair_features,diverse_picks

HERE=Path(__file__).parent;BRIDGE=HERE.parent.parent
import sys
sys.path.insert(0,str(BRIDGE/'review'))
from section_matches import SectionIndex,key_relation
import part_preferences

def api(route):
    with urllib.request.urlopen(os.environ.get('VYNULL_API','http://127.0.0.1:9443/api/')+route,timeout=30) as r:return json.load(r)

def source_candidates(times,raw,bpm,duration,overlap_bars,loop_bars=0,loops=None,downbeat_count=None):
    bar=240/bpm
    playable=(times+((loop_bars or overlap_bars)*bar)<duration-.1)&(raw[:,52]>-20)
    if loop_bars:
        accepted={round(r['start'],6) for r in loops if r['accepted']}
        playable &= np.array([round(float(t),6) in accepted for t in times])
    else:playable &= times>=4*bar
    if downbeat_count is not None:
        positions=np.arange(len(times))*4
        playable &= positions+(loop_bars or overlap_bars)<downbeat_count
        if not loop_bars:playable &= positions>=4
    return np.where(playable)[0]

def choose(points,scores,bars,loop_bars=0):
    if not loop_bars:return [i for i in diverse_picks(points,scores,bars) if np.isfinite(scores[i])]
    selected=[]
    for i in np.argsort(-scores,kind='stable'):
        if not np.isfinite(scores[i]):break
        if all(abs(points[i,0]-points[j,0])>=loop_bars*bars[0] for j in selected):selected.append(int(i))
        if len(selected)==3:break
    return selected

def plan(source,target,method='semantic',overlap_bars=8,loop_bars=0,use_feedback=True,target_loop_bars=0):
    if method not in ['semantic','measured'] or overlap_bars not in [4,8,16]:raise ValueError('Unsupported planning configuration')
    if loop_bars not in [0,4,8]:raise ValueError('Choose a four- or eight-bar loop')
    if target_loop_bars not in [0,4,8] or target_loop_bars and (loop_bars or use_feedback):
        raise ValueError('Incoming-loop planning requires feedback off and no outgoing loop')
    tracks={t['id']:t for t in api('tracks')}
    if source==target or source not in tracks or target not in tracks:raise ValueError('Choose two different library songs')
    modelpath=HERE/'artifacts_scaled_v2/semantic_part_scorer.joblib'
    bundle=joblib.load(modelpath);cache={}
    for tid in [source,target]:
        path=HERE/'halo_features'/f'{tid}.npz'
        if not path.exists():raise ValueError('Music features for this song are still being prepared')
        d=np.load(path);config=json.loads(str(d['config']))
        if hashlib.sha256(Path(tracks[tid]['file_path']).read_bytes()).hexdigest()!=config['audio_sha256']:
            raise ValueError('Song audio changed; music features need refresh')
        current=api(f'analysis/{tid}')
        current_times=np.array([v['time_ms']/1000 for v in current['beats'] if v['beat_in_bar']==1])[::4]
        current_times=current_times[current_times<config['duration']]
        if abs(current['bpm']-config['bpm'])>1e-6 or current_times.shape!=d['times'].shape or not np.allclose(current_times,d['times'],rtol=0,atol=1e-6):
            raise ValueError('The beat grid changed; music features need refresh')
        if config['revision']!=bundle['encoder']['revision']:raise ValueError('Music encoder version mismatch')
        raw,embeddings=d['raw'],d['embeddings'];compressed=bundle['semantic_pca'].transform(embeddings)
        probabilities=functional_probabilities(bundle['functional'],np.c_[raw,compressed])
        nodes=np.c_[raw[:,:4],probabilities,raw[:,bundle['node_core_indices']],compressed[:,:16]].astype(np.float32)
        assert nodes.shape[1]==43 and np.isfinite(nodes).all()
        cache[tid]={'times':d['times'],'raw':raw,'embeddings':embeddings,'nodes':nodes,'config':config,'probabilities':probabilities,'fingerprint':str(d['fingerprint']),'downbeats':d['downbeats']}
    a,b=cache[source],cache[target];bpm=a['config']['bpm'];ratio=bpm/b['config']['bpm'];bar=240/bpm
    if not .94<=ratio<=1.06:raise ValueError('These songs need more than six percent tempo adjustment; choose a closer pair')
    loop_data=None
    if loop_bars:
        path=HERE/'halo_features'/f'{source}_loops_{loop_bars}.json'
        if not path.exists():raise ValueError('Loop analysis needs refresh')
        loop_data=json.load(open(path))
        continuity_hash=hashlib.sha256((HERE.parent/'raveform_audio_v1/artifacts/loop_region_continuity.joblib').read_bytes()).hexdigest()
        if loop_data.get('version')!='source_loop_candidates_v2_phase_matched' or loop_data.get('source')!=source or loop_data.get('loop_bars')!=loop_bars or loop_data.get('source_feature_fingerprint')!=a['fingerprint'] or loop_data.get('continuity_model_sha256')!=continuity_hash:
            raise ValueError('Loop analysis needs refresh')
    ai=source_candidates(a['times'],a['raw'],bpm,a['config']['duration'],overlap_bars,loop_bars,loop_data['candidates'] if loop_data else None,len(a['downbeats']))
    bi=np.where((b['times']+(overlap_bars+4)*bar*ratio<b['config']['duration']-.1)&(b['raw'][:,52]>-20)&(np.arange(len(b['times']))*4+overlap_bars+4<len(b['downbeats'])))[0]
    # Incoming loops are an opt-in listening experiment. Rank all eligible
    # regions after the continuity prefilter, rather than forcing a loop onto
    # one of the unconstrained top-three choices.
    target_loop_data=None
    if target_loop_bars:
        path=HERE/'halo_features'/f'{target}_loops_{target_loop_bars}.json'
        if not path.exists():raise ValueError('Incoming loop analysis needs refresh')
        target_loop_data=json.load(open(path))
        model_hash=hashlib.sha256((HERE.parent/'raveform_audio_v1/artifacts/loop_region_continuity.joblib').read_bytes()).hexdigest()
        if (target_loop_data.get('version')!='source_loop_candidates_v2_phase_matched'
            or target_loop_data.get('source')!=target or target_loop_data.get('loop_bars')!=target_loop_bars
            or target_loop_data.get('source_feature_fingerprint')!=b['fingerprint']
            or target_loop_data.get('audio_sha256')!=b['config']['audio_sha256']
            or target_loop_data.get('continuity_model_sha256')!=model_hash):
            raise ValueError('Incoming loop analysis needs refresh')
        accepted={round(row['start'],6) for row in target_loop_data['candidates'] if row['accepted']}
        bi=np.array([i for i in bi if round(float(b['times'][i]),6) in accepted],dtype=int)
    if not len(ai) and loop_bars:raise ValueError('No source regions pass the loop checks for this length; try a shorter loop or a blend')
    if not len(ai) or not len(bi):raise ValueError('Not enough playable audio for this overlap length')
    ia,ib=np.meshgrid(ai,bi,indexing='ij');ia=ia.ravel();ib=ib.ravel()
    x=pair_features(a['nodes'][ia],b['nodes'][ib],a['embeddings'][ia],b['embeddings'][ib])
    logits=bundle['pair_head'].decision_function(x)
    features=json.load(open(BRIDGE/'review/measured-audio.json'));index=SectionIndex(features)
    def measured_rows(tid,times):
        rows=[r for r in index.rows if r['track_id']==tid]
        if not rows:raise ValueError('Measured analysis for this song is not ready')
        return [min(rows,key=lambda r:abs(r['start_seconds']-float(t))) for t in times]
    ar,br=measured_rows(source,a['times'][ai]),measured_rows(target,b['times'][bi])
    ac=np.array([r['_core'] for r in ar]);bc=np.array([r['_core'] for r in br])
    measured=-np.mean((ac[:,None,:]-bc[None,:,:])**2,axis=2)
    if all(r['_texture'] is not None for r in ar+br):
        at=np.array([r['_texture'] for r in ar]);bt=np.array([r['_texture'] for r in br])
        measured=.55*measured-.45*np.mean((at[:,None,:]-bt[None,:,:])**2,axis=2)
    compatible=np.array([[key_relation(r.get('key'),s.get('key'))=='compatible' for s in br] for r in ar])
    mask=compatible.ravel();harmonic_filter=bool(mask.any())
    if not harmonic_filter:mask=np.ones(len(ia),bool)
    base_scores=logits.copy() if method=='semantic' else measured.ravel().copy()
    points=np.c_[a['times'][ia],b['times'][ib]];bars=np.array([240/bpm,240/b['config']['bpm']])
    preferences=[];feedback_policy=dict(enabled=bool(use_feedback))
    if use_feedback:
        preferences,evidence=part_preferences.load(BRIDGE/'review/review.sqlite',HERE.parent/'listening_review_v1/legacy-replay-bindings.json',
            source,target,a['config']['audio_sha256'],b['config']['audio_sha256'],overlap_bars,loop_bars)
        feedback_policy.update(evidence)
    effect=part_preferences.effects(points,bars,preferences)
    liked=np.asarray(effect['liked'],bool);avoided=np.asarray(effect['avoided'],bool);exceptions=liked&~mask
    allowed=(mask|liked)&~avoided;scores=base_scores.copy()
    if liked.any() and allowed.any():
        lo,hi=float(base_scores[allowed].min()),float(base_scores[allowed].max());span=max(1.,hi-lo)
        strength=np.asarray(effect['strength'])
        scores[liked&allowed]=hi+1+span*(1+strength[liked&allowed])+.01*(base_scores[liked&allowed]-lo)/span
    scores[~allowed]=-np.inf
    feedback_policy.update(liked_anchor_points=int(liked.sum()),explicit_part_rejections=int(avoided.sum()),key_filter_exception_points=int(exceptions.sum()))
    picks=choose(points,scores,bars,loop_bars)
    if not picks:raise ValueError('No feasible section pair')
    def selected(chosen):
        sa,sb=int(ia[chosen]),int(ib[chosen]);start,end=float(a['times'][sa]),float(b['times'][sb])
        qa,qb=ar[list(ai).index(sa)],br[list(bi).index(sb)]
        source_kind=bundle['kinds'][int(np.argmax(a['probabilities'][sa]))]
        target_kind=bundle['kinds'][int(np.argmax(b['probabilities'][sb]))]
        selection={'source_track':source,'target_track':target,'source_title':tracks[source]['title'],'target_title':tracks[target]['title'],
        'source_start_seconds':start,'target_start_seconds':end,'mix_bpm':bpm,'target_tempo_factor':ratio,'overlap_bars':overlap_bars,
        'source_section':{'name':'estimated '+source_kind,'start':start,'end':start+(loop_bars or overlap_bars)*bar},
        'target_section':{'name':'estimated '+target_kind,'start':end,'end':end+overlap_bars*bar*ratio},
        'model_logit':float(logits[chosen]),'ranking_score':float(scores[chosen]),'selector':method,
        'harmonic_key_compatible':key_relation(qa.get('key'),qb.get('key'))=='compatible',
        'measured_source_key':qa.get('key'),'measured_target_key':qb.get('key'),
        'source_function_probabilities':a['probabilities'][sa].tolist(),'target_function_probabilities':b['probabilities'][sb].tolist()}
        if effect['evidence'][chosen]:
            preference=effect['evidence'][chosen]
            selection['listening_preference']={k:preference[k] for k in ['kind','review_id','supporting_review_ids','binding','snap_distance_bars','new_render_approved']}
            selection['listening_preference']['key_filter_exception']=bool(exceptions[chosen])
            selection['base_ranking_score']=float(base_scores[chosen])
        row=None
        if loop_data:
            row=next(r for r in loop_data['candidates'] if abs(r['start']-start)<1e-5)
            row=dict(row,continuity_model_sha256=loop_data['continuity_model_sha256'],source_feature_fingerprint=a['fingerprint'],
                model_target='Same annotated functional region; not human-rated loop quality')
        return {'selection':selection,'loop_candidate':row}
    first=selected(picks[0]);selection=first['selection']
    result={'selection':selection,'source_audio_sha256':a['config']['audio_sha256'],'target_audio_sha256':b['config']['audio_sha256'],
        'source_duration_seconds':a['config']['duration'],'target_duration_seconds':b['config']['duration'],
        'source_bpm':bpm,'target_bpm':b['config']['bpm'],'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),
        'selector_version':'semantic_function_pair_v2_scale_floor_001_listening_memory_v1','feasible_candidates':len(ia),'harmonic_candidates':int(mask.sum()),
        'planning_version':'loop_conditioned_v1' if loop_bars else 'blend_v1','requested_loop_bars':loop_bars,
        'loop_candidate':first['loop_candidate'],'alternatives':[selected(i) for i in picks[1:]],
        'harmonic_filter_applied':harmonic_filter,'top_candidates':[{'source_start':float(points[i,0]),'target_start':float(points[i,1]),'score':float(scores[i])} for i in picks],
        'listening_policy':feedback_policy,
        'scope':'Experimental candidate for listening; model trained on observed DJ overlaps, not verified mix quality. No deck commands.'}
    if target_loop_data:
        result.update(planning_version='incoming_loop_conditioned_lab_v1',requested_target_loop_bars=target_loop_bars,
            target_loop_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=int,required=True);parser.add_argument('--target',type=int,required=True)
    parser.add_argument('--method',choices=['semantic','measured'],default='semantic');parser.add_argument('--bars',type=int,default=8);parser.add_argument('--loop-bars',type=int,default=0);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ignore-feedback',action='store_true')
    args=parser.parse_args();result=plan(args.source,args.target,args.method,args.bars,args.loop_bars,use_feedback=not args.ignore_feedback)
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp.json');temp.write_text(json.dumps(result,indent=2));temp.replace(args.output)
    print(json.dumps({'selected':result['selection'],'feasible_candidates':result['feasible_candidates']}),flush=True)

if __name__=='__main__':main()
