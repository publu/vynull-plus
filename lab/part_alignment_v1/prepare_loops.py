"""Cache loop-feasible source regions before ranking a loop-assisted handoff."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,argparse,json,hashlib,urllib.request,fcntl,time
from pathlib import Path
import numpy as np,joblib

HERE=Path(__file__).parent;TRAIN=HERE.parent;FEATURES=TRAIN/'semantic_pair_v1/halo_features'
sys.path.insert(0,str(TRAIN/'semantic_pair_v1'))
from audio_context import extract
from loop_boundaries import metrics
MODEL=TRAIN/'raveform_audio_v1/artifacts/loop_region_continuity.joblib'
VERSION='source_loop_candidates_v2_phase_matched'

def api(route):
    with urllib.request.urlopen(os.environ.get('VYNULL_API','http://127.0.0.1:9443/api/')+route,timeout=30) as r:return json.load(r)

def prepare(source,bars):
    if bars not in [4,8]:raise ValueError('Choose a four- or eight-bar loop')
    path=FEATURES/f'{source}.npz'
    if not path.exists():raise ValueError('Music features for this song are still being prepared')
    data=np.load(path);config=json.loads(str(data['config']));fingerprint=str(data['fingerprint'])
    tracks={t['id']:t for t in api('tracks')}
    if source not in tracks:raise ValueError('Song is not in the library')
    audio=Path(tracks[source]['file_path']);digest=hashlib.sha256(audio.read_bytes()).hexdigest()
    if digest!=config['audio_sha256']:raise ValueError('Song audio changed; music features need refresh')
    model_hash=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    wanted={'version':VERSION,'source_feature_fingerprint':fingerprint,'continuity_model_sha256':model_hash,'loop_bars':bars}
    dest=FEATURES/f'{source}_loops_{bars}.json'
    lock=dest.with_suffix('.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if dest.exists():
            cached=json.load(open(dest))
            if all(cached.get(k)==v for k,v in wanted.items()):return dict(cached,cache_hit=True)
        bpm=config['bpm'];bar=240/bpm
        valid=(data['times']+bars*bar<config['duration']-.1)&(np.arange(len(data['times']))*4+bars<len(data['downbeats']))
        times=data['times'][valid]
        if not len(times):raise ValueError('This song has no complete loop of the requested length')
        _,vectors,y,sr=extract(audio,times,bpm,config['duration'],loops=[(float(t),bars) for t in times],return_audio=True)
        model=joblib.load(MODEL);probabilities=model.predict_proba(vectors)[:,1];rows=[]
        for stamp,values,probability in zip(times,vectors,probabilities):
            start=float(stamp);end=start+bars*bar;begin=values[2:50];finish=values[50:98]
            a,b=begin[-12:],finish[-12:];tonal=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-10))
            unit=y[round(start*sr):round(end*sr)];boundary=metrics(unit,sr,bar)
            rms=boundary['rms'];level=boundary['boundary_level_difference_db']
            accepted=bool(probability>=.8 and tonal>=.5 and level<=6 and rms>=.005)
            rows.append({'start':start,'end':end,'bars':bars,'structural_continuity_probability':float(probability),
                'boundary_chroma_cosine':tonal,**boundary,'accepted':accepted})
        if hashlib.sha256(audio.read_bytes()).hexdigest()!=digest:raise ValueError('Song audio changed; music features need refresh')
        result=dict(wanted,source=source,audio_sha256=digest,bpm=bpm,duration=config['duration'],created=time.time(),
            accepted_count=sum(r['accepted'] for r in rows),candidates=rows,
            criteria={'structural_continuity_min':.8,'boundary_chroma_cosine_min':.5,'boundary_level_difference_db_max':6,'rms_min':.005},
            scope='Loop-region prefilter from a learned structural-continuity proxy and measured boundary checks; musical quality remains unverified')
        temp=dest.with_suffix('.tmp.json');temp.write_text(json.dumps(result,indent=2));temp.replace(dest)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=int,required=True);parser.add_argument('--bars',type=int,required=True)
    args=parser.parse_args();result=prepare(args.source,args.bars)
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'}),flush=True)

if __name__=='__main__':main()
