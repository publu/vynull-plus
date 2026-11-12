"""All-library section retrieval. Candidates remain unchecked until audition/vocal audit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,time,hashlib,re
import numpy as np
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent;sys.path.insert(0,str(TRAIN.parent/'review'));sys.path.insert(0,str(TRAIN/'part_alignment_v1'))
from section_matches import SectionIndex,camelot
import render_live as r

def main():
 started=time.perf_counter();tracks=r.api('tracks');byid={t['id']:t for t in tracks};native=json.loads((HERE/'native-screen.json').read_text());measured_path=TRAIN.parent/'review/measured-audio.json';measured=json.loads(measured_path.read_text());index=SectionIndex(measured);points=[];records=[]
 for t in tracks:
  tid=t['id'];record=dict(id=tid,title=t['title'],artist=t.get('artist'),bpm=t.get('bpm'),key=t.get('key'),native_status=native['tracks'].get(str(tid),{}).get('status','unchecked'));path=TRAIN/'semantic_pair_v1/halo_features'/f'{tid}.npz';rows=[r for r in index.rows if r['track_id']==tid]
  if not path.exists() or not rows:record.update(search_status='missing_section_features');records.append(record);continue
  n=native['tracks'].get(str(tid),{})
  if n.get('status') not in ['supported','partial']:record.update(search_status='needs_grid_or_pulse_review');records.append(record);continue
  with np.load(path) as data:
   config=json.loads(str(data['config']));times=data['times'];downbeats=data['downbeats']
  if config['audio_sha256']!=n['audio_sha256'] or abs(config['bpm']-n['tagged_bpm'])>1e-6:record.update(search_status='stale_features');records.append(record);continue
  bar=240/config['bpm'];count=0
  for i,stamp in enumerate(times):
   source_ok=bool(stamp>=4*bar and i*4>=4 and stamp+8*bar<config['duration']-.1 and i*4+8<len(downbeats))
   target_ok=bool(stamp+12*bar<config['duration']-.1 and i*4+12<len(downbeats))
   if not source_ok and not target_ok:continue
   row=min(rows,key=lambda row:abs(row['start_seconds']-float(stamp)))
   if row['_texture'] is None:continue
   k=camelot(row.get('key'));points.append(dict(track=tid,start=float(stamp),bpm=config['bpm'],key=row.get('key'),camelot=list(k) if k else None,source_eligible=source_ok,target_eligible=target_ok,core=row['_core'],texture=row['_texture'],rms=row['mean_rms_dbfs'],label=row['label']))
   count+=1
  record.update(search_status='ready' if count else 'no_complete_regions',candidate_regions=count);records.append(record)
 X=np.asarray([p['core'] for p in points],np.float32);T=np.asarray([p['texture'] for p in points],np.float32);ids=np.array([p['track'] for p in points]);bpms=np.array([p['bpm'] for p in points]);target_ok=np.array([p['target_eligible'] for p in points]);rms=np.array([p['rms'] for p in points]);kn=np.array([p['camelot'][0] if p['camelot'] else -99 for p in points]);km=np.array([p['camelot'][1] if p['camelot'] else '?' for p in points]);pairs=[]
 for t in tracks:
  tid=t['id'];source_indices=np.array([i for i,p in enumerate(points) if p['track']==tid and p['source_eligible']],dtype=int)
  if not len(source_indices):continue
  A=X[source_indices];B=T[source_indices];distance=.55*np.maximum(0,(np.sum(A*A,axis=1)[:,None]+np.sum(X*X,axis=1)[None,:]-2*A@X.T)/X.shape[1])+.45*np.maximum(0,(np.sum(B*B,axis=1)[:,None]+np.sum(T*T,axis=1)[None,:]-2*B@T.T)/T.shape[1]);ratio=t['bpm']/bpms
  key_distance=abs(kn[source_indices,None]-kn[None,:]);key_distance=np.minimum(key_distance,12-key_distance);compatible=(kn[source_indices,None]>0)&(kn[None,:]>0)&((key_distance==0)|((key_distance==1)&(km[source_indices,None]==km[None,:])))
  allowed=compatible&target_ok[None,:]&(ids[None,:]!=tid)&(ratio[None,:]>=.94)&(ratio[None,:]<=1.06);delta=bpms-t['bpm'];distance+=.15*(delta[None,:]/4)**2+.1*abs(rms[source_indices,None]-rms[None,:])/6;distance[~allowed]=np.inf
  for target in sorted(set(ids[(ratio>=.94)&(ratio<=1.06)&target_ok])):
   target=int(target)
   if target==tid:continue
   a,b=byid[tid],byid[target]
   if (str(a.get('artist')).casefold(),a['title'].casefold())==(str(b.get('artist')).casefold(),b['title'].casefold()):continue
   if native['tracks'][str(tid)]['audio_sha256']==native['tracks'][str(target)]['audio_sha256']:continue
   dest_indices=np.where(ids==target)[0];values=distance[:,dest_indices];flat=np.argsort(values.ravel(),kind='stable');choices=[]
   for offset in flat:
    si,ti=np.unravel_index(offset,values.shape);score=float(values[si,ti])
    if not np.isfinite(score):break
    source,target_point=points[source_indices[si]],points[dest_indices[ti]]
    if any(abs(source['start']-c['source_start_seconds'])<4*240/t['bpm'] and abs(target_point['start']-c['target_start_seconds'])<4*240/b['bpm'] for c in choices):continue
    choices.append(dict(source_start_seconds=source['start'],target_start_seconds=target_point['start'],source_key=source['key'],target_key=target_point['key'],distance=score,source_label=source['label'],target_label=target_point['label']))
    if len(choices)==3:break
   if choices:pairs.append(dict(source=tid,target=target,distance=choices[0]['distance'],tempo_factor=float(t['bpm']/b['bpm']),sections=choices,status='candidate_unchecked',vocal_status='unchecked'))
  print('CATALOG_SOURCE',tid,'pairs',sum(p['source']==tid for p in pairs),flush=True)
 pairs.sort(key=lambda p:(p['source'],p['distance'],p['target']));result=dict(version='library_catalog_v2',created=time.time(),total_tracks=len(tracks),measured_tracks=len(measured),candidate_regions=len(points),tracks=records,ordered_pairs=len(pairs),pairs=pairs,elapsed_seconds=time.perf_counter()-started,native_sha256=hashlib.sha256((HERE/'native-screen.json').read_bytes()).hexdigest(),measured_sha256=hashlib.sha256(measured_path.read_bytes()).hexdigest(),scope='All eligible directed track pairs from cached local section descriptors, tempo bounds and local key relationships. Retrieval only; actual beat/vocal compatibility remains unchecked until rendered analysis.')
 temp=HERE/'catalog.tmp.json';temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(HERE/'catalog.json');print('CATALOG_READY',len(tracks),len(pairs),len(points),result['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
