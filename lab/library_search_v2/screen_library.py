"""Parallel original-audio pulse checks for every current library track."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,time,concurrent.futures,traceback
import numpy as np,librosa
from scipy.signal import resample_poly
from scipy.stats import uniform
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN/'library_lab_v1'))
from deep_merge_analysis import r,period,sha,write,FS,HOP
from summarize_deep import original_check

def check(track):
 tid=track['id'];analysis=r.api(f'analysis/{tid}');bpm=float(analysis['bpm']);digest=sha(track['file_path']);duration=float(track['duration']);length=min(20.,duration/3);starts=[min(25.,duration*.15),min(85.,max(0.,duration-length-.2))]
 path=HERE/'native-cache'/f'{tid}.json';config=dict(version='library_native_v2',audio_sha256=digest,tagged_bpm=bpm,length=length,starts=starts)
 if path.exists():
  cached=json.loads(path.read_text())
  if cached.get('config')==config:return dict(cached,cache_hit=True)
 # Exact prior twenty-second probes can be reused without changing evidence.
 oldpath=TRAIN/'discovery_rounds_v1/native-screen.json';old=json.loads(oldpath.read_text())['tracks'].get(str(tid)) if oldpath.exists() else None
 if old and old['audio_sha256']==digest and old['tagged_bpm']==bpm and [x['start_seconds'] for x in old['excerpts']]==starts and all(abs(x['seconds']-length)<.001 for x in old['excerpts']):rows=old['excerpts'];reused=True
 else:
  rows=[];reused=False
  for start in starts:
   y=r.decode(track['file_path'],start,start+length).mean(axis=1);y=resample_poly(y,147,320).astype(np.float32);env=librosa.onset.onset_strength(y=y,sr=FS,hop_length=HOP,n_fft=2048,n_mels=64)
   wide=float(np.asarray(librosa.feature.tempo(onset_envelope=env,sr=FS,hop_length=HOP,prior=uniform(50,170),max_tempo=220,ac_size=min(10,length/2))).reshape(-1)[0]);near=period(env,bpm);error=100*(near['bpm']/bpm-1) if near['bpm'] else None;good=bool(error is not None and abs(error)<=2 and near['correlation']>=.25)
   rows.append(dict(start_seconds=start,seconds=len(y)/FS,wide_pulse_bpm=wide,near=near,near_grid_supported=good))
 result=dict(config=config,id=tid,title=track['title'],artist=track.get('artist'),audio_sha256=digest,tagged_bpm=bpm,excerpts=rows,eligible=any(x['near_grid_supported'] for x in rows),reused_prior_evidence=reused)
 native=original_check(result);result['grid_check']=native;count=sum(x['near_grid_supported'] for x in rows);result['status']='grid_warning' if native['grid_warning'] else 'supported' if count==2 else 'partial' if count else 'uncertain';write(path,result);return dict(result,cache_hit=False)
def main():
 tracks=r.api('tracks');(HERE/'native-cache').mkdir(exist_ok=True);report=dict(created=time.time(),version='library_native_v2',total_tracks=len(tracks),tracks={},errors={});started=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  pending={pool.submit(check,t):t for t in tracks}
  for future in concurrent.futures.as_completed(pending):
   track=pending[future]
   try:result=future.result();report['tracks'][str(track['id'])]=result;print('LIBRARY_PULSE',track['id'],result['status'],flush=True)
   except Exception as e:report['errors'][str(track['id'])]=str(e);traceback.print_exc()
   report['elapsed_seconds']=time.perf_counter()-started;write(HERE/'native-progress.json',report)
 report.update(completed=time.time());write(HERE/'native-screen.json',report);print('LIBRARY_PULSE_DONE',len(report['tracks']),len(report['errors']),report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
