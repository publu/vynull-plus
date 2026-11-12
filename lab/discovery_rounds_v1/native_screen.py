import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import sys,json,time
import numpy as np,librosa
from scipy.signal import resample_poly
from scipy.stats import uniform
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN/'library_lab_v1'))
from deep_merge_analysis import r,period,sha,write,FS,HOP

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());tracks={t['id']:t for t in r.api('tracks')};path=HERE/'native-screen.json';report=json.loads(path.read_text()) if path.exists() else dict(created=time.time(),tracks={})
 ids=sorted({p[k] for p in protocol['pairs'] for k in ['source','target']})
 for tid in ids:
  t=tracks[tid];analysis=r.api(f'analysis/{tid}');digest=sha(t['file_path']);cached=report['tracks'].get(str(tid))
  if cached and cached['audio_sha256']==digest and cached['tagged_bpm']==analysis['bpm']:continue
  rows=[]
  for start in protocol['native_check']['offsets']:
   y=r.decode(t['file_path'],start,start+protocol['native_check']['excerpts_seconds']).mean(axis=1);y=resample_poly(y,147,320).astype(np.float32)
   env=librosa.onset.onset_strength(y=y,sr=FS,hop_length=HOP,n_fft=2048,n_mels=64)
   wide=float(np.asarray(librosa.feature.tempo(onset_envelope=env,sr=FS,hop_length=HOP,prior=uniform(50,170),max_tempo=220,ac_size=10)).reshape(-1)[0]);near=period(env,analysis['bpm']);near['error_percent']=float(100*(near['bpm']/analysis['bpm']-1)) if near['bpm'] else None
   good=bool(near['bpm'] and near['correlation']>=.25 and abs(near['error_percent'])<=2);rows.append(dict(start_seconds=start,seconds=len(y)/FS,wide_pulse_bpm=wide,near=near,near_grid_supported=good))
  report['tracks'][str(tid)]=dict(id=tid,title=t['title'],artist=t.get('artist'),audio_sha256=digest,tagged_bpm=analysis['bpm'],excerpts=rows,eligible=any(x['near_grid_supported'] for x in rows),scope='Pulse evidence near stored grid from two original excerpts; ambiguous meters retained as uncertainty, not silently corrected')
  write(path,report);print('NATIVE_SCREEN',tid,report['tracks'][str(tid)]['eligible'],flush=True)
 report.update(completed=time.time(),total_tracks=len(ids),eligible_tracks=sum(v['eligible'] for v in report['tracks'].values()));write(path,report)
 print('NATIVE_SCREEN_DONE',report['eligible_tracks'],len(ids),flush=True)
if __name__=='__main__':main()
