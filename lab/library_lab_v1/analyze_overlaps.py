"""Librosa diagnostics over exact, reconstructed audition overlaps and loop repeats.

These descriptors measure acoustics. They are not calibrated musical-quality or
vocal-clash scores. Existing audio/ratings/manifests remain immutable.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import sys,json,time,wave,hashlib,copy
import numpy as np,librosa,joblib
from scipy.signal import resample_poly
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from render_group import r,mix,sha,write
SR=48000;FS=22050;HOP=256;FFT=2048

def binding(p):
 return hashlib.sha256(json.dumps({k:p.get(k) for k in ['selection','engine','audio','eq_enabled','comparison']},sort_keys=True).encode()).hexdigest()
def corr(a,b):
 if len(a)<4 or np.std(a)<1e-8 or np.std(b)<1e-8:return None
 return float(np.clip(np.corrcoef(a,b)[0,1],-1,1))
def cosine(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b)
 return float(np.clip(np.dot(a,b)/d,0,1)) if d>1e-9 else None
def features(stereo):
 cache=HERE/'overlap-feature-cache';cache.mkdir(exist_ok=True)
 digest=hashlib.sha256(stereo.tobytes()+f'{FS}:{HOP}:{FFT}:{librosa.__version__}'.encode()).hexdigest();path=cache/(digest+'.npz')
 if path.exists():
  with np.load(path) as saved:return {key:saved[key] for key in saved.files}
 mono=resample_poly(np.mean(stereo,axis=1),147,320).astype(np.float32)
 D=abs(librosa.stft(mono,n_fft=FFT,hop_length=HOP));P=D*D
 harmonic,_=librosa.decompose.hpss(D,margin=(2.,2.))
 chroma=librosa.feature.chroma_stft(S=harmonic**2,sr=FS,n_fft=FFT,hop_length=HOP,tuning=0,norm=1)
 mel=librosa.feature.melspectrogram(S=P,sr=FS,n_fft=FFT,n_mels=40)
 onset=librosa.onset.onset_strength(S=librosa.power_to_db(mel,ref=1.),sr=FS,hop_length=HOP,n_fft=FFT)
 mfcc=librosa.feature.mfcc(S=librosa.power_to_db(mel,ref=1.),n_mfcc=13)[1:]
 contrast=librosa.feature.spectral_contrast(S=D,sr=FS,n_fft=FFT,n_bands=5)
 freqs=librosa.fft_frequencies(sr=FS,n_fft=FFT)
 bass=P[freqs<=180].sum(axis=0);energy=P.sum(axis=0)
 result=dict(chroma=chroma,onset=onset,mfcc=mfcc,contrast=contrast,bass=bass,energy=energy,time=librosa.frames_to_time(np.arange(P.shape[1]),sr=FS,hop_length=HOP))
 np.savez_compressed(path,**result);return result
def compare(a,b,start,end,bpm):
 mask=(a['time']>=start)&(a['time']<end)
 pa,pb=a['energy'][mask],b['energy'][mask];maximum=np.maximum(pa,pb)
 audible=(np.minimum(pa,pb)>=maximum*10**(-12/10))&(maximum>1e-8)
 f=np.flatnonzero(mask);both=f[audible]
 # No tonal/rhythmic fit is claimed where one stream has faded away.
 weights=np.minimum(pa,pb);total=weights.sum()
 active=both
 tone=cosine(np.average(a['chroma'][:,active],axis=1,weights=a['energy'][active]),np.average(b['chroma'][:,active],axis=1,weights=b['energy'][active])) if len(active) else None
 frame_cosine=None
 if len(active):
  ca,cb=a['chroma'][:,active],b['chroma'][:,active]
  norms=np.linalg.norm(ca,axis=0)*np.linalg.norm(cb,axis=0);valid=norms>1e-9
  if valid.any():frame_cosine=float(np.average(np.sum(ca[:,valid]*cb[:,valid],axis=0)/norms[valid],weights=np.minimum(a['energy'][active][valid],b['energy'][active][valid])))
 bass_a,bass_b=a['bass'][mask],b['bass'][mask]
 bass_coexist=float(np.sum(2*np.minimum(bass_a,bass_b))/max(np.sum(bass_a+bass_b),1e-12))
 onset_a,onset_b=a['onset'][mask],b['onset'][mask];zero=corr(onset_a,onset_b)
 maxlag=round((.5*60/bpm)*FS/HOP);lagged=[]
 for lag in range(-maxlag,maxlag+1):
  x,y=(onset_a[-lag:],onset_b[:lag]) if lag<0 else (onset_a[:-lag],onset_b[lag:]) if lag else (onset_a,onset_b)
  value=corr(x,y)
  if value is not None:lagged.append((value,lag))
 best=max(lagged,key=lambda v:(v[0],-abs(v[1]))) if lagged else None
 # MFCC distances have units and no fitted threshold; omit loudness coefficient 0.
 texture=float(np.sqrt(np.mean((np.mean(a['mfcc'][:,active],axis=1)-np.mean(b['mfcc'][:,active],axis=1))**2))) if len(active) else None
 return dict(harmonic_chroma_cosine=tone,frame_harmonic_chroma_cosine=frame_cosine,coaudible_fraction=float(np.mean(audible)),simultaneous_bass_ratio=bass_coexist,onset_correlation=zero,best_onset_lag_ms=float(best[1]*HOP/FS*1000) if best else None,best_lag_onset_correlation=best[0] if best else None,mfcc_distance=texture,spectral_contrast_distance=float(np.sqrt(np.mean((np.mean(a['contrast'][:,active],axis=1)-np.mean(b['contrast'][:,active],axis=1))**2))) if len(active) else None)
def rebuild(p,tracks,gate):
 s=p['selection'];bpm=s['mix_bpm'];bar=240/bpm;npre=round(4*bar*SR);nover=round(s['overlap_bars']*bar*SR)
 source,target=tracks[s['source_track']],tracks[s['target_track']]
 assert sha(source['file_path'])==p['source_audio_sha256'] and sha(target['file_path'])==p['target_audio_sha256']
 a,_=r.aligned_stem(source,s['source_start_seconds'],4,4+s['overlap_bars'],bpm,gate,npre+nover)
 b,_=r.aligned_stem(target,s['target_start_seconds'],0,s['overlap_bars']+4,bpm,gate,nover+npre)
 gain=p['engine']['incoming_level_gain'];b*=gain
 loop=p['engine'].get('loop');target_loop=p['engine'].get('target_loop')
 if loop:
  period=loop['period_frames'];unit,_=r.aligned_stem(source,s['source_start_seconds'],0,loop['bars'],bpm,gate,period)
  a=np.concatenate([a[:npre],r.repeat_loop(unit,nover//period,sample_rate=SR)])
 if target_loop:
  period=target_loop['period_frames'];unit,_=r.aligned_stem(target,s['target_start_seconds'],0,target_loop['bars'],bpm,gate,period)
  continuation=b[period:period+npre].copy();n=round(.003*SR);continuation[:n]*=np.linspace(0,1,n)[:,None]
  b=np.concatenate([r.repeat_loop(unit*gain,nover//period,sample_rate=SR),continuation])
 mode=p['engine']['strategy'];reference=mix(a,b,npre,nover,mode);outgain=p['audio']['gain_applied'];reference*=outgain;n=round(.01*SR);reference[:n]*=np.linspace(0,1,n)[:,None];reference[-n:]*=np.linspace(1,0,n)[:,None]
 with wave.open(p['audio']['path']) as f:published=np.frombuffer(f.readframes(f.getnframes()),'<i2').reshape(-1,2)
 assert sha(p['audio']['path'])==p['audio']['sha256']
 pcm=(reference*32767).astype('<i2');assert pcm.shape==published.shape
 difference=int(np.max(abs(pcm.astype(np.int32)-published.astype(np.int32))));assert difference==0,('Audio reconstruction mismatch',difference)
 # Isolate each actual rendered contribution through the linear mix engine.
 ca=mix(a,np.zeros_like(b),npre,nover,mode)[npre:npre+nover]*outgain
 cb=mix(np.zeros_like(a),b,npre,nover,mode)[npre:npre+nover]*outgain
 assert np.max(abs(ca+cb-reference[npre:npre+nover]))<1e-7
 return a[npre:]*outgain,b[:nover]*outgain,ca,cb,difference

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());tracks={t['id']:t for t in r.api('tracks')};gate=joblib.load(HERE.parent/'part_alignment_v1/artifacts/beat_repair_gate.joblib')
 report=dict(version='exact_overlap_librosa_v1',created=time.time(),librosa_version=librosa.__version__,analysis_sample_rate=FS,fft_size=FFT,hop_length=HOP,metric_version='simultaneous_frame_chroma_and_rendered_onsets_v2',coaudibility_definition='Both rendered streams within 12 dB of each other, above numerical energy floor; a diagnostic visibility rule, not a perceptual threshold',onset_visibility_policy='Use the full rendered bar for attack comparisons: a displaced attack must not disappear merely because the other stream is quiet at that instant.',onset_lag_definition='Exploratory cross-correlation over ±half a beat; positive means incoming onset envelope lags outgoing. Does not establish the downbeat or instruct a cue correction.',scope='Actual repeated/aligned audio, raw region comparison and fade/EQ/echo-weighted contributions. Diagnostic descriptors; no calibrated good/bad or vocal-clash judgement.',method_sources=['https://librosa.org/doc/0.11.0/generated/librosa.feature.chroma_stft.html','https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html'],previews={})
 for spec in protocol['experiments']:
  for p in json.loads((HERE/(spec['id']+'.json')).read_text())['previews']:
   digest=p['audio']['sha256'];s=p['selection'];bar=240/s['mix_bpm'];print('ANALYZING',spec['id'],p['comparison']['variant'],flush=True)
   a,b,ca,cb,difference=rebuild(p,tracks,gate);raw_a,raw_b=features(a),features(b);eff_a,eff_b=features(ca),features(cb);bars=[]
   for i in range(s['overlap_bars']):
    start,end=i*bar,(i+1)*bar;lo,hi=round(start*SR),min(len(ca),round(end*SR));segment=ca[lo:hi]+cb[lo:hi]
    bars.append(dict(bar=i+1,preview_seconds=4*bar+start,raw=compare(raw_a,raw_b,start,end,s['mix_bpm']),rendered=compare(eff_a,eff_b,start,end,s['mix_bpm']),mix_rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(segment**2)),1e-12))),source_loop_cycle=i//p['engine']['loop']['bars']+1 if p['engine'].get('loop') else None,target_loop_cycle=i//p['engine']['target_loop']['bars']+1 if p['engine'].get('target_loop') else None))
   entry=dict(audio_sha256=digest,preview_snapshot_sha256=binding(p),group_id=p['comparison']['group_id'],variant=p['comparison']['variant'],source_track=s['source_track'],target_track=s['target_track'],label=p['comparison_label'],reconstruction_pcm_max_difference=difference,overlap_bars=s['overlap_bars'],summary=dict(raw=compare(raw_a,raw_b,0,s['overlap_bars']*bar,s['mix_bpm']),rendered=compare(eff_a,eff_b,0,s['overlap_bars']*bar,s['mix_bpm'])),bars=bars)
   report['previews'][digest]=entry;write(HERE/'overlap-analysis-progress.json',report)
 report.update(completed=time.time(),clips=len(report['previews']),bars=sum(len(p['bars']) for p in report['previews'].values()),all_reconstructions_bit_exact=True,script_sha256=sha(__file__));assert report['clips']==17;write(HERE/'overlap-analysis.json',report);print('OVERLAP_ANALYSIS_READY',report['clips'],report['bars'],flush=True)
if __name__=='__main__':main()
