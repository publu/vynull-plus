"""Investigate drum pulse, drift, local keys and tuning on verified real overlaps.
Screening rules are descriptive heuristics, not trained listening-quality labels.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,librosa,joblib
from scipy.signal import resample_poly
from scipy.stats import uniform,theilslopes
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent.parent))
from analyze_overlaps import rebuild,binding,SR
from render_group import r,sha,write
from section_matches import key_relation,camelot
FS=22050;HOP=128;FFT=2048
NAMES=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
MAJOR=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def wrap(x):return (np.asarray(x)+.5)%1-.5
def key_estimate(chroma,weights=None):
 profile=np.average(chroma,axis=1,weights=weights) if weights is not None and np.sum(weights)>0 else np.mean(chroma,axis=1)
 if np.std(profile)<1e-6:return dict(key=None,confident=False,top=[],margin=0)
 rows=[]
 for suffix,template in [('',MAJOR),('m',MINOR)]:
  for root in range(12):rows.append(dict(key=NAMES[root]+suffix,correlation=float(np.corrcoef(profile,np.roll(template,root))[0,1])))
 rows.sort(key=lambda x:-x['correlation']);margin=rows[0]['correlation']-rows[1]['correlation']
 return dict(key=rows[0]['key'],confident=bool(rows[0]['correlation']>=.6 and margin>=.08),top=rows[:3],margin=float(margin),method='Krumhansl major/minor profile correlation on harmonic CQT chroma; confidence is a screening rule, not calibrated probability')
def phase(env,t,bpm):
 if len(env)<4 or np.sum(env)<1e-8:return dict(phase_beats=None,concentration=0.)
 z=np.sum(env*np.exp(2j*np.pi*t*bpm/60))/np.sum(env)
 return dict(phase_beats=float(np.angle(z)/(2*np.pi)),concentration=float(abs(z)))
def period(env,nominal):
 # Local pulse estimate deliberately searches a region, rather than setting the
 # answer to the renderer's BPM. Wide unconstrained aliases are reported too.
 centered=np.asarray(env)-np.mean(env);n=len(centered)
 lo=max(2,int(FS/HOP*60/(nominal*1.12)));hi=min(n//3,int(np.ceil(FS/HOP*60/(nominal*.88))))
 if hi<=lo or np.linalg.norm(centered)<1e-8:return dict(bpm=None,correlation=0.)
 ac=librosa.autocorrelate(centered,max_size=hi+2)
 energy=np.cumsum(np.r_[0,centered**2]);lags=np.arange(lo,hi+1)
 den=np.sqrt((energy[n]-energy[lags])*(energy[n-lags]-energy[0]));values=ac[lags]/np.maximum(den,1e-12)
 k=int(np.argmax(values));lag=float(lags[k]);coefficient=float(values[k])
 if 0<k<len(values)-1:
  left,center,right=values[k-1:k+2];d=left-2*center+right
  if d<0:lag+=float(np.clip(.5*(left-right)/d,-.5,.5))
 return dict(bpm=float(60*FS/HOP/lag),correlation=coefficient,search_percent=12)
def percussion_features(stereo):
 cache=HERE/'deep-feature-cache';cache.mkdir(exist_ok=True)
 digest=hashlib.sha256(stereo.tobytes()+b'deep-v1-h128-cqt-tuning').hexdigest();path=cache/(digest+'.npz')
 if path.exists():
  with np.load(path) as saved:return {k:saved[k] for k in saved.files}
 y=resample_poly(np.mean(stereo,axis=1),147,320).astype(np.float32)
 D=librosa.stft(y,n_fft=FFT,hop_length=HOP);harmonic,percussive=librosa.decompose.hpss(D,margin=(2,2));P=abs(percussive)**2
 mel=librosa.feature.melspectrogram(S=P,sr=FS,n_fft=FFT,n_mels=64);freq=librosa.mel_frequencies(n_mels=64,fmin=0,fmax=FS/2)
 envs={}
 for name,mask in [('low',(freq>=30)&(freq<=220)),('mid',(freq>=220)&(freq<2500)),('high',freq>=2500),('full',np.ones(len(freq),bool))]:
  envs[name]=librosa.onset.onset_strength(S=librosa.power_to_db(mel[mask],ref=1.),sr=FS,hop_length=HOP,n_fft=FFT)
 hy=librosa.istft(harmonic,hop_length=HOP,length=len(y));tuning=float(librosa.estimate_tuning(S=abs(harmonic),sr=FS,n_fft=FFT, resolution=.01))
 # Keep a fixed pitch-class reference so independently detuned streams are not
 # silently retuned to make their chroma look compatible.
 chroma=librosa.feature.chroma_cqt(y=hy,sr=FS,hop_length=HOP,tuning=0,n_chroma=12,n_octaves=6)
 result=dict(**envs,chroma=chroma,tuning_cents=np.array(tuning*100),time=librosa.frames_to_time(np.arange(len(envs['low'])),sr=FS,hop_length=HOP),chroma_time=librosa.frames_to_time(np.arange(chroma.shape[1]),sr=FS,hop_length=HOP))
 np.savez_compressed(path,**result);return result

def fit_phase(phases,bpm):
 valid=[p for p in phases if p['concentration']>=.15 and p['phase_beats'] is not None]
 if len(valid)<4:return dict(reliable=False,bars=len(valid),bpm=None,residual_beats=None)
 t=np.array([p['time'] for p in valid]);q=np.unwrap(np.array([p['phase_beats'] for p in valid])*2*np.pi)/(2*np.pi)
 slope=float(theilslopes(q,t)[0]);intercept=float(np.median(q-slope*t));residual=float(np.median(abs(q-(slope*t+intercept))))
 return dict(reliable=bool(residual<=.08),bars=len(valid),bpm=float(bpm-60*slope),residual_beats=residual,phase_drift_beats=float(slope*(t[-1]-t[0])))
def lag_compare(a,b,bpm):
 maximum=round(.5*60/bpm*FS/HOP);rows=[]
 for lag in range(-maximum,maximum+1):
  x,y=(a[-lag:],b[:lag]) if lag<0 else (a[:-lag],b[lag:]) if lag else (a,b)
  if len(x)<8 or np.std(x)<1e-8 or np.std(y)<1e-8:continue
  rows.append((float(np.corrcoef(x,y)[0,1]),lag))
 if not rows:return dict(offset_ms=None,peak_correlation=None,zero_correlation=None)
 best=max(rows,key=lambda v:(v[0],-abs(v[1])));zero=next((v[0] for v in rows if v[1]==0),None)
 return dict(offset_ms=float(best[1]*HOP/FS*1000),peak_correlation=best[0],zero_correlation=zero)
def track_analysis(features,bpm,bars):
 bar=240/bpm;phases=[];local=[]
 for i in range(bars):
  t=features['time'];mask=(t>=i*bar+.03)&(t<(i+1)*bar-.03)
  v=phase(features['low'][mask],t[mask],bpm);phases.append(dict(v,bar=i+1,time=(i+.5)*bar))
  if i%2==0:
   mask=(t>=i*bar)&(t<min(bars,i+4)*bar);local.append(dict(start_bar=i+1,**period(features['low'][mask],bpm)))
 near=period(features['low'],bpm);wide={}
 for band in ['low','full']:
  env=features[band]
  if np.std(env)<1e-8:wide[band]=None
  else:wide[band]=float(np.asarray(librosa.feature.tempo(onset_envelope=env,sr=FS,hop_length=HOP,prior=uniform(50,170),max_tempo=220,ac_size=min(8,len(env)*HOP/FS))).reshape(-1)[0])
 return dict(near_renderer_bpm=near,unconstrained_tempo=wide,bar_phases=phases,phase_fit=fit_phase(phases,bpm),local_tempo=local,key=key_estimate(features['chroma']),tuning_cents=float(features['tuning_cents']))
def rhythm_assessment(a,b,aa,bb,bpm,bar_records):
 observations=[]
 for x,y,record in zip(aa['bar_phases'],bb['bar_phases'],bar_records):
  if x['concentration']>=.15 and y['concentration']>=.15 and record['rendered']['coaudible_fraction']>=.15:
   observations.append(dict(bar=x['bar'],offset_beats=float(wrap(y['phase_beats']-x['phase_beats'])),source_concentration=x['concentration'],target_concentration=y['concentration']))
 delta=None;jitter=None
 if observations:
  q=np.array([v['offset_beats'] for v in observations]);delta=float(np.angle(np.mean(np.exp(2j*np.pi*q)))/(2*np.pi));jitter=float(np.median(abs(wrap(q-delta))))
 lag=lag_compare(a['low'],b['low'],bpm)
 fits=[aa['phase_fit'],bb['phase_fit']];periods=[aa['near_renderer_bpm'],bb['near_renderer_bpm']]
 supported=all(f['reliable'] and f['bpm'] is not None and p['bpm'] is not None and p['correlation']>=.3 and abs(f['bpm']-p['bpm'])<=1.2 for f,p in zip(fits,periods))
 gap=float(fits[1]['bpm']-fits[0]['bpm']) if supported else None
 stable=bool(len(observations)>=3 and jitter is not None and jitter<=.07)
 concordant=bool(stable and lag['offset_ms'] is not None and lag['peak_correlation']>=.25 and abs(wrap(delta-lag['offset_ms']/1000*bpm/60))<=.12)
 offset_ms=delta*60/bpm*1000 if delta is not None else None
 if supported and abs(gap)>.8:state='possible_tempo_drift';text=f'Possible pulse-speed mismatch: incoming minus outgoing {gap:+.2f} BPM. Check the audible merge.'
 elif concordant and abs(offset_ms)>max(40,.08*60/bpm*1000):state='consistent_drum_offset';text=f'Repeated low-drum offset around {offset_ms:+.0f} ms; the speeds can match while the hits remain displaced.'
 elif supported and stable and abs(offset_ms)<=40:state='timing_supported';text=f'Pulse-speed estimates agree within {abs(gap):.2f} BPM; low-drum offset about {offset_ms:+.0f} ms.'
 else:state='rhythm_uncertain';text='Drum evidence is ambiguous or varies across the merge; no confident timing pass/fail.'
 return dict(state=state,text=text,renderer_target_bpm=bpm,pulse_bpm_gap=gap,tempo_evidence_supported=supported,low_drum_offset_ms=offset_ms,relative_phase_jitter_beats=jitter,coaudible_supported_bars=len(observations),bar_offsets=observations,phase_and_correlation_agree=concordant,low_band_correlation=lag)
def harmonic_assessment(fa,fb,aa,bb,bar_records,bpm):
 ka,kb=aa['key'],bb['key'];relation=key_relation(ka['key'],kb['key']);confident=ka['confident'] and kb['confident'];bar=240/bpm;bars=[]
 for record in bar_records:
  i=record['bar']-1;vals=[]
  for f in [fa,fb]:
   mask=(f['chroma_time']>=i*bar)&(f['chroma_time']<(i+1)*bar);vals.append(key_estimate(f['chroma'][:,mask]))
  bars.append(dict(bar=i+1,source_key=vals[0],target_key=vals[1],relation=key_relation(vals[0]['key'],vals[1]['key']),both_present_fraction=record['rendered']['coaudible_fraction'],simultaneous_chroma_cosine=record['rendered']['frame_harmonic_chroma_cosine']))
 tuning_delta=float((bb['tuning_cents']-aa['tuning_cents']+50)%100-50)
 if confident and relation=='different':state='different_local_keys';text=f"Local key estimates differ: {ka['key']} → {kb['key']}. Check sustained melodies during the merge."
 elif confident:state='related_local_keys';text=f"Local key estimates are related: {ka['key']} → {kb['key']}. Chord-by-chord fit still matters."
 else:state='key_uncertain';text=f"Local key is ambiguous; leading candidates are {ka['key']} → {kb['key']}. Do not treat this as a confirmed key clash."
 return dict(state=state,text=text,source=ka,target=kb,camelot_relation=relation,both_keys_confident=confident,source_tuning_cents=aa['tuning_cents'],target_tuning_cents=bb['tuning_cents'],relative_tuning_cents=tuning_delta,bars=bars)
def main():
 protocol=json.loads((HERE/'protocol.json').read_text());overlaps=json.loads((HERE/'overlap-analysis.json').read_text());tracks={t['id']:t for t in r.api('tracks')};gate=joblib.load(HERE.parent/'part_alignment_v1/artifacts/beat_repair_gate.joblib')
 report=dict(version='deep_merge_librosa_v1',created=time.time(),scope='Measured screening of actual tempo-warped/looped audition regions, weighted toward bars where both are present; no listener approval or automated cue changes.',rules=dict(near_bpm_search_percent=12,phase_min_concentration=.15,phase_fit_max_median_residual_beats=.08,phase_correlation_min=.25,tempo_gap_watch_bpm=.8,offset_watch_ms=40,key_min_profile_correlation=.6,key_min_top_margin=.08),method_notes=['Low/mid/high/full percussive HPSS onset envelopes; independent wide Librosa tempo estimates retain half/double-tempo ambiguity.','Constrained local pulse estimates are explicitly labeled near renderer BPM; bar-phase drift provides a separate cross-check.','Krumhansl key profiles on harmonic CQT chroma are local hypotheses. A Camelot relationship does not prove that simultaneous chords fit.','Offset flags can reflect intentional syncopation. These are review prompts, not calibrated musical quality decisions.'],previews={})
 for spec in protocol['experiments']:
  for p in json.loads((HERE/(spec['id']+'.json')).read_text())['previews']:
   print('DEEP_ANALYZE',spec['id'],p['comparison']['variant'],flush=True);a,b,ca,cb,diff=rebuild(p,tracks,gate);fa,fb=percussion_features(a),percussion_features(b);s=p['selection'];bpm=s['mix_bpm'];bars=s['overlap_bars'];aa,bb=track_analysis(fa,bpm,bars),track_analysis(fb,bpm,bars);old=overlaps['previews'][p['audio']['sha256']];assert binding(p)==old['preview_snapshot_sha256']
   rhythm=rhythm_assessment(fa,fb,aa,bb,bpm,old['bars']);harmonic=harmonic_assessment(fa,fb,aa,bb,old['bars'],bpm)
   report['previews'][p['audio']['sha256']]=dict(audio_sha256=p['audio']['sha256'],preview_snapshot_sha256=binding(p),reconstruction_pcm_max_difference=diff,group_id=p['comparison']['group_id'],variant=p['comparison']['variant'],source_track=s['source_track'],target_track=s['target_track'],source_title=s['source_title'],target_title=s['target_title'],label=p['comparison_label'],rhythm=rhythm,harmony=harmonic,source_analysis=aa,target_analysis=bb)
   write(HERE/'deep-merge-progress.json',report)
 report.update(completed=time.time(),clips=len(report['previews']),all_reconstructions_bit_exact=True,script_sha256=sha(__file__));assert report['clips']==17;write(HERE/'deep-merge-analysis.json',report);print('DEEP_MERGE_READY',report['clips'],flush=True)
if __name__=='__main__':main()
