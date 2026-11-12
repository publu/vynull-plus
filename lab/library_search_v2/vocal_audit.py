"""Estimate simultaneous prominent singing from separated, actually faded vocals.

HTDemucs separates vocals, not lead versus backing singers. This is a conservative
vocal-overlap warning, not a calibrated judgment of musical or lyrical clashes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import sys,json,time,hashlib,fcntl,uuid,wave,copy
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
import numpy as np,torch,joblib
from scipy.signal import resample_poly,butter,sosfilt
from demucs import pretrained
from demucs.apply import apply_model
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
# DSP dependencies live in the established analysis environment. No package
# changes are made to that environment by the isolated separator runtime.
for p in (TRAIN/'library_lab_v1',TRAIN/'part_alignment_v1'):
 sys.path.append(str(p))
from render_group import r,mix,sha,write
from analyze_overlaps import binding
SR=48000
RULES=dict(frame_seconds=.05,vocal_floor_dbfs=-40,vocal_to_own_stream_rms_min=.25,vocal_to_mix_rms_min=.20,max_voice_level_gap_db=10,sustained_seconds=.5,total_seconds=1.)

def spans(mask,step):
 changes=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1);return [(float(s*step),float(e*step)) for s,e in zip(starts,ends)]
def rms_frames(y,n):
 # Stereo energy avoids cancellation in a mono sum.
 count=len(y)//n;return np.sqrt(np.mean(y[:count*n].reshape(count,n,2)**2,axis=(1,2))+1e-12)
def measure(vs,vt,source,target,bpm,overlap_bars):
 n=round(RULES['frame_seconds']*SR);a,b,c,d=[rms_frames(y,n) for y in [vs,vt,source,target]];combined=rms_frames(source+target,n);floor=10**(RULES['vocal_floor_dbfs']/20)
 sa=(a>=floor)&(a/np.maximum(c,1e-10)>=RULES['vocal_to_own_stream_rms_min'])&(a/np.maximum(combined,1e-10)>=RULES['vocal_to_mix_rms_min']);sb=(b>=floor)&(b/np.maximum(d,1e-10)>=RULES['vocal_to_own_stream_rms_min'])&(b/np.maximum(combined,1e-10)>=RULES['vocal_to_mix_rms_min']);gap=abs(20*np.log10(np.maximum(a,1e-10)/np.maximum(b,1e-10)));both=sa&sb&(gap<=RULES['max_voice_level_gap_db']);intervals=spans(both,RULES['frame_seconds']);longest=max([end-start for start,end in intervals],default=0.);total=float(both.sum()*RULES['frame_seconds']);watch=longest>=RULES['sustained_seconds'] or total>=RULES['total_seconds'];bar=240/bpm
 return dict(status='vocal_overlap_watch' if watch else 'no_sustained_overlap_detected',text=f'Possible competing vocals: {total:.2f}s total, longest stretch {longest:.2f}s.' if watch else 'No sustained simultaneous prominent vocals detected at these settings.',simultaneous_vocal_seconds=total,longest_vocal_overlap_seconds=longest,overlap_fraction=float(np.mean(both)),source_vocal_present_seconds=float(sa.sum()*RULES['frame_seconds']),target_vocal_present_seconds=float(sb.sum()*RULES['frame_seconds']),intervals=[dict(start_seconds=s,end_seconds=e,preview_seconds=4*bar+s,merge_bar=1+int(s/bar)) for s,e in intervals if e-s>=.1],bars=[dict(bar=i+1,simultaneous_vocal_seconds=float(both[(np.arange(len(both))*.05>=i*bar)&(np.arange(len(both))*.05<(i+1)*bar)].sum()*.05)) for i in range(overlap_bars)])

class Separator:
 def __init__(self):
  torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);self.device='cuda' if torch.cuda.is_available() else 'cpu';self.model=pretrained.get_model('htdemucs').eval().to(self.device);self.index=self.model.sources.index('vocals');self.weights=Path.home()/'.cache/torch/hub/checkpoints/955717e8-8726e21a.th';self.model_sha=sha(self.weights);assert self.model_sha.startswith('8726e21a');self.hits=0;self.misses=0
 def vocals(self,y):
  config=dict(version='htdemucs_vocal_context_v1',weights=self.model_sha,device=self.device,attention='math_mha_slow',shifts=0,overlap=.25,input_sample_rate=SR,model_sample_rate=44100,torch=torch.__version__,normalization='Stereo mean standard deviation; zero mean',frames=len(y));key=hashlib.sha256(y.astype('<f4',copy=False).tobytes()+json.dumps(config,sort_keys=True).encode()).hexdigest();folder=HERE/'vocal-cache';folder.mkdir(exist_ok=True);path=folder/(key+'.npz')
  with (folder/(key+'.lock')).open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if path.exists():
    with np.load(path) as data:result=data['vocals'];assert result.shape==y.shape and np.isfinite(result).all()
    self.hits+=1;return result,dict(cache_key=key,cache_hit=True)
   started=time.perf_counter();x=resample_poly(y.T,147,160,axis=1).astype(np.float32);reference=x.mean(axis=0);mean=float(reference.mean());std=max(float(reference.std()),1e-6);tensor=torch.from_numpy((x-mean)/std)[None].to(self.device)
   with torch.inference_mode():prediction=apply_model(self.model,tensor,device=self.device,shifts=0,split=True,overlap=.25,progress=False)[0].cpu().numpy()
   vocals=prediction[self.index]*std;result=resample_poly(vocals,160,147,axis=1).T[:len(y)].astype(np.float32);assert result.shape==y.shape and np.isfinite(result).all();temp=folder/(key+'.'+uuid.uuid4().hex+'.tmp.npz')
   with temp.open('wb') as file:np.savez(file,vocals=result,config=np.array(json.dumps(config)))
   temp.replace(path);self.misses+=1;return result,dict(cache_key=key,cache_hit=False,seconds=time.perf_counter()-started)

def audit(p,separator,tracks,gate):
 s=p['selection'];bar=240/s['mix_bpm'];npre=round(4*bar*SR);nover=round(s['overlap_bars']*bar*SR)
 for side in ['source','target']:assert sha(tracks[s[side+'_track']]['file_path'])==p[side+'_audio_sha256']
 a,ai=r.aligned_stem(tracks[s['source_track']],s['source_start_seconds'],4,4+s['overlap_bars'],s['mix_bpm'],gate,npre+nover);b,bi=r.aligned_stem(tracks[s['target_track']],s['target_start_seconds'],0,s['overlap_bars']+4,s['mix_bpm'],gate,nover+npre)
 if p['engine'].get('loop'):
  loop=p['engine']['loop'];period=loop['period_frames'];unit,_=r.aligned_stem(tracks[s['source_track']],s['source_start_seconds'],0,loop['bars'],s['mix_bpm'],gate,period);a=np.concatenate([a[:npre],r.repeat_loop(unit,nover//period,sample_rate=SR)])
 av,acache=separator.vocals(a);bv,bcache=separator.vocals(b);gain=p['engine']['incoming_level_gain'];b*=gain;bv*=gain;mode=p['engine']['strategy'];output_gain=p['audio']['gain_applied'];reference=mix(a,b,npre,nover,mode)*output_gain;nf=round(.01*SR);reference[:nf]*=np.linspace(0,1,nf)[:,None];reference[-nf:]*=np.linspace(1,0,nf)[:,None]
 with wave.open(p['audio']['path']) as f:published=np.frombuffer(f.readframes(f.getnframes()),'<i2').reshape(-1,2)
 assert sha(p['audio']['path'])==p['audio']['sha256'];assert np.array_equal((reference*32767).astype('<i2'),published),'Vocal analysis references a different mix'
 source=mix(a,np.zeros_like(b),npre,nover,mode)[npre:npre+nover]*output_gain;target=mix(np.zeros_like(a),b,npre,nover,mode)[npre:npre+nover]*output_gain;vs=mix(av,np.zeros_like(bv),npre,nover,mode)[npre:npre+nover]*output_gain;vt=mix(np.zeros_like(av),bv,npre,nover,mode)[npre:npre+nover]*output_gain
 # Remove separator DC/rumble before prominence measurements.
 sos=butter(2,100,btype='highpass',fs=SR,output='sos');vs=sosfilt(sos,vs,axis=0);vt=sosfilt(sos,vt,axis=0);source=sosfilt(sos,source,axis=0);target=sosfilt(sos,target,axis=0)
 result=measure(vs,vt,source,target,s['mix_bpm'],s['overlap_bars']);result.update(audio_sha256=p['audio']['sha256'],preview_snapshot_sha256=binding(p),reconstruction_pcm_max_difference=0,model='htdemucs',model_sha256=separator.model_sha,source_cache=acache,target_cache=bcache,scope='Estimated prominent vocal overlap after actual EQ/fades/loop/cut. Model can leak instruments and does not identify lead versus backing singers.');return result

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=TRAIN/'discovery_rounds_v1/published.json');parser.add_argument('--output',type=Path,default=HERE/'vocal-audit.json');args=parser.parse_args();data=json.loads(args.input.read_text());items=data.get('previews',[]);tracks={t['id']:t for t in r.api('tracks')};gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');separator=Separator();report=dict(version='vocal_overlap_v1',created=time.time(),rules=RULES,model_sha256=separator.model_sha,previews={},errors={})
 for p in items:
  try:entry=audit(p,separator,tracks,gate);report['previews'][p['audio']['sha256']]=entry;print('VOCAL_AUDIT',p['selection']['source_title'],p['selection']['target_title'],p['comparison']['variant'],entry['status'],entry['simultaneous_vocal_seconds'],flush=True)
  except Exception as e:import traceback;traceback.print_exc();report['errors'][p['audio']['sha256']]=str(e)
  report.update(cache_hits=separator.hits,cache_misses=separator.misses);write(args.output.with_name(args.output.stem+'-progress.json'),report)
 report.update(completed=time.time(),clips=len(report['previews']),script_sha256=sha(__file__),all_reconstructions_bit_exact=not report['errors']);write(args.output,report);print('VOCAL_AUDIT_READY',report['clips'],len(report['errors']),separator.hits,separator.misses,flush=True)
if __name__=='__main__':main()
