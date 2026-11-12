"""Offline, immutable listening experiments across distinct library songs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,json,copy,time,uuid,hashlib,wave,fcntl,argparse
from pathlib import Path
import numpy as np,joblib
from scipy.signal import butter,sosfilt
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN/'part_alignment_v1'))
import render_live as r
SR=r.SR

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
 temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)
def ramp(n,start=0.,end=1.):
 x=np.clip((np.linspace(0,1,n)-start)/(end-start),0,1);return x*x*(3-2*x)
def mix(a,b,npre,nover,mode='blend'):
 t=ramp(nover);lo=t;hi=t
 if mode=='bands':lo=ramp(nover,0,.25);hi=ramp(nover,.75,1)
 if mode in ['cut','dry','echo']:
  width=round(.01*SR) if mode=='cut' else round(nover/8)
  end=nover//2;start=end-width
  u=np.clip((np.arange(nover)-start)/max(width,1),0,1);u=u*u*(3-2*u)
  blend=a[npre:]*np.cos(u[:,None]*np.pi/2)+b[:nover]*np.sin(u[:,None]*np.pi/2)
  if mode=='echo':
   # Feed only the final actual half-beat before the exit into four delayed taps.
   delay=round(nover/64);seed=a[npre+end-delay:npre+end].copy()
   taper=min(round(.003*SR),len(seed)//4);seed[:taper]*=np.linspace(0,1,taper)[:,None];seed[-taper:]*=np.linspace(1,0,taper)[:,None]
   for k in range(4):
    begin=end+k*delay;finish=min(nover,begin+len(seed));blend[begin:finish]+=seed[:finish-begin]*(.35*.45**k)
  return np.concatenate([a[:npre],blend,b[nover:]])
 lowa=sosfilt(butter(3,180,fs=SR,output='sos'),a,axis=0);lowb=sosfilt(butter(3,180,fs=SR,output='sos'),b,axis=0)
 blend=lowa[npre:]*(1-lo[:,None])+lowb[:nover]*lo[:,None]+(a[npre:]-lowa[npre:])*np.cos(hi[:,None]*np.pi/2)+(b[:nover]-lowb[:nover])*np.sin(hi[:,None]*np.pi/2)
 return np.concatenate([a[:npre],blend,b[nover:]])
def main(ident):
 protocol=json.loads((HERE/'protocol.json').read_text());spec=next(s for s in protocol['experiments'] if s['id']==ident);group='wide-'+ident+'-v1'
 receipt=HERE/(ident+'.json');assert not receipt.exists(),'Inspect existing immutable receipt'
 plan=json.loads((HERE/(ident+'-plan.json')).read_text());tracks={t['id']:t for t in r.api('tracks')};source=tracks[spec['source']];target=tracks[spec['target']]
 assert len({spec['source'],spec['target']}&{5,126,10,101})==0
 assert sha(source['file_path'])==plan['source_audio_sha256'] and sha(target['file_path'])==plan['target_audio_sha256']
 gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');bpm=plan['selection']['mix_bpm'];bar=240/bpm;npre=round(4*bar*SR)
 def stems(p,bars):
  s=p['selection'];nover=round(bars*bar*SR)
  a,ai=r.aligned_stem(source,s['source_start_seconds'],4,4+bars,bpm,gate,npre+nover)
  b,bi=r.aligned_stem(target,s['target_start_seconds'],0,bars+4,bpm,gate,nover+npre)
  assert a.shape==(npre+nover,2) and b.shape==(nover+npre,2)
  return a,b,ai,bi,nover
 # Every planned alternative is genuinely ranked by the existing scorer. Physical
 # rejection is recorded; no failed stem is padded with invented music.
 attempts=[];candidates=[plan]+[dict(plan,selection=v['selection'],loop_candidate=v['loop_candidate']) for v in plan['alternatives']]
 for p in candidates:
  try:
   if spec['kind']=='target_loop':
    data=json.loads((TRAIN/'semantic_pair_v1/halo_features'/f"{spec['target']}_loops_4.json").read_text())
    assert data['audio_sha256']==plan['target_audio_sha256']
    assert data['continuity_model_sha256']==sha(TRAIN/'raveform_audio_v1/artifacts/loop_region_continuity.joblib')
    matching=[x for x in data['candidates'] if x['accepted'] and abs(x['start']-p['selection']['target_start_seconds'])<1e-5]
    if not matching:raise ValueError('Ranked incoming part did not pass loop pre-screen')
    incoming_check=matching[0]
   a,b,ai,bi,nover=stems(p,spec['bars'])
   if spec['kind']=='source_loop':
    assert p['loop_candidate']['accepted'];unit,ui=r.aligned_stem(source,p['selection']['source_start_seconds'],0,4,bpm,gate,npre)
    boundary=r.loop_boundary_metrics(unit,SR,bar)
    if boundary['boundary_level_difference_db']>8:raise ValueError('Rendered source-loop boundary exceeds 8 dB')
   if spec['kind']=='target_loop':
    unit,ui=r.aligned_stem(target,p['selection']['target_start_seconds'],0,4,bpm,gate,npre)
    boundary=r.loop_boundary_metrics(unit,SR,bar)
    if boundary['boundary_level_difference_db']>8:raise ValueError('Rendered incoming-loop boundary exceeds 8 dB')
   attempts.append(dict(source=p['selection']['source_start_seconds'],target=p['selection']['target_start_seconds'],accepted=True));break
  except ValueError as e:attempts.append(dict(source=p['selection']['source_start_seconds'],target=p['selection']['target_start_seconds'],accepted=False,reason=str(e)))
 else:raise ValueError('No ranked candidate passed physical/loop checks: '+str(attempts))
 # One incoming gain and one output gain for the complete comparison, including
 # alternate parts. This deliberately avoids normalizing each variant separately.
 gain=float(np.clip(np.sqrt(np.mean(a[npre:npre+min(nover,2*npre)]**2)+1e-12)/np.sqrt(np.mean(b[:min(nover,2*npre)]**2)+1e-12),.5,2));b*=gain
 variants=[]
 def add(label,aa,bb,n,pp=p,source_info=ai,target_info=bi,mode='blend',loop=None,target_loop=None):
  selection=copy.deepcopy(pp['selection']);selection['overlap_bars']=round(n/SR/bar)
  selection['source_section']['end']=selection['source_start_seconds']+(4 if loop else selection['overlap_bars'])*bar
  selection['target_section']['end']=selection['target_start_seconds']+(4 if target_loop else selection['overlap_bars'])*bar*selection['target_tempo_factor']
  variants.append(dict(label=label,samples=mix(aa,bb,npre,n,mode),selection=selection,source_info=source_info,target_info=target_info,mode=mode,loop=loop,target_loop=target_loop))
 kind=spec['kind']
 if kind=='length':
  shorter=8 if spec['bars']==16 else 4;aa,bb,axi,bxi,n=stems(p,shorter);bb*=gain
  add(f'{shorter}-bar blend',aa,bb,n,source_info=axi,target_info=bxi);add(f"{spec['bars']}-bar blend",a,b,nover)
 elif kind=='parts':
  add('Section pair A',a,b,nover)
  alternate=next((q for q in candidates if q['selection']!=p['selection']),None);assert alternate
  aa,bb,axi,bxi,n=stems(alternate,spec['bars']);bb*=gain;add('Section pair B',aa,bb,n,pp=alternate,source_info=axi,target_info=bxi)
 elif kind=='source_loop':
  add('Straight blend',a,b,nover)
  aa=np.concatenate([a[:npre],r.repeat_loop(unit,2,sample_rate=SR)]);assert aa.shape==a.shape
  loop=dict(p['loop_candidate'],bars=4,period_frames=npre,**boundary,edge_taper_ms=3,musical_review='pending',scope='Outgoing overlap only; actual pre-roll unchanged')
  add('Repeat outgoing 4-bar groove',aa,b,nover,source_info=dict(ai,loop_unit_alignment=ui),loop=loop)
 elif kind=='target_loop':
  add('Straight incoming song',a,b,nover)
  # Two exact 4-bar units, then the real next 4 bars. Taper release boundaries.
  continuation=b[npre:2*npre].copy();taper=round(.003*SR);continuation[:taper]*=np.linspace(0,1,taper)[:,None]
  bb=np.concatenate([r.repeat_loop(unit*gain,2,sample_rate=SR),continuation]);assert bb.shape==b.shape
  loop=dict(incoming_check,bars=4,period_frames=npre,**boundary,edge_taper_ms=3,musical_review='pending',release_source_seconds=p['selection']['target_start_seconds']+4*240/plan['target_bpm'])
  add('Repeat incoming 4-bar bed, then release',a,bb,nover,target_info=dict(bi,loop_unit_alignment=ui),target_loop=loop)
 else:
  add('Standard 8-bar blend',a,b,nover)
  if kind=='echo':add('Short exit · dry',a,b,nover,mode='dry');add('Short exit · half-beat echo',a,b,nover,mode='echo')
  elif kind=='bands':add('Bass first · melody later',a,b,nover,mode='bands')
  elif kind=='cut':add('Phrase cut · 10 ms crossfade',a,b,nover,mode='cut')
  else:raise ValueError(kind)
 for v in variants:assert np.isfinite(v['samples']).all() and len(v['samples'])==npre+round(v['selection']['overlap_bars']*bar*SR)+npre
 peak=max(float(abs(v['samples']).max()) for v in variants);output_gain=min(1,.95/max(peak,1e-10));published=[]
 for index,v in enumerate(variants):
  y=v.pop('samples');raw_peak=float(abs(y).max());y*=output_gain;nf=round(.01*SR);y[:nf]*=np.linspace(0,1,nf)[:,None];y[-nf:]*=np.linspace(1,0,nf)[:,None];after=float(abs(y).max());assert after<=.950001
  job=uuid.uuid4().hex;path=r.ROOT/f'live_{job}.wav';assert not path.exists()
  with wave.open(str(path),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(SR);f.writeframes((y*32767).astype('<i2').tobytes())
  with wave.open(str(path)) as f:assert f.getnframes()==len(y) and f.getnchannels()==2 and f.getframerate()==SR
  item=dict(selection=v['selection'],engine=dict(pre_bars=4,post_bars=4,loop=v['loop'],target_loop=v['target_loop'],source_alignment=v['source_info'],target_alignment=v['target_info'],incoming_level_gain=gain,level_matching='Shared incoming RMS gain and shared output peak gain for this comparison',strategy=v['mode']),eq_enabled=v['mode'] in ['blend','bands'],audio=dict(path=str(path),seconds=len(y)/SR,frames=len(y),sample_rate=SR,peak_before_gain=raw_peak,gain_applied=output_gain,peak_after_gain=after,sha256=sha(path)),comparison_label=v['label'],job_id=job,created=time.time(),selector_model_sha256=plan['model_sha256'],selector_version=plan['selector_version'],source_audio_sha256=plan['source_audio_sha256'],target_audio_sha256=plan['target_audio_sha256'],planning_version='wide_library_listening_v1',algorithm='Experimental ranked parts → gated beat repair → pitch-preserving R3 beat maps → controlled listening strategy',listener_review='pending',deck_commands_sent=False,
   comparison=dict(group_id=group,variant=chr(65+index),kind='library_'+kind,collection_id=protocol['collection_id'],idea=spec['idea'],description=spec['description'],style=spec['style'],experiment_order=protocol['experiments'].index(spec),same_output_gain=True,same_incoming_gain=True,source_artist=source.get('artist'),target_artist=target.get('artist'),musical_quality='Pending listening review',parameter_replay_supported=False))
  if kind=='echo' and v['mode']=='echo':item['engine']['echo']=dict(delay_beats=.5,taps=4,wet=.35,feedback=.45,seed='Final actual half-beat before exit',source_cut_after_bars=4)
  published.append(item)
 result=dict(id=ident,state='complete',created=time.time(),group_id=group,previews=published,physical_candidate_attempts=attempts,deck_commands_sent=False,listening_quality_verified=False,script_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'))
 write(HERE/(ident+'-ready.json'),result)
 with (r.ROOT/'manifest.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);manifest=r.ROOT/'manifest.json';data=json.loads(manifest.read_text());assert not any(x.get('comparison',{}).get('group_id')==group for x in data['renders'])
  data['renders'].extend(published);temp=r.ROOT/('manifest.'+uuid.uuid4().hex+'.tmp.json');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(manifest)
 write(receipt,result);print('GROUP_READY',ident,len(published),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);main(parser.parse_args().spec)
