"""Phrase-aware incoming-loop bridges: old vocal finishes, bed holds, new vocal follows."""
from pathlib import Path
import sys,json,time,copy,uuid,wave,subprocess,hashlib,fcntl,math,argparse,importlib.metadata,zipfile
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN/'library_search_v2'))
from vocal_audit import Separator,r,sha,write,np,joblib,rms_frames,measure,SR
from scipy.signal import butter,sosfilt
sys.path.insert(0,str(TRAIN/'library_lab_v1'))
from analyze_overlaps import features,compare,binding
from deep_merge_analysis import percussion_features,period,track_analysis,rhythm_assessment,harmonic_assessment
sys.path.insert(0,str(TRAIN.parent/'review'))
from section_matches import key_relation

def smooth(x):x=np.clip(x,0,1);return x*x*(3-2*x)
def strict_decode(path,start,end):
 process=subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-i',str(path),'-t',str(end-start),'-vn','-ac','2','-ar',str(SR),'-f','f32le','pipe:1'],capture_output=True,timeout=120)
 if process.returncode or process.stderr.strip():raise ValueError('Audio decode error in proposed section; try another clean region')
 y=np.frombuffer(process.stdout,'<f4').reshape(-1,2).copy()
 if not len(y) or not np.isfinite(y).all():raise ValueError('Empty or invalid audio section')
 return y

_RUNTIME=None

def aligned(track,start,before,bars,bpm,gate,frames):
 global _RUNTIME
 analysis=r.api(f"analysis/{track['id']}")
 if _RUNTIME is None:
  _RUNTIME={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','joblib']}
  for tool,flag in [('ffmpeg','-version'),('rubberband','--version')]:
   p=subprocess.run([tool,flag],capture_output=True,text=True,timeout=10);_RUNTIME[tool]=(p.stdout+p.stderr).splitlines()[0]
 config=dict(version='strict_phrase_stem_v1',audio_sha256=sha(track['file_path']),grid=analysis['beats'],start=float(start),before=before,bars=bars,bpm=bpm,frames=frames,gate=joblib.hash(gate),runtime=_RUNTIME,engine_sha256=sha(__file__),warp_sha256=sha(TRAIN/'part_alignment_v1/beatwarp.py'),repair_sha256=sha(TRAIN/'part_alignment_v1/refine_beats.py'))
 key=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest();folder=HERE/'strict-stems';folder.mkdir(exist_ok=True);path=folder/(key+'.npz')
 with (folder/(key+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if path.exists():
   try:
    with np.load(path) as data:y=data['audio'];info=json.loads(str(data['info']));assert str(data['key'])==key and y.shape==(frames,2) and np.isfinite(y).all()
    return y,info
   except (ValueError,OSError,KeyError,EOFError,AssertionError,zipfile.BadZipFile):pass
  original=r.decode;r.decode=strict_decode
  try:y,info=r._aligned_stem_uncached(track,start,before,bars,bpm,gate,frames,_analysis=analysis)
  finally:r.decode=original
  temp=folder/(key+'.'+uuid.uuid4().hex+'.tmp.npz')
  with temp.open('wb') as file:np.savez(file,audio=y,info=np.array(json.dumps(info)),key=np.array(key))
  temp.replace(path);return y,info

def load_map(track,separator):
 path=HERE/f"vocal-map-{track['id']}.json";digest=sha(track['file_path'])
 if path.exists():
  data=json.loads(path.read_text())
  if data['audio_sha256']==digest and data['model_sha256']==separator.model_sha:
   current=r.api(f"analysis/{track['id']}");data.update(bpm=current['bpm'],beats=current['beats']);return data
 y=r.decode(track['file_path'],0,float(track['duration']));vocals,_=separator.vocals(y);level=rms_frames(y,2400);voice=rms_frames(vocals,2400);active=(voice>=.01)&(voice/np.maximum(level,1e-10)>=.25);analysis=r.api(f"analysis/{track['id']}");data=dict(id=track['id'],title=track['title'],artist=track.get('artist'),audio_sha256=digest,model_sha256=separator.model_sha,frame_seconds=.05,duration=len(y)/SR,bpm=analysis['bpm'],beats=analysis['beats'],rms_db=(20*np.log10(np.maximum(level,1e-10))).tolist(),vocal_rms_db=(20*np.log10(np.maximum(voice,1e-10))).tolist(),vocal_active=active.astype(int).tolist(),scope='Candidate map; chosen sections require fresh strict decoding and rendered-vocal checks');write(path,data);return data

def downbeats(m):return np.array([v['time_ms']/1000 for v in m['beats'] if v['beat_in_bar']==1])
def window(m,start,end):
 lo=max(0,int(np.ceil(start/.05)));hi=min(len(m['vocal_active']),int(end/.05));return np.array(m['vocal_active'][lo:hi],bool),np.array(m['rms_db'][lo:hi],float)
def last_voice(m,start,end):
 lo=max(0,int(np.ceil(start/.05)));hi=min(len(m['vocal_active']),int(end/.05));where=np.flatnonzero(np.array(m['vocal_active'][lo:hi],bool));return (lo+int(where[-1])+1)*.05 if len(where) else None

def exit_candidates(m,overlap_bars):
 beats=downbeats(m);bar=240/m['bpm'];active=np.array(m['vocal_active'],bool);changes=np.diff(np.r_[False,~active,False].astype(int));rests=list(zip(np.flatnonzero(changes==1)*.05,np.flatnonzero(changes==-1)*.05));rows=[]
 for i,end in enumerate(beats):
  if i%4 or i<overlap_bars+8 or not .35*m['duration']<=end<=.88*m['duration']:continue
  _,rms=window(m,end-bar,end);_,prior=window(m,end-4*bar,end-2*bar)
  if not len(rms) or not len(prior) or np.mean(rms)<-32 or np.mean(rms)-np.mean(prior)<-4:continue
  gaps=[(a,b) for a,b in rests if a>=end-4*bar and b<=end and end-b<=2*bar and b-a>=.65]
  for a,b in gaps:
   score=abs(end/m['duration']-.7)+1/(1+(b-a)/(60/m['bpm']))
   rows.append(dict(start=float(beats[i-overlap_bars]),end=float(end),end_bar_index=i,native_rest_start=float(a),native_rest_end=float(b),score=float(score)))
 return sorted(rows,key=lambda row:row['score'])

def entry_candidates(m,unit_bars):
 beats=downbeats(m);bar=240/m['bpm'];active=np.array(m['vocal_active'],bool);rows=[]
 for i,start in enumerate(beats):
  if i%4 or i+unit_bars+16>=len(beats) or start>.75*m['duration']:continue
  end=float(beats[i+unit_bars]);unit,rms=window(m,start,end)
  if not len(unit) or unit.mean()>.01 or np.mean(rms)<-30:continue
  lo=int(np.ceil(end/.05));hi=min(len(active),int((end+8*bar)/.05));tail=active[lo:hi];run=0;entry=None
  for j,present in enumerate(tail):
   run=run+1 if present else 0
   if run>=3:entry=(lo+j-2)*.05;break
  if entry is None:continue
  score=(entry-end)/bar+.3*(start/m['duration']);rows.append(dict(start=float(start),end=end,start_bar_index=i,unit_bars=unit_bars,first_native_vocal=entry,score=float(score)))
 return sorted(rows,key=lambda row:row['score'])

def build_bed(track,m,candidate,bpm,gate,separator):
 unit_bars=candidate['unit_bars'];bar=240/bpm;i=candidate['start_bar_index'];native_bar=240/m['bpm'];post_bars=max(10,int(np.ceil((candidate['first_native_vocal']-candidate['end'])/native_bar))+8)
 total=unit_bars+post_bars;full,info=aligned(track,candidate['start'],0,total,bpm,gate,round(total*bar*SR));unit=full[:round(unit_bars*bar*SR)].copy();vocal,_=separator.vocals(full);voice_unit=vocal[:len(unit)];raw=rms_frames(unit,2400);vr=rms_frames(voice_unit,2400);fraction=float(np.mean((vr>=.01)&(vr/np.maximum(raw,1e-10)>=.25)))
 if fraction>.01:raise ValueError('Incoming loop contains a prominent vocal')
 boundary=r.loop_boundary_metrics(unit,SR,bar)
 if boundary['boundary_level_difference_db']>6:raise ValueError('Incoming loop has a level discontinuity')
 f=percussion_features(unit);pulse=max([period(f['low'],bpm),period(f['full'],bpm)],key=lambda p:p['correlation'])
 if pulse['correlation']<.3 or pulse['bpm'] is None or abs(pulse['bpm']/bpm-1)>.025:raise ValueError('Instrumental loop lacks a stable rhythmic pulse near the mix tempo')
 chroma=features(unit)['chroma'];count=max(1,round(bar*22050/256));a=chroma[:,:count].mean(axis=1);b=chroma[:,-count:].mean(axis=1);tonal=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-10))
 if tonal<.5:raise ValueError('Incoming loop has a large harmonic boundary change')
 return dict(candidate=candidate,full=full,vocal=vocal,unit=unit,alignment=info,boundary=boundary,loop_vocal_fraction=fraction,pulse=pulse,boundary_chroma_cosine=tonal,post_bars=post_bars)

def source_clip(track,m,candidate,overlap_bars,bpm,gate,separator):
 beats=downbeats(m);i=candidate['end_bar_index'];start=float(beats[i-overlap_bars]);bar=240/bpm;npre=round(8*bar*SR);nover=round(overlap_bars*bar*SR);full,info=aligned(track,start,8,8+overlap_bars,bpm,gate,npre+nover);vocal,_=separator.vocals(full);raw=rms_frames(full[npre:],2400);voice=rms_frames(vocal[npre:],2400);active=(voice>=.01)&(voice/np.maximum(raw,1e-10)>=.25);beat=60/bpm
 native_first=info['corrected_start_seconds'];src=np.array(info['source_anchor_seconds']);dst=np.array(info['target_anchor_seconds'])
 proposed_start=float(np.interp(candidate['native_rest_start']-native_first,src,dst)-npre/SR);proposed_end=float(np.interp(candidate['native_rest_end']-native_first,src,dst)-npre/SR)
 changes=np.diff(np.r_[False,~active,False].astype(int));rests=list(zip(np.flatnonzero(changes==1)*.05,np.flatnonzero(changes==-1)*.05));valid=[]
 for a,b in rests:
  if b-a<.4 or abs(a-proposed_start)>bar or a<(overlap_bars-4)*bar:continue
  fade_start=math.ceil((a+.075)/beat)*beat;fade_stop=math.floor((min(b-.075,overlap_bars*bar))/beat)*beat
  if fade_stop-fade_start>=.5*beat:valid.append((abs(a-proposed_start)+abs(b-proposed_end),fade_start,fade_stop,a,b))
 if not valid:raise ValueError('No beat-aligned quiet rest was found before the next outgoing vocal line')
 _,fade_start,fade_stop,a,b=min(valid);where=np.flatnonzero(active[:int(fade_start/.05)]);voice_end=(int(where[-1])+1)*.05 if len(where) else 0.
 last=full[-round(bar*SR):];prior=full[-round(4*bar*SR):-round(2*bar*SR)];delta=float(20*np.log10(max(np.sqrt(np.mean(last**2)),1e-10)/max(np.sqrt(np.mean(prior**2)),1e-10)))
 if delta<-4:raise ValueError('Outgoing recording is already fading away at this cue')
 return dict(full=full,vocal=vocal,alignment=info,candidate=dict(candidate,start=start),fade_start_seconds=fade_start,fade_stop_seconds=fade_stop,last_vocal_seconds=voice_end,actual_rest_seconds=[float(a),float(b)],raw_tail_level_change_db=delta)

def compose(source,bed,overlap_bars,bpm,incoming_gain):
 bar=240/bpm;npre=round(8*bar*SR);nover=round(overlap_bars*bar*SR);L=bed['candidate']['unit_bars'];period=len(bed['unit']);repetitions=overlap_bars//L+1;tail=bed['full'][period:].copy();vtail=bed['vocal'][period:].copy();taper=round(.003*SR);tail[:taper]*=np.linspace(0,1,taper)[:,None];vtail[:taper]*=np.linspace(0,1,taper)[:,None]
 b=np.concatenate([r.repeat_loop(bed['unit'],repetitions,sample_rate=SR)*incoming_gain['loop'],tail*incoming_gain['track']]);bv=np.concatenate([r.repeat_loop(bed['vocal'][:period],repetitions,sample_rate=SR)*incoming_gain['loop'],vtail*incoming_gain['track']]);a=source['full'];av=source['vocal'];length=npre+len(b);t=np.arange(nover)/SR;fade=source['fade_start_seconds'];stop=source['fade_stop_seconds'];u=smooth((t-fade)/(stop-fade));ga=np.cos(u*np.pi/2);gb=np.where(t<fade,.35*smooth(t/max(fade,1e-6)),.35+.65*np.sin(u*np.pi/2));bass=smooth((t-2*bar)/max(min((overlap_bars-2)*bar,stop)-2*bar,bar));lowfilter=butter(3,180,fs=SR,output='sos')
 def project(aa,bb):
  al=sosfilt(lowfilter,aa,axis=0);bl=sosfilt(lowfilter,bb,axis=0);ca=np.zeros((length,2),np.float64);cb=np.zeros_like(ca);ca[:npre]=aa[:npre];ca[npre:npre+nover]=al[npre:]*(1-bass[:,None])+(aa[npre:]-al[npre:])*ga[:,None];cb[npre:npre+nover]=bl[:nover]*bass[:,None]+(bb[:nover]-bl[:nover])*gb[:,None];cb[npre+nover:]=bb[nover:];return ca,cb
 ca,cb=project(a,b);va,vb=project(av,bv);assert np.max(abs(ca[npre+nover:]))==0
 return dict(source=ca,target=cb,source_vocal=va,target_vocal=vb,mix=ca+cb,incoming=b,source_raw=a,period_frames=period,repetitions=repetitions,release_preview_seconds=(8+overlap_bars+L)*bar,source_out_seconds=8*bar+source['fade_stop_seconds'],fade_start_preview_seconds=8*bar+fade,solo_incoming_bars=L)

def vocal_structure(rendered,bpm,overlap_bars):
 npre=round(8*240/bpm*SR);voices=measure(rendered['source_vocal'][npre:],rendered['target_vocal'][npre:],rendered['source'][npre:],rendered['target'][npre:],bpm,overlap_bars)
 # measure() assumes four bars of pre-roll for its seek timestamps.
 for interval in voices['intervals']:interval['preview_seconds']+=4*240/bpm
 step=.05;n=2400;v=rms_frames(rendered['target_vocal'],n);total=rms_frames(rendered['target'],n);active=(v>=.01)&(v/np.maximum(total,1e-10)>=.25);start=round(rendered['source_out_seconds']/step);run=0;first=None
 for i in range(start,len(active)):
  run=run+1 if active[i] else 0
  if run>=3:first=(i-2)*step;break
 if first is None:raise ValueError('The preview does not reach a clear incoming vocal phrase')
 gap=(first-rendered['source_out_seconds'])/(240/bpm)
 if gap<rendered['solo_incoming_bars']-.1:raise ValueError('Incoming vocal starts before the instrumental breathing room finishes')
 if voices['status']=='vocal_overlap_watch':raise ValueError('Prominent vocals still overlap in the bridge')
 return voices,dict(passed=True,source_out_seconds=rendered['source_out_seconds'],incoming_vocal_seconds=first,incoming_vocal_gap_bars=gap,solo_incoming_bars=rendered['solo_incoming_bars'],loop_release_seconds=rendered['release_preview_seconds'],source_fade_begins_seconds=rendered['fade_start_preview_seconds'],rule='Incoming loop stays instrumental through the handoff; source vocal finishes before its mid/high fade; at least one complete incoming loop plays alone before release.')

def generate(source_id,target_id,overlaps=(8,16),units=(4,2),job_id=None):
 tracks={t['id']:t for t in r.api('tracks')};A,B=tracks[source_id],tracks[target_id]
 native_path=TRAIN/'library_search_v2/native-screen.json'
 if native_path.exists():
  native=json.loads(native_path.read_text())['tracks']
  for track in [A,B]:
   current=r.api(f"analysis/{track['id']}");evidence=native.get(str(track['id']))
   if not evidence or evidence['audio_sha256']!=sha(track['file_path']) or evidence['tagged_bpm']!=current['bpm'] or evidence['status'] not in ['supported','partial']:raise ValueError('The tempo grid needs review before making this loop bridge')
 separator=Separator();am,bm=load_map(A,separator),load_map(B,separator);bpm=am['bpm'];ratio=bpm/bm['bpm']
 if not .94<=ratio<=1.06:raise ValueError('This pair needs more than six percent tempo adjustment')
 gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');decisions=[];beds={}
 for L in units:
  for candidate in entry_candidates(bm,L)[:8]:
   try:beds[L]=build_bed(B,bm,candidate,bpm,gate,separator);decisions.append(dict(side='incoming',unit_bars=L,candidate=candidate,accepted=True));break
   except ValueError as e:decisions.append(dict(side='incoming',unit_bars=L,candidate=candidate,accepted=False,reason=str(e)))
 if not beds:raise ValueError('No clean rhythmic instrumental loop with a following vocal phrase was found')
 exits=exit_candidates(am,max(overlaps));source_by_overlap={};chosen=None
 for exit in exits[:10]:
  try:
   for N in overlaps:source_by_overlap[N]=source_clip(A,am,exit,N,bpm,gate,separator)
   chosen=exit;decisions.append(dict(side='outgoing',candidate=exit,accepted=True));break
  except ValueError as e:source_by_overlap={};decisions.append(dict(side='outgoing',candidate=exit,accepted=False,reason=str(e)))
 if chosen is None:
  write(HERE/f'{source_id}-{target_id}-search-rejected.json',dict(decisions=decisions,exit_candidates=exits))
  raise ValueError('No energetic phrase exit left enough room to finish the outgoing vocal before fading')
 # The same gain applies to every variant of this pair.
 reference=source_by_overlap[min(overlaps)]['full'];bar=240/bpm;npre=round(8*bar*SR);ref=reference[npre-round(4*bar*SR):npre+round(4*bar*SR)];ref_rms=float(np.sqrt(np.mean(ref**2)+1e-12));gains={}
 for L,bed in list(beds.items()):
  info=bed['alignment'];relative=bed['candidate']['first_native_vocal']-info['corrected_start_seconds'];voice_at=float(np.interp(relative,info['source_anchor_seconds'],info['target_anchor_seconds']));lo=round(voice_at*SR);body=bed['full'][lo:lo+round(4*bar*SR)];loop_rms=float(np.sqrt(np.mean(bed['unit']**2)+1e-12));body_rms=float(np.sqrt(np.mean(body**2)+1e-12));lg=float(np.clip(ref_rms/loop_rms,.5,2));tg=float(np.clip(ref_rms/body_rms,.5,2));change=float(20*np.log10(body_rms*tg/(loop_rms*lg)))
  if abs(change)>4:
   decisions.append(dict(side='incoming',unit_bars=L,accepted=False,reason='Loop and following phrase cannot be gain-matched within headroom limits',release_level_change_db=change));beds.pop(L);continue
  gains[L]=dict(loop=lg,track=tg,reference_rms=ref_rms,loop_rms=loop_rms,phrase_rms=body_rms,release_level_change_db=change)
 if not beds:raise ValueError('No loop can maintain level into the incoming vocal phrase')
 variants=[]
 for N in overlaps:
  for L,bed in beds.items():
   if N==max(overlaps) and N!=min(overlaps) and L!=max(beds):continue
   y=compose(source_by_overlap[N],bed,N,bpm,gains[L]);voices,structure=vocal_structure(y,bpm,N);variants.append(dict(rendered=y,source=source_by_overlap[N],bed=bed,bars=N,unit=L,vocals=voices,structure=structure))
 peak=max(float(abs(v['rendered']['mix']).max()) for v in variants);gain=min(1,.95/max(peak,1e-10));previews=[];analyses=[]
 for index,v in enumerate(variants):
  d=v['rendered'];samples=d['mix']*gain;nf=round(.01*SR);samples[:nf]*=np.linspace(0,1,nf)[:,None];samples[-nf:]*=np.linspace(1,0,nf)[:,None];ident=job_id if job_id and len(variants)==1 else uuid.uuid4().hex;path=r.ROOT/f'live_{ident}.wav';assert not path.exists()
  with wave.open(str(path),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(SR);f.writeframes((samples*32767).astype('<i2').tobytes())
  s=v['source'];bed=v['bed'];candidate=bed['candidate'];N=v['bars'];L=v['unit'];label=f'{L}-bar incoming loop · {N}-bar handoff';native_exit=float(np.interp(8*240/bpm+s['fade_stop_seconds'],s['alignment']['target_anchor_seconds'],s['alignment']['source_anchor_seconds'])+s['alignment']['corrected_start_seconds']);selection=dict(source_track=source_id,target_track=target_id,source_title=A['title'],target_title=B['title'],source_start_seconds=s['candidate']['start'],target_start_seconds=candidate['start'],mix_bpm=bpm,target_tempo_factor=ratio,overlap_bars=N,source_section=dict(name='live phrase into instrumental exit',start=s['candidate']['start'],end=native_exit),target_section=dict(name='instrumental entry loop, then natural phrase',start=candidate['start'],end=candidate['end']),selector='phrase_bridge',model_logit=0.,ranking_score=0.)
  p=dict(selection=selection,engine=dict(recipe='phrase_bridge_v1',pre_bars=8,post_bars=bed['post_bars']+L,loop=None,target_loop=dict(bars=L,period_frames=d['period_frames'],repetitions=d['repetitions'],start_seconds=candidate['start'],end_seconds=candidate['end'],release_native_seconds=candidate['end'],boundary=bed['boundary'],predicted_vocal_fraction=bed['loop_vocal_fraction']),source_alignment=s['alignment'],target_alignment=bed['alignment'],incoming_level_gain=gains[L]['track'],incoming_loop_gain=gains[L]['loop'],level_match=gains[L],strategy='phrase_bridge',structure=v['structure'],source_fade_start_seconds=s['fade_start_seconds'],source_fade_stop_seconds=s['fade_stop_seconds'],actual_outgoing_vocal_rest=s['actual_rest_seconds'],source_last_vocal_seconds=s['last_vocal_seconds'],source_tail_level_change_db=s['raw_tail_level_change_db']),eq_enabled=True,audio=dict(path=str(path),frames=len(samples),seconds=len(samples)/SR,sample_rate=SR,gain_applied=gain,peak_after_gain=float(abs(samples).max()),sha256=sha(path)),job_id=ident,created=time.time(),source_audio_sha256=am['audio_sha256'],target_audio_sha256=bm['audio_sha256'],comparison_label=label,selector_version='phrase_bridge_v1',selector_model_sha256=separator.model_sha,planning_version='incoming_instrumental_runway_v1',listener_review='pending',deck_commands_sent=False,comparison=dict(collection_id='phrase-bridges-v1',group_id=f'phrase-{source_id}-{target_id}-v1',variant=chr(65+index),kind='phrase_bridge',idea=f"{A['artist']} → {B['artist']}",style='Instrumental loop handoff',description='The incoming instrumental loop carries the handoff and then plays alone. It releases into the original song only after the old track is fully out. Compare two/four-bar loop beds and eight/sixteen-bar handoffs.',experiment_order=0 if source_id==118 else 1,source_artist=A.get('artist'),target_artist=B.get('artist'),musical_quality='Pending listening review',parameter_replay_supported=False))
  # Deterministic reconstruction from saved source/unit arrays and gain.
  reconstructed=compose(s,bed,N,bpm,gains[L])['mix']*gain;reconstructed[:nf]*=np.linspace(0,1,nf)[:,None];reconstructed[-nf:]*=np.linspace(1,0,nf)[:,None]
  with wave.open(str(path)) as f:pcm=np.frombuffer(f.readframes(f.getnframes()),'<i2').reshape(-1,2)
  assert np.array_equal((reconstructed*32767).astype('<i2'),pcm);assert abs(samples).max()<=.950001
  voice=copy.deepcopy(v['vocals']);voice.update(audio_sha256=p['audio']['sha256'],preview_snapshot_sha256=binding(p),reconstruction_pcm_max_difference=0,scope='Separated vocals projected through the exact phrase-bridge gains; candidate sections were strictly decoded.',model_sha256=separator.model_sha)
  structure=dict(v['structure'],audio_sha256=p['audio']['sha256'],preview_snapshot_sha256=binding(p),reconstruction_pcm_max_difference=0,outgoing_native_start=s['candidate']['start'],outgoing_native_exit=native_exit,outgoing_span_end=s['candidate']['end'],incoming_native_loop_start=candidate['start'],incoming_native_loop_end=candidate['end'],incoming_native_first_vocal=candidate['first_native_vocal'],loop_bars=L,overlap_bars=N)
  analyses.append(dict(vocal=voice,structure=structure));previews.append(p);print('PHRASE_BRIDGE_RENDERED',source_id,target_id,label,'new vocal gap bars',structure['incoming_vocal_gap_bars'],flush=True)
 result=dict(created=time.time(),source=source_id,target=target_id,previews=previews,analyses=analyses,decisions=decisions,all_reconstructions_bit_exact=True,script_sha256=sha(__file__),incoming_gains=gains,common_output_gain=gain,approval_claimed=False);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source',type=int,required=True);parser.add_argument('--target',type=int,required=True);parser.add_argument('--bars',type=int,default=0);parser.add_argument('--entry-loop',type=int,default=0);parser.add_argument('--job-id');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=generate(args.source,args.target,(args.bars,) if args.bars else (8,16),(args.entry_loop,) if args.entry_loop else (4,2),args.job_id);write(args.output,result)
if __name__=='__main__':main()
