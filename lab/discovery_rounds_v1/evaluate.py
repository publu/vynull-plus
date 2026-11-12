"""Render and measure one discovery candidate, without publishing or voting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import sys,json,time,uuid,hashlib,wave,argparse,copy
import numpy as np,joblib
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN/'library_lab_v1'))
from render_group import r,mix,sha,write
from analyze_overlaps import features,compare,rebuild,binding,SR
from deep_merge_analysis import percussion_features,track_analysis,rhythm_assessment,harmonic_assessment,period,lag_compare
from summarize_deep import octave_nearest,original_check

def render(plan,spec,round_number,label,mode='blend',bars=8,loop_bars=0):
 tracks={t['id']:t for t in r.api('tracks')};s=copy.deepcopy(plan['selection']);s['overlap_bars']=bars;source,target=tracks[s['source_track']],tracks[s['target_track']]
 for key,t in [('source',source),('target',target)]:assert sha(t['file_path'])==plan[key+'_audio_sha256']
 gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');bpm=s['mix_bpm'];bar=240/bpm;npre=round(4*bar*SR);nover=round(bars*bar*SR)
 a,ai=r.aligned_stem(source,s['source_start_seconds'],4,4+bars,bpm,gate,npre+nover);b,bi=r.aligned_stem(target,s['target_start_seconds'],0,bars+4,bpm,gate,nover+npre);loop=None
 if loop_bars:
  cache=json.loads((TRAIN/'semantic_pair_v1/halo_features'/f"{s['source_track']}_loops_{loop_bars}.json").read_text());assert cache['audio_sha256']==plan['source_audio_sha256'];assert cache['continuity_model_sha256']==sha(TRAIN/'raveform_audio_v1/artifacts/loop_region_continuity.joblib')
  row=next((v for v in cache['candidates'] if abs(v['start']-s['source_start_seconds'])<1e-5 and v['accepted']),None)
  if row is None:raise ValueError('Selected part did not pass loop pre-screen')
  nunit=round(loop_bars*bar*SR);unit,ui=r.aligned_stem(source,s['source_start_seconds'],0,loop_bars,bpm,gate,nunit);boundary=r.loop_boundary_metrics(unit,SR,bar)
  if boundary['boundary_level_difference_db']>8:raise ValueError('Rendered loop failed boundary check')
  a=np.concatenate([a[:npre],r.repeat_loop(unit,nover//nunit,sample_rate=SR)]);ai=dict(ai,loop_unit_alignment=ui);loop=dict(row,**boundary,period_frames=nunit,edge_taper_ms=3,musical_review='pending')
 incoming_gain=float(np.clip(np.sqrt(np.mean(a[npre:]**2)+1e-12)/np.sqrt(np.mean(b[:nover]**2)+1e-12),.5,2));b*=incoming_gain;y=mix(a,b,npre,nover,mode);peak=float(abs(y).max());gain=min(1,.95/max(peak,1e-10));y*=gain;nf=round(.01*SR);y[:nf]*=np.linspace(0,1,nf)[:,None];y[-nf:]*=np.linspace(1,0,nf)[:,None]
 assert np.isfinite(y).all() and abs(y).max()<=.950001 and len(y)==npre+nover+npre
 ident=uuid.uuid4().hex;path=r.ROOT/f'live_{ident}.wav';assert not path.exists()
 with wave.open(str(path),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(SR);f.writeframes((y*32767).astype('<i2').tobytes())
 s['source_section']['end']=s['source_start_seconds']+(loop_bars or bars)*bar;s['target_section']['end']=s['target_start_seconds']+bars*bar*s['target_tempo_factor']
 return dict(selection=s,engine=dict(pre_bars=4,post_bars=4,loop=loop,target_loop=None,source_alignment=ai,target_alignment=bi,incoming_level_gain=incoming_gain,strategy=mode,level_matching='Overlap RMS matching capped at ±6 dB; each audition peak-limited to 0.95'),eq_enabled=mode=='blend',audio=dict(path=str(path),seconds=len(y)/SR,frames=len(y),sample_rate=SR,peak_before_gain=peak,gain_applied=gain,peak_after_gain=float(abs(y).max()),sha256=sha(path)),comparison_label=label,job_id=ident,created=time.time(),source_audio_sha256=plan['source_audio_sha256'],target_audio_sha256=plan['target_audio_sha256'],selector_model_sha256=plan['model_sha256'],selector_version=plan['selector_version'],planning_version='adaptive_discovery_v1',listener_review='pending',deck_commands_sent=False,comparison=dict(group_id=f"discover-{s['source_track']}-{s['target_track']}-v1",variant={1:'A',2:'B',3:'C',4:'D'}[round_number],kind='library_discovery',collection_id='discovery-rounds-v1',idea=spec['idea'],style=spec['style'],description='Compare candidates found over three search rounds: initial parts, alternative sections, then a shorter blend or loop/phrase cut. Beat and key checks describe the measured evidence; musical fit awaits listening.',experiment_order=spec['order'],source_artist=source.get('artist'),target_artist=target.get('artist'),discovery_round=min(round_number,3),musical_quality='Pending listening review',parameter_replay_supported=False))

def analyze(p):
 tracks={t['id']:t for t in r.api('tracks')};gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');a,b,ca,cb,diff=rebuild(p,tracks,gate);s=p['selection'];bpm=s['mix_bpm'];bar=240/bpm
 ra,rb,ea,eb=features(a),features(b),features(ca),features(cb);bars=[]
 for i in range(s['overlap_bars']):
  lo,hi=round(i*bar*SR),round((i+1)*bar*SR);segment=ca[lo:hi]+cb[lo:hi]
  bars.append(dict(bar=i+1,preview_seconds=(4+i)*bar,raw=compare(ra,rb,i*bar,(i+1)*bar,bpm),rendered=compare(ea,eb,i*bar,(i+1)*bar,bpm),mix_rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(segment**2)),1e-12))),source_loop_cycle=i//p['engine']['loop']['bars']+1 if p['engine'].get('loop') else None,target_loop_cycle=None))
 overlap=dict(audio_sha256=p['audio']['sha256'],preview_snapshot_sha256=binding(p),reconstruction_pcm_max_difference=diff,overlap_bars=s['overlap_bars'],group_id=p['comparison']['group_id'],variant=p['comparison']['variant'],summary=dict(raw=compare(ra,rb,0,s['overlap_bars']*bar,bpm),rendered=compare(ea,eb,0,s['overlap_bars']*bar,bpm)),bars=bars)
 fa,fb=percussion_features(a),percussion_features(b);aa,bb=track_analysis(fa,bpm,s['overlap_bars']),track_analysis(fb,bpm,s['overlap_bars']);rhythm=rhythm_assessment(fa,fb,aa,bb,bpm,bars);harmony=harmonic_assessment(fa,fb,aa,bb,bars,bpm)
 windows=[]
 for i in range(s['overlap_bars']-1):
  mask=(fa['time']>=i*bar)&(fa['time']<(i+2)*bar);v=lag_compare(fa['low'][mask],fb['low'][mask],bpm);v.update(start_bar=i+1,both_present_fraction=float(np.mean([x['rendered']['coaudible_fraction'] for x in bars[i:i+2]])));windows.append(v)
 supported=[v for v in windows if v['both_present_fraction']>=.15 and v['peak_correlation'] is not None and v['peak_correlation']>=.3];lag=float(np.median([v['offset_ms'] for v in supported])) if supported else None;mad=float(np.median(abs(np.array([v['offset_ms'] for v in supported])-lag))) if supported else None
 strong=[];estimates=[];full_periods=[]
 for f,analysis in [(fa,aa),(fb,bb)]:
  full=period(f['full'],bpm);full_periods.append(full);best=max([analysis['near_renderer_bpm'],full],key=lambda v:v['correlation']);wide=analysis['unconstrained_tempo']['full'];strong.append(best['correlation']>=.4 and wide is not None and abs(octave_nearest(wide,bpm)-bpm)<2);estimates.append(best['bpm'])
 native=json.loads((HERE/'native-screen.json').read_text());checks={side:original_check(native['tracks'][str(s[side+'_track'])]) for side in ['source','target']};rhythm.update(sliding_low_drum_windows=windows,stable_drum_windows=len(supported),median_window_offset_ms=lag,window_offset_mad_ms=mad,full_band_periods=full_periods,measured_pulse_estimates_bpm=estimates,long_original_checks=checks)
 if any(v['grid_warning'] for v in checks.values()):rhythm.update(state='tempo_grid_warning',text='Longer original excerpts disagree with a stored tempo grid. Review the grid before trusting this merge.')
 elif all(strong) and lag is not None and len(supported)>=3 and mad<=20:
  gap=estimates[1]-estimates[0];rhythm['pulse_bpm_gap']=gap
  if abs(gap)>.8:rhythm.update(state='possible_tempo_drift',text=f'Measured pulse-speed difference {gap:+.2f} BPM: listen for drift.')
  elif abs(lag)>40:rhythm.update(state='consistent_drum_offset',text=f'Pulse speeds agree within {abs(gap):.2f} BPM, but low-drum patterns are displaced about {lag:+.0f} ms. Check for a doubled beat or intentional syncopation.')
  else:rhythm.update(state='timing_supported',text=f'Pulse speeds agree within {abs(gap):.2f} BPM; low-drum patterns align within about {abs(lag):.0f} ms across {len(supported)} audible windows.')
 elif all(strong):rhythm.update(state='tempo_agrees_rhythm_varies',text=f'Pulse speeds agree within {abs(estimates[1]-estimates[0]):.2f} BPM; shared attack placement is less certain.')
 deep=dict(audio_sha256=p['audio']['sha256'],preview_snapshot_sha256=binding(p),reconstruction_pcm_max_difference=diff,group_id=p['comparison']['group_id'],variant=p['comparison']['variant'],source_title=s['source_title'],target_title=s['target_title'],source_track=s['source_track'],target_track=s['target_track'],label=p['comparison_label'],rhythm=rhythm,harmony=harmony,source_analysis=aa,target_analysis=bb)
 return overlap,deep

def priority(deep):
 timing={'timing_supported':4,'tempo_agrees_rhythm_varies':3,'rhythm_uncertain':2,'consistent_drum_offset':1,'possible_tempo_drift':0,'tempo_grid_warning':-1}.get(deep['rhythm']['state'],0)
 harmonic={'related_local_keys':2,'key_uncertain':1,'different_local_keys':0}.get(deep['harmony']['state'],0)
 corr=deep['rhythm']['low_band_correlation']['peak_correlation'] or 0
 return [timing,harmonic,float(corr)]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);args=parser.parse_args();request=json.loads(Path(args.request).read_text());dest=Path(request['result'])
 if dest.exists():raise RuntimeError('Immutable candidate already evaluated; inspect receipt')
 plan=json.loads(Path(request['plan']).read_text());p=render(plan,request['spec'],request['variant_number'],request['label'],request.get('mode','blend'),request.get('bars',8),request.get('loop_bars',0));overlap,deep=analyze(p);result=dict(created=time.time(),request=request,preview=p,overlap=overlap,deep=deep,screen_priority=priority(deep),script_sha256=sha(__file__),quality_approved=False);write(dest,result);print('CANDIDATE_READY',p['comparison']['group_id'],p['comparison']['variant'],deep['rhythm']['state'],deep['harmony']['state'],flush=True)
if __name__=='__main__':main()
