"""Reconstruct saved bridge recipes, then measure their actual handoff sections."""
from pathlib import Path
import sys,json,wave,time,copy
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent;sys.path.insert(0,str(HERE))
from engine import *
from deep_merge_analysis import key_estimate,lag_compare

def rebuild(p,separator,tracks,gate):
 s=p['selection'];e=p['engine'];bpm=s['mix_bpm'];bar=240/bpm;npre=round(8*bar*SR);nover=round(s['overlap_bars']*bar*SR)
 for side in ['source','target']:assert sha(tracks[s[side+'_track']]['file_path'])==p[side+'_audio_sha256']
 a,ai=aligned(tracks[s['source_track']],s['source_start_seconds'],8,8+s['overlap_bars'],bpm,gate,npre+nover);av,_=separator.vocals(a);total=(e['target_alignment']['anchor_count']-1)//4;b,bi=aligned(tracks[s['target_track']],s['target_start_seconds'],0,total,bpm,gate,e['target_alignment']['target_frames']);bv,_=separator.vocals(b);L=e['target_loop']['bars'];period=e['target_loop']['period_frames'];source=dict(full=a,vocal=av,fade_start_seconds=e['source_fade_start_seconds'],fade_stop_seconds=e['source_fade_stop_seconds']);bed=dict(candidate={'unit_bars':L},full=b,vocal=bv,unit=b[:period]);gains=dict(loop=e['incoming_loop_gain'],track=e['incoming_level_gain']);d=compose(source,bed,s['overlap_bars'],bpm,gains)
 y=d['mix']*p['audio']['gain_applied'];nf=round(.01*SR);y[:nf]*=np.linspace(0,1,nf)[:,None];y[-nf:]*=np.linspace(1,0,nf)[:,None]
 with wave.open(p['audio']['path']) as f:published=np.frombuffer(f.readframes(f.getnframes()),'<i2').reshape(-1,2)
 assert sha(p['audio']['path'])==p['audio']['sha256'];assert np.array_equal((y*32767).astype('<i2'),published),'Saved recipe does not reproduce audition'
 source_end=npre+int(np.ceil(e['source_fade_stop_seconds']*SR));assert abs(d['source'][source_end:]).max()<1e-12
 nominal_end=round(d['source_out_seconds']*SR);assert abs(d['source'][nominal_end:]).max()*p['audio']['gain_applied']<1/32767
 unit=d['incoming'][:period]
 for i in range(1,d['repetitions']):assert np.array_equal(unit,d['incoming'][i*period:(i+1)*period])
 voices,structure=vocal_structure(d,bpm,s['overlap_bars']);assert voices['status']=='no_sustained_overlap_detected';assert structure['incoming_vocal_gap_bars']>=L-.1
 # Source vocals must stay outside the fade interval.
 raw=rms_frames(a[npre:],2400);voice=rms_frames(av[npre:],2400);mask=(voice>=.01)&(voice/np.maximum(raw,1e-10)>=.25);ts=np.arange(len(mask))*.05;during=(ts>=e['source_fade_start_seconds'])&(ts<e['source_fade_stop_seconds']);assert not mask[during].any()
 return d,voices,structure

def analyze(p,separator,tracks,gate):
 d,voices,structure=rebuild(p,separator,tracks,gate);s=p['selection'];bpm=s['mix_bpm'];bar=240/bpm;npre=round(8*bar*SR);nover=round(s['overlap_bars']*bar*SR);gain=p['audio']['gain_applied'];a=d['source_raw'][npre:]*gain;b=d['incoming'][:nover]*gain;ca=d['source'][npre:npre+nover]*gain;cb=d['target'][npre:npre+nover]*gain;ra,rb,ea,eb=features(a),features(b),features(ca),features(cb);bars=[]
 for i in range(s['overlap_bars']):
  lo,hi=round(i*bar*SR),round((i+1)*bar*SR);mix=ca[lo:hi]+cb[lo:hi];bars.append(dict(bar=i+1,preview_seconds=(8+i)*bar,raw=compare(ra,rb,i*bar,(i+1)*bar,bpm),rendered=compare(ea,eb,i*bar,(i+1)*bar,bpm),mix_rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(mix**2)),1e-12))),source_loop_cycle=None,target_loop_cycle=i//p['engine']['target_loop']['bars']+1))
 fa,fb=percussion_features(a),percussion_features(b);aa,bb=track_analysis(fa,bpm,s['overlap_bars']),track_analysis(fb,bpm,s['overlap_bars']);rhythm=rhythm_assessment(fa,fb,aa,bb,bpm,bars);harmony=harmonic_assessment(fa,fb,aa,bb,bars,bpm)
 # Describe the harmonic material that is actually audible in the overlap.
 for f,energy,track in [(fa,ea,aa),(fb,eb,bb)]:track['key']=key_estimate(f['chroma'],weights=np.interp(f['chroma_time'],energy['time'],energy['energy']))
 harmony.update(source=aa['key'],target=bb['key']);harmony['both_keys_confident']=aa['key']['confident'] and bb['key']['confident'];harmony['camelot_relation']=key_relation(aa['key']['key'],bb['key']['key'])
 harmony['state']='related_local_keys' if harmony['both_keys_confident'] and harmony['camelot_relation']=='compatible' else 'different_local_keys' if harmony['both_keys_confident'] else 'key_uncertain';harmony['text']=f"Audible loop/exit key candidates: {aa['key']['key']} → {bb['key']['key']}. "+('Related local key estimates.' if harmony['state']=='related_local_keys' else 'Different estimates; listen for melodic tension.' if harmony['state']=='different_local_keys' else 'Tonal confidence is limited; judge the instrumental blend by listening.')
 windows=[]
 for i in range(s['overlap_bars']-1):
  mask=(fa['time']>=i*bar)&(fa['time']<(i+2)*bar);v=lag_compare(fa['low'][mask],fb['low'][mask],bpm);v.update(start_bar=i+1,both_present_fraction=float(np.mean([r['rendered']['coaudible_fraction'] for r in bars[i:i+2]])));windows.append(v)
 supported=[v for v in windows if v['both_present_fraction']>=.15 and v['peak_correlation'] is not None and v['peak_correlation']>=.3];lag=float(np.median([v['offset_ms'] for v in supported])) if supported else None;mad=float(np.median(abs(np.array([v['offset_ms'] for v in supported])-lag))) if supported else None;full_periods=[period(f['full'],bpm) for f in [fa,fb]];estimates=[];strong=[]
 for f,analysis,full in [(fa,aa,full_periods[0]),(fb,bb,full_periods[1])]:
  best=max([analysis['near_renderer_bpm'],full],key=lambda r:r['correlation']);wide=analysis['unconstrained_tempo']['full'];strong.append(best['correlation']>=.4 and wide is not None and min(abs(wide*x-bpm) for x in [.5,1,2])<2);estimates.append(best['bpm'])
 if all(strong) and lag is not None and len(supported)>=3 and mad<=20 and abs(estimates[0]-estimates[1])<=.8 and abs(lag)<=40:rhythm.update(state='timing_supported',text=f'Loop and outgoing pulse estimates agree within {abs(estimates[0]-estimates[1]):.2f} BPM; shared low-drum patterns differ by about {abs(lag):.0f} ms.')
 elif all(strong):rhythm.update(state='tempo_agrees_rhythm_varies',text=f'Loop and outgoing pulse estimates are near the {bpm:.1f}-BPM target; attack-pattern alignment is less certain.')
 rhythm.update(sliding_low_drum_windows=windows,stable_drum_windows=len(supported),median_window_offset_ms=lag,window_offset_mad_ms=mad,full_band_periods=full_periods,measured_pulse_estimates_bpm=estimates)
 common=dict(audio_sha256=p['audio']['sha256'],preview_snapshot_sha256=binding(p),reconstruction_pcm_max_difference=0);overlap=dict(common,overlap_bars=s['overlap_bars'],summary=dict(raw=compare(ra,rb,0,s['overlap_bars']*bar,bpm),rendered=compare(ea,eb,0,s['overlap_bars']*bar,bpm)),bars=bars);deep=dict(common,rhythm=rhythm,harmony=harmony,source_analysis=aa,target_analysis=bb);voices.update(common,model_sha256=separator.model_sha);structure.update(common,outgoing_native_start=s['source_start_seconds'],outgoing_native_exit=s['source_section']['end'],incoming_native_loop_start=s['target_start_seconds'],incoming_native_loop_end=p['engine']['target_loop']['end_seconds'],loop_bars=p['engine']['target_loop']['bars'],overlap_bars=s['overlap_bars']);return dict(preview=p,overlap=overlap,deep=deep,vocal=voices,structure=structure)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();data=json.loads(args.input.read_text());separator=Separator();tracks={t['id']:t for t in r.api('tracks')};gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');result=dict(created=time.time(),entries=[])
 for p in data['previews']:
  result['entries'].append(analyze(p,separator,tracks,gate));print('BRIDGE_CHECKED',p['selection']['source_track'],p['comparison']['variant'],result['entries'][-1]['deep']['rhythm']['state'],flush=True)
 result.update(completed=time.time(),all_reconstructions_bit_exact=True);write(args.output,result)
if __name__=='__main__':main()
