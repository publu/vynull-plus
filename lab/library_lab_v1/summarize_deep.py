"""Cross-check merge evidence against longer originals and sliding drum windows."""
from pathlib import Path
import sys,json,time,copy
import numpy as np,joblib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from deep_merge_analysis import *

def octave_nearest(value,target):return min([value*.5,value,value*2],key=lambda v:abs(v-target))
def original_check(record):
 values=[octave_nearest(e['wide_pulse_bpm'],record['tagged_bpm']) for e in record['excerpts']]
 stable=abs(values[0]-values[1])<=1.5;mean=float(np.mean(values));ratio=mean/record['tagged_bpm']
 familiar=min([.5,2/3,.75,1,4/3,1.5,2],key=lambda v:abs(v-ratio));metrical_alias=abs(ratio-familiar)<.025 and familiar!=1
 return dict(original_tagged_bpm=record['tagged_bpm'],independent_pulse_bpm=mean,stable_across_excerpts=stable,relative_difference_percent=float((ratio-1)*100),metrical_alias_possible=metrical_alias,grid_warning=bool(stable and abs(ratio-1)>.08 and not metrical_alias),audio_sha256=record['audio_sha256'])
def main():
 report=json.loads((HERE/'deep-merge-analysis.json').read_text());original=json.loads((HERE/'original-pulse-check.json').read_text());overlaps=json.loads((HERE/'overlap-analysis.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text());tracks={t['id']:t for t in r.api('tracks')};gate=joblib.load(HERE.parent/'part_alignment_v1/artifacts/beat_repair_gate.joblib')
 report['version']='deep_merge_librosa_v2';report['crosschecked']=time.time();report['summary_rules']=dict(strong_pulse_correlation=.4,wide_tempo_must_support_octave_near_grid=True,minimum_stable_drum_windows=3,window_lag_mad_ms=20,long_original_warning_percent=8,long_original_consistency_bpm=1.5,common_meter_aliases_excluded_from_grid_warning=[.5,2/3,.75,4/3,1.5,2])
 for spec in protocol['experiments']:
  for p in json.loads((HERE/(spec['id']+'.json')).read_text())['previews']:
   key=p['audio']['sha256'];e=report['previews'][key];rhythm=e['rhythm'];rhythm['initial_screen']=dict(state=rhythm['state'],text=rhythm['text']);s=p['selection'];bpm=s['mix_bpm'];bar=240/bpm;a,b,ca,cb,_=rebuild(p,tracks,gate);fa,fb=percussion_features(a),percussion_features(b);old=overlaps['previews'][key];windows=[]
   for i in range(max(0,s['overlap_bars']-1)):
    mask=(fa['time']>=i*bar)&(fa['time']<(i+2)*bar);v=lag_compare(fa['low'][mask],fb['low'][mask],bpm);v.update(start_bar=i+1,both_present_fraction=float(np.mean([x['rendered']['coaudible_fraction'] for x in old['bars'][i:i+2]])));windows.append(v)
   supported_windows=[v for v in windows if v['both_present_fraction']>=.15 and v['peak_correlation'] is not None and v['peak_correlation']>=.3]
   lag=float(np.median([v['offset_ms'] for v in supported_windows])) if supported_windows else None
   mad=float(np.median(abs(np.array([v['offset_ms'] for v in supported_windows])-lag))) if supported_windows else None
   strong=[];estimates=[];full_periods=[]
   for f,analysis in [(fa,e['source_analysis']),(fb,e['target_analysis'])]:
    full=period(f['full'],bpm);full_periods.append(full);best=max([analysis['near_renderer_bpm'],full],key=lambda v:v['correlation']);wide=analysis['unconstrained_tempo']['full'];supported=best['correlation']>=.4 and wide is not None and abs(octave_nearest(wide,bpm)-bpm)<2
    strong.append(supported);estimates.append(best['bpm'])
   rhythm.update(sliding_low_drum_windows=windows,stable_drum_windows=len(supported_windows),median_window_offset_ms=lag,window_offset_mad_ms=mad,full_band_periods=full_periods,measured_pulse_estimates_bpm=estimates)
   checks={}
   for side,tid in [('source',s['source_track']),('target',s['target_track'])]:
    native=original['tracks'].get(str(tid))
    if native:
     assert native['audio_sha256']==p[side+'_audio_sha256'];checks[side]=original_check(native)
   rhythm['long_original_checks']=checks
   warnings=[(side,v) for side,v in checks.items() if v['grid_warning']]
   if warnings:
    side,v=warnings[0];rendered_pulse=v['independent_pulse_bpm']*(s['target_tempo_factor'] if side=='target' else 1)
    rhythm.update(state='tempo_grid_warning',text=f"{'Incoming' if side=='target' else 'Outgoing'} tempo grid is suspect: tag {v['original_tagged_bpm']:.1f} BPM, longer original excerpts support a pulse around {v['independent_pulse_bpm']:.1f}. That pulse is about {rendered_pulse:.1f} BPM in this {bpm:.1f}-BPM merge. Check the grid before trusting the blend.")
   elif all(strong) and lag is not None and len(supported_windows)>=3 and mad<=20:
    gap=estimates[1]-estimates[0];rhythm['pulse_bpm_gap']=gap
    if abs(gap)>.8:rhythm.update(state='possible_tempo_drift',text=f'Measured drum-pulse speeds differ by {gap:+.2f} BPM despite a shared render target. Check for drift.')
    elif abs(lag)>40:rhythm.update(state='consistent_drum_offset',text=f'Drum-pulse speeds agree within {abs(gap):.2f} BPM, but repeated low-drum patterns are displaced by about {lag:+.0f} ms. Listen for a doubled beat or intentional syncopation.')
    else:rhythm.update(state='timing_supported',text=f'Drum-pulse speeds agree within {abs(gap):.2f} BPM; repeated low-drum patterns line up within about {abs(lag):.0f} ms across {len(supported_windows)} audible windows. No obvious timing problem in these measurements.')
   elif all(strong):rhythm.update(state='tempo_agrees_rhythm_varies',text=f'Drum-pulse speeds agree within {abs(estimates[1]-estimates[0]):.2f} BPM; attack placement varies or has weak shared evidence. Matching speed alone does not establish a clean beat blend.')
   print('FINDING',e['group_id'],e['variant'],rhythm['state'],flush=True)
 report['summary_script_sha256']=sha(__file__);report['original_check_sha256']=sha(HERE/'original-pulse-check.json');write(HERE/'deep-merge-analysis-v2.json',report)
if __name__=='__main__':main()
