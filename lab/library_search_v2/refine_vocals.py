"""Try low-vocal outgoing loops for three flagged pairs, then measure the mix."""
from pathlib import Path
import os,sys,json,copy,time,subprocess
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(HERE))
from vocal_audit import Separator,r,sha,write,np,joblib,rms_frames,RULES,SR,audit

def main():
 source=json.loads((TRAIN/'discovery_rounds_v1/published.json').read_text());tracks={t['id']:t for t in r.api('tracks')};measured=json.loads((TRAIN.parent/'review/measured-audio.json').read_text());gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');separator=Separator();report=dict(created=time.time(),decisions=[],previews=[],results=[],vocal_entries={},model_sha256=separator.model_sha)
 for a,b in [(118,144),(74,150),(117,125)]:
  winner=next(p for p in source['previews'] if p['selection']['source_track']==a and p['selection']['target_track']==b and p['comparison']['screening_choice']);selection=winner['selection'];bpm=selection['mix_bpm'];bar=240/bpm
  subprocess.run([os.environ.get('LAB_PYTHON',str(TRAIN/'.venv/bin/python')),str(TRAIN/'part_alignment_v1/prepare_loops.py'),'--source',str(a),'--bars','4'],check=True)
  cache=json.loads((TRAIN/'semantic_pair_v1/halo_features'/f'{a}_loops_4.json').read_text())
  with np.load(TRAIN/'semantic_pair_v1/halo_features'/f'{a}.npz') as data:config=json.loads(str(data['config']));times=data['times'];downbeats=data['downbeats']
  allowed=[row for row in cache['candidates'] if row['accepted'] and row['start']>=4*bar and row['start']+8*bar<config['duration']-.1 and int(np.argmin(abs(times-row['start'])))*4+8<len(downbeats)]
  choices=[]
  if allowed:
   candidates=[min(allowed,key=lambda v:v['start']),min(allowed,key=lambda v:abs(v['start']-selection['source_start_seconds'])),max(allowed,key=lambda v:v['start']),max(allowed,key=lambda v:v['structural_continuity_probability'])]
   for row in candidates:
    if not any(abs(row['start']-v['start'])<1e-5 for v in choices):choices.append(row)
  decisions=[];chosen=None
  for row in choices:
   try:
    unit,_=r.aligned_stem(tracks[a],row['start'],0,4,bpm,gate,round(4*bar*SR));vocals,meta=separator.vocals(unit);raw=rms_frames(unit,2400);voice=rms_frames(vocals,2400);fraction=float(np.mean((voice>=.01)&(voice/np.maximum(raw,1e-10)>=.25)));boundary=r.loop_boundary_metrics(unit,SR,bar);accepted=fraction<=.03 and boundary['boundary_level_difference_db']<=8;decisions.append(dict(start=row['start'],predicted_vocal_active_fraction=fraction,rendered_boundary_db=boundary['boundary_level_difference_db'],accepted=accepted))
    if accepted:chosen=row;break
   except Exception as e:decisions.append(dict(start=row['start'],accepted=False,error=str(e)))
  report['decisions'].append(dict(source=a,target=b,candidates=decisions,chosen_start=chosen['start'] if chosen else None))
  if chosen:
   plan=json.loads((TRAIN/'discovery_rounds_v1'/f'{a}-{b}-plan.json').read_text());s=copy.deepcopy(selection);nearest=min(measured[str(a)]['sections'],key=lambda v:abs(v['start_seconds']-chosen['start']));s.update(source_start_seconds=chosen['start'],source_section=dict(name='screened low-vocal loop',start=chosen['start'],end=chosen['start']+4*bar),measured_source_key=nearest.get('key'),overlap_bars=8,selector='vocal_guard_loop',model_logit=0.,ranking_score=0.);s.pop('source_function_probabilities',None);s['target_section']['end']=s['target_start_seconds']+8*bar*s['target_tempo_factor'];plan.update(selection=s,alternatives=[],requested_loop_bars=4,selector_version='vocal_guard_loop_v1',planning_version='vocal_screened_source_loop_v1',vocal_screen_model_sha256=separator.model_sha);planpath=HERE/f'refined-{a}-{b}-plan.json';write(planpath,plan)
   spec=json.loads((TRAIN/'discovery_rounds_v1'/f'{a}-{b}-1.json').read_text())['request']['spec'];dest=HERE/f'refined-{a}-{b}.json';request=dict(spec=spec,plan=str(planpath),variant_number=4,label='Low-vocal loop bridge · 8-bar blend',bars=8,mode='blend',loop_bars=4,result=str(dest));requestpath=HERE/f'refined-{a}-{b}-request.json';write(requestpath,request)
   try:
    if not dest.exists():subprocess.run([os.environ.get('LAB_PYTHON',str(TRAIN/'.venv/bin/python')),'-u',str(TRAIN/'discovery_rounds_v1/evaluate.py'),'--request',str(requestpath)],check=True)
    result=json.loads(dest.read_text());p=result['preview'];entry=audit(p,separator,tracks,gate);report['previews'].append(p);report['results'].append(str(dest));report['vocal_entries'][p['audio']['sha256']]=entry;print('VOCAL_REFINED',a,b,entry['status'],entry['simultaneous_vocal_seconds'],flush=True)
   except Exception as e:report['decisions'][-1]['render_error']=str(e)
  else:print('NO_LOW_VOCAL_LOOP_FOUND',a,b,flush=True)
  write(HERE/'vocal-refinement-progress.json',report)
 report.update(completed=time.time());write(HERE/'vocal-refinements.json',report)
if __name__=='__main__':main()
