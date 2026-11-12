"""Preview API boundary: validated IDs, bounded queue, deduplication and recovery."""
import tempfile,json,os
from pathlib import Path
import mix_jobs as jobs

class Queue:
 def __init__(self):self.calls=[]
 def submit(self,*args):self.calls.append(args)

with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);jobs.TRAINING=root/'training';cache=jobs.TRAINING/'semantic_pair_v1/halo_features';cache.mkdir(parents=True)
 for i in range(1,7):(cache/f'{i}.npz').touch()
 tracks={i:{'id':i} for i in range(1,7)};jobs.POOL=Queue()
 request={'source':1,'target':2,'method':'semantic','bars':8,'loop_bars':0}
 first=jobs.create(root,request,tracks);second=jobs.create(root,request,tracks)
 assert first['id']==second['id'] and len(jobs.POOL.calls)==1
 assert jobs.read(root/'mix-jobs'/first['id']/'status.json')['state']=='queued'
 assert 'owner_pid' not in jobs.public(first)
 for bad in [dict(request,source=True),dict(request,source=999),dict(request,target=1),dict(request,method='shell'),dict(request,loop_bars=3),dict(request,bars=999),[]]:
  try:jobs.create(root,bad,tracks)
  except ValueError:pass
  else:raise AssertionError('Invalid preview request accepted')
 for target in [3,4,5]:jobs.create(root,dict(request,target=target),tracks)
 try:jobs.create(root,dict(request,target=6),tracks)
 except ValueError as e:assert 'queued' in str(e)
 else:raise AssertionError('Unbounded render queue')
 status=root/'mix-jobs'/first['id']/'status.json';first['owner_pid']=999999999;jobs.save(status,first)
 recovered=jobs.read(status);assert recovered['state']=='failed' and 'interrupted' in recovered['error']
 assert json.loads(status.read_text())['state']=='failed'
 # A missing/stale feature cache is repaired once; unrelated planner failures
 # must fail directly rather than invoking the encoder or masking the error.
 original_run=jobs.subprocess.run;original_completed=jobs.completed
 retry=root/'retry';retry.mkdir();calls=[];attempt=[0]
 state=dict(request,id='a'*32,state='queued',created=1,updated=1,owner_pid=os.getpid())
 jobs.save(retry/'status.json',state)
 def fake_run(command,**kwargs):
  script=Path(command[2]).name;calls.append(script)
  if script=='halo_planner.py':
   attempt[0]+=1
   if attempt[0]==1:
    kwargs['stderr'].write(b'ValueError: The beat grid changed; music features need refresh\n')
    raise jobs.subprocess.CalledProcessError(1,command)
  if script=='render_live.py':(retry/'result.json').write_text('{"selection":{}}')
 jobs.subprocess.run=fake_run;jobs.completed=lambda s,p:dict(s,state='complete')
 try:jobs.run(retry)
 finally:jobs.subprocess.run=original_run;jobs.completed=original_completed
 assert calls==['halo_planner.py','extract_halo.py','halo_planner.py','render_live.py'],calls
 assert json.loads((retry/'status.json').read_text())['state']=='complete'
 failed=root/'unsupported';failed.mkdir();jobs.save(failed/'status.json',state);calls=[]
 def unsupported(command,**kwargs):
  calls.append(Path(command[2]).name);kwargs['stderr'].write(b'ValueError: These songs need more than six percent tempo adjustment\n')
  raise jobs.subprocess.CalledProcessError(1,command)
 jobs.subprocess.run=unsupported
 try:jobs.run(failed)
 finally:jobs.subprocess.run=original_run
 assert calls==['halo_planner.py'] and json.loads((failed/'status.json').read_text())['state']=='failed'
 loop=root/'loop';loop.mkdir();loop_state=dict(state,loop_bars=4);jobs.save(loop/'status.json',loop_state);calls=[];attempt=[0]
 def loop_run(command,**kwargs):
  script=Path(command[2]).name;calls.append(script)
  if script=='halo_planner.py':
   assert '--loop-bars' in command and command[command.index('--loop-bars')+1]=='4'
   attempt[0]+=1
   if attempt[0] in [1,2]:
    message='Music features for this song are still being prepared' if attempt[0]==1 else 'Loop analysis needs refresh'
    kwargs['stderr'].write(('ValueError: '+message+'\n').encode());raise jobs.subprocess.CalledProcessError(1,command)
  if script=='prepare_loops.py':assert command[-2:]==['--bars','4']
  if script=='render_live.py':(loop/'result.json').write_text('{"selection":{}}')
 jobs.subprocess.run=loop_run;jobs.completed=lambda s,p:dict(s,state='complete')
 try:jobs.run(loop)
 finally:jobs.subprocess.run=original_run;jobs.completed=original_completed
 assert calls==['halo_planner.py','extract_halo.py','halo_planner.py','prepare_loops.py','halo_planner.py','render_live.py'],calls
 assert json.loads((loop/'status.json').read_text())['state']=='complete'
 print('PASS: validated inputs, deduplication, bounded queue, recovery, stale-feature refresh/retry, unrelated failures stay failures')

with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);jobs.POOL=Queue();tracks={i:{'id':i} for i in [1,2]}
 request=dict(source=1,target=2,method='semantic',bars=8,loop_bars=0)
 plain=jobs.create(root,request,tracks);status=root/'mix-jobs'/plain['id']/'status.json'
 legacy=json.loads(status.read_text());del legacy['phase_align'];jobs.save(status,legacy)
 assert jobs.create(root,request,tracks)['id']==plain['id']
 checked=jobs.create(root,dict(request,phase_align=True),tracks)
 assert checked['id']!=plain['id'] and jobs.create(root,dict(request,phase_align=True),tracks)['id']==checked['id']
 for value in [1,0,'true',None]:
  try:jobs.create(root,dict(request,phase_align=value),tracks)
  except ValueError:pass
  else:raise AssertionError('Non-boolean phase option accepted')
 folder=root/'checked';folder.mkdir();state=dict(checked,state='queued',owner_pid=os.getpid());jobs.save(folder/'status.json',state)
 calls=[];original_run=jobs.subprocess.run;original_completed=jobs.completed
 def checked_run(command,**kwargs):
  script=Path(command[2]).name;calls.append(script)
  if script=='halo_planner.py':(folder/'plan.json').write_text('{"selection":{}}')
  if script=='adjust_plan.py':
   assert Path(command[command.index('--plan')+1]).name=='plan-unadjusted.json'
   (folder/'plan.json').write_text('{"selection":{"bar_phase_alignment":{"source":{"applied":false},"target":{"applied":false}}}}')
  if script=='render_live.py':
   assert 'bar_phase_alignment' in json.loads((folder/'plan.json').read_text())['selection']
   (folder/'result.json').write_text('{"selection":{}}')
 jobs.subprocess.run=checked_run;jobs.completed=lambda s,p:dict(s,state='complete')
 try:jobs.run(folder)
 finally:jobs.subprocess.run=original_run;jobs.completed=original_completed
 assert calls==['halo_planner.py','adjust_plan.py','render_live.py']
 assert json.loads((folder/'status.json').read_text())['state']=='complete'
 folder=root/'phase-failed';folder.mkdir();jobs.save(folder/'status.json',state);calls=[]
 def phase_failed(command,**kwargs):
  script=Path(command[2]).name;calls.append(script)
  if script=='halo_planner.py':(folder/'plan.json').write_text('{"selection":{}}')
  if script=='adjust_plan.py':
   kwargs['stderr'].write(b'ValueError: Bar-phase model has not passed its offline gate\n')
   raise jobs.subprocess.CalledProcessError(1,command)
 jobs.subprocess.run=phase_failed
 try:jobs.run(folder)
 finally:jobs.subprocess.run=original_run
 assert calls==['halo_planner.py','adjust_plan.py']
 assert 'offline gate' in json.loads((folder/'status.json').read_text())['error']
 print('PASS: optional phase stage, legacy request deduplication, strict flag validation, failure never bypasses requested check')

with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);jobs.POOL=Queue();tracks={1:{'id':1},2:{'id':2}};request=dict(source=1,target=2)
 a=jobs.create(root,request,tracks);status=root/'mix-jobs'/a['id']/'status.json';legacy=json.loads(status.read_text());del legacy['use_feedback'];jobs.save(status,legacy)
 assert jobs.create(root,request,tracks)['id']==a['id']
 b=jobs.create(root,dict(request,use_feedback=False),tracks);assert a['id']!=b['id']
 for value in [1,0,'false',None]:
  try:jobs.create(root,dict(request,use_feedback=value),tracks)
  except ValueError:pass
  else:raise AssertionError('Non-boolean feedback flag accepted')
 folder=root/'without-feedback';folder.mkdir();jobs.save(folder/'status.json',dict(b,state='queued',owner_pid=os.getpid()));calls=[]
 original_run=jobs.subprocess.run;original_completed=jobs.completed
 def ignore(command,**kwargs):
  script=Path(command[2]).name;calls.append(script)
  if script=='halo_planner.py':assert '--ignore-feedback' in command
  if script=='render_live.py':(folder/'result.json').write_text('{"selection":{}}')
 jobs.subprocess.run=ignore;jobs.completed=lambda s,p:dict(s,state='complete')
 try:jobs.run(folder)
 finally:jobs.subprocess.run=original_run;jobs.completed=original_completed
 assert calls==['halo_planner.py','render_live.py']
 print('PASS: feedback-on/off request isolation, legacy defaults, strict validation and planner opt-out')

with tempfile.TemporaryDirectory() as directory:
 jobs.POOL=Queue();root=Path(directory);tracks={1:{'id':1},2:{'id':2}}
 eight=jobs.create(root,dict(source=1,target=2,bars=8),tracks)
 long=jobs.create(root,dict(source=1,target=2,bars=16),tracks)
 assert eight['id']!=long['id'] and long['bars']==16
 assert jobs.create(root,dict(source=1,target=2,bars=16),tracks)['id']==long['id']
 print('PASS: sixteen-bar preview requests are supported and configuration-isolated')
