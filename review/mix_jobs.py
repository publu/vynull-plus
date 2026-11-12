"""Bounded local preview jobs. Public inputs are library IDs, never paths/commands."""
import concurrent.futures,json,os,re,subprocess,threading,time,uuid
from pathlib import Path
import mix_auditions

TRAINING=Path(os.environ.get('VYNULL_LAB_DIR',Path(__file__).resolve().parent.parent/'lab'))

def lab_python():
 return os.environ.get('LAB_PYTHON',str(TRAINING/'.venv/bin/python'))
POOL=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='mix-preview')
LOCK=threading.Lock()

def save(path,data):
 temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(data,allow_nan=False));temp.replace(path)

def public(state):
 return {k:v for k,v in state.items() if k in {'id','state','created','updated','source','target','method','bars','loop_bars','phase_align','phase_alignment','use_feedback','listening_preference','phase','error','preview_url','audio_url','selection','style','entry_loop_bars','bridge'}}

def completed(state,preview):
 item=next((p for p in mix_auditions.previews() if p.get('job_id')==state['id']),None)
 if not item:raise ValueError('Rendered preview was not published')
 state.update(state='complete',phase='Ready to audition',updated=time.time(),preview_url='/api/review/auditions#'+item['preview_id'],
  audio_url='/api/review/audition-audio?id='+item['preview_id'],
  selection={k:v for k,v in preview['selection'].items() if k in {'source_title','target_title','source_start_seconds','target_start_seconds','mix_bpm','overlap_bars'}})
 alignment=preview['selection'].get('bar_phase_alignment')
 if alignment:
  state['phase_alignment']={side:{k:v for k,v in alignment[side].items() if k in {'eligible','applied','shift_beats','reason'}} for side in ['source','target']}
 if preview['selection'].get('listening_preference'):state['listening_preference']=preview['selection']['listening_preference']
 if preview.get('engine',{}).get('recipe')=='phrase_bridge_v1':
  state['bridge']=preview['engine']['structure'];state['phase']='Ready: instrumental handoff, then the next vocal'
  c=item.get('comparison',{});state['preview_url']='/api/review/auditions?collection='+c['collection_id']+'&comparison='+c['group_id']+'#'+item['preview_id']
 return state

def read(path):
 state=json.loads(path.read_text());owner=state.get('owner_pid')
 if state['state'] in ('queued','running') and owner!=os.getpid():
  try:os.kill(owner,0)
  except (ProcessLookupError,TypeError):
   result=path.parent/'result.json'
   if result.exists():state=completed(state,json.loads(result.read_text()))
   else:state.update(state='failed',error='Preview generation was interrupted; create another preview',updated=time.time())
   save(path,state)
 return state

def run(folder):
 path=folder/'status.json';state=read(path)
 try:
  if state.get('style','blend')=='bridge':
   state.update(state='running',phase='Finding an instrumental loop and a vocal phrase exit',updated=time.time());save(path,state)
   command=[lab_python(),'-u',str(TRAINING/'phrase_bridges_v1/make_job.py'),
    '--source',str(state['source']),'--target',str(state['target']),'--bars',str(state['bars']),
    '--entry-loop',str(state.get('entry_loop_bars',0)),'--job-id',state['id'],'--output',str(folder/'result.json')]
   with (folder/'process.log').open('ab',buffering=0) as log:subprocess.run(command,stdout=log,stderr=log,timeout=900,check=True)
   state=completed(state,json.loads((folder/'result.json').read_text()));save(path,state);return
  state.update(state='running',phase='Choosing song parts',updated=time.time());save(path,state)
  planner=[lab_python(),'-u',str(TRAINING/'semantic_pair_v1/halo_planner.py'),
   '--source',str(state['source']),'--target',str(state['target']),'--method',state['method'],
   '--bars',str(state['bars']),'--output',str(folder/'plan.json')]
  if state['loop_bars']:planner+=['--loop-bars',str(state['loop_bars'])]
  if not state.get('use_feedback',True):planner+=['--ignore-feedback']
  refreshed_music=False;prepared_loops=False
  for attempt in range(3):
   offset=(folder/'process.log').stat().st_size if (folder/'process.log').exists() else 0
   try:
    with (folder/'process.log').open('ab',buffering=0) as log:subprocess.run(planner,stdout=log,stderr=log,timeout=180,check=True)
    break
   except subprocess.CalledProcessError:
    text=(folder/'process.log').read_bytes()[offset:].decode(errors='replace').lower()
    if 'music features' in text and not refreshed_music:
     state.update(phase='Analyzing song music',updated=time.time());save(path,state)
     command=[lab_python(),'-u',str(TRAINING/'semantic_pair_v1/extract_halo.py'),'--ids',str(state['source'])+','+str(state['target'])]
     with (folder/'process.log').open('ab',buffering=0) as log:subprocess.run(command,stdout=log,stderr=log,timeout=360,check=True)
     refreshed_music=True
    elif 'loop analysis needs refresh' in text and state['loop_bars'] and not prepared_loops:
     state.update(phase='Finding repeatable song regions',updated=time.time());save(path,state)
     command=[lab_python(),'-u',str(TRAINING/'part_alignment_v1/prepare_loops.py'),'--source',str(state['source']),'--bars',str(state['loop_bars'])]
     with (folder/'process.log').open('ab',buffering=0) as log:subprocess.run(command,stdout=log,stderr=log,timeout=180,check=True)
     prepared_loops=True
    else:raise
    state.update(phase='Choosing song parts',updated=time.time());save(path,state)
  else:raise ValueError('The song analysis changed repeatedly; try again when it settles')
  if state.get('phase_align',False):
   state.update(phase='Checking bar alignment',updated=time.time());save(path,state)
   (folder/'plan-unadjusted.json').write_bytes((folder/'plan.json').read_bytes())
   command=[lab_python(),'-u',str(TRAINING/'bar_phase_v1/adjust_plan.py'),
    '--plan',str(folder/'plan-unadjusted.json'),'--output',str(folder/'plan.json')]
   with (folder/'process.log').open('ab',buffering=0) as log:subprocess.run(command,stdout=log,stderr=log,timeout=180,check=True)
  state.update(phase='Aligning beats and rendering',updated=time.time());save(path,state)
  renderer=[lab_python(),'-u',str(TRAINING/'part_alignment_v1/render_live.py'),
   '--plan',str(folder/'plan.json'),'--job-id',state['id'],'--output',str(folder/'result.json')]
  if state['loop_bars']:renderer+=['--loop-bars',str(state['loop_bars'])]
  with (folder/'process.log').open('ab',buffering=0) as log:subprocess.run(renderer,stdout=log,stderr=log,timeout=240,check=True)
  state=completed(state,json.loads((folder/'result.json').read_text()));save(path,state)
 except Exception:
  error='Could not find a clean instrumental loop and phrase exit. Try another incoming track.' if state.get('style')=='bridge' else 'Could not render this preview. Try another pair or a blend instead of a loop.'
  log=folder/'process.log'
  if log.exists():
   messages=[line[len('ValueError: '):] for line in log.read_text(errors='replace').splitlines() if line.startswith('ValueError: ')]
   if messages:error=messages[-1][:300]
  state.update(state='failed',phase='Not rendered',error=error,updated=time.time());save(path,state)

def create(root,body,tracks):
 if not isinstance(body,dict):raise ValueError('Invalid preview request')
 source,target=body.get('source'),body.get('target');method=body.get('method','semantic');bars=body.get('bars',8);loop=body.get('loop_bars',0)
 phase_align=body.get('phase_align',False)
 style=body.get('style','blend');entry_loop=body.get('entry_loop_bars',0)
 use_feedback=body.get('use_feedback',style!='bridge')
 if type(source)!=int or type(target)!=int or source==target or source not in tracks or target not in tracks:raise ValueError('Choose two different library songs')
 if method not in ('semantic','measured') or type(bars)!=int or bars not in (4,8,16) or type(loop)!=int or loop not in (0,4,8) or type(phase_align)!=bool or type(use_feedback)!=bool:raise ValueError('Invalid preview configuration')
 if style not in ('blend','bridge') or type(entry_loop)!=int or entry_loop not in (0,2,4):raise ValueError('Invalid transition style or incoming loop length')
 if style=='blend' and entry_loop:raise ValueError('Incoming entry loops require the loop bridge style')
 if style=='bridge' and (bars not in (8,16) or loop or phase_align or use_feedback):raise ValueError('Loop bridges use eight/sixteen-bar handoffs and their own vocal/phrase checks; use straight blend for earlier controls')
 directory=Path(root)/'mix-jobs';directory.mkdir(exist_ok=True)
 request={'source':source,'target':target,'method':method,'bars':bars,'loop_bars':loop,'phase_align':phase_align,'use_feedback':use_feedback,'style':style,'entry_loop_bars':entry_loop}
 with LOCK:
  pending=[]
  for path in directory.glob('*/status.json'):
   state=read(path)
   if state['state'] in ('queued','running'):pending.append(state)
  for state in pending:
   if all(state.get(k,{'phase_align':False,'use_feedback':True,'style':'blend','entry_loop_bars':0}.get(k))==v for k,v in request.items()):return state
  if len(pending)>=4:raise ValueError('Four previews are already queued; let one finish first')
  ident=uuid.uuid4().hex;folder=directory/ident;folder.mkdir()
  state=dict(request,id=ident,state='queued',phase='Queued',created=time.time(),updated=time.time(),owner_pid=os.getpid())
  save(folder/'status.json',state);POOL.submit(run,folder)
 return state

def serve(handler,route,query,root,body=None,tracks=None):
 if route!='mix-jobs':return False
 try:
  if body is not None:handler.send(202,public(create(root,body,tracks)));return True
  ident=query.get('id',[''])[0]
  if ident:
   if not re.fullmatch('[0-9a-f]{32}',ident):raise ValueError('Invalid preview job')
   path=Path(root)/'mix-jobs'/ident/'status.json'
   if not path.exists():handler.send(404,{'error':'Preview job not found'});return True
   handler.send(200,public(read(path)));return True
  rows=[public(read(p)) for p in (Path(root)/'mix-jobs').glob('*/status.json')]
  handler.send(200,{'jobs':sorted(rows,key=lambda r:-r['created'])[:20]});return True
 except ValueError as e:handler.send(400,{'error':str(e)});return True
