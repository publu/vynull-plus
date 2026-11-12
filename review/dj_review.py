#!/usr/bin/env python3
"""Contextual DJ suggestions, request queue, and persistent taste feedback."""
import json,math,sqlite3,time,urllib.request,urllib.parse,threading,re,os
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(os.environ.get('DJ_REVIEW_DIR',Path(__file__).resolve().parent))
ROOT.mkdir(parents=True,exist_ok=True)
DB=ROOT/'review.sqlite'
import mix_auditions
import mix_jobs
mix_auditions.install_database(DB)
LAB=Path(os.environ.get('VYNULL_LAB_DIR',Path(__file__).resolve().parent.parent/'lab'))
PLAN_PATH=LAB/'raveform_v1/artifacts/halo_advisory.json'  # optional per-library plan
FEATURES=json.loads((ROOT/'measured-audio.json').read_text()) if (ROOT/'measured-audio.json').exists() else {}
from section_matches import SectionIndex
PART_INDEX=SectionIndex(FEATURES)

def db():
 c=sqlite3.connect(DB,timeout=15);c.row_factory=sqlite3.Row;return c
with db() as c:
 c.executescript('''CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, source INTEGER, target INTEGER, kind TEXT, vote INTEGER, start REAL, end REAL, created REAL);
 CREATE TABLE IF NOT EXISTS queue(id INTEGER PRIMARY KEY, text TEXT, track_id INTEGER, status TEXT DEFAULT 'pending', created REAL);''')
CACHE={};LOCK=threading.Lock()
def library():
 with LOCK:
  if time.time()-CACHE.get('at',0)>10:
   with urllib.request.urlopen(os.environ.get('VYNULL_API','http://127.0.0.1:9443/api/')+'tracks',timeout=15) as r:CACHE['tracks']=json.load(r)
   CACHE['at']=time.time()
  return CACHE['tracks']

def numeric(value):
 return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def sound_similarity(a,b):
 fa,fb=FEATURES.get(str(a)),FEATURES.get(str(b))
 if not fa or not fb:return None
 dims=[('brightness',2500),('dynamic_range_db',15),('hook_density_per_min',15),('flatness',.15)]
 distances=[min(2,abs(fa['features'][k]-fb['features'][k])/scale) for k,scale in dims if numeric(fa['features'].get(k)) and numeric(fb['features'].get(k))]
 return max(0,1-sum(distances)/len(distances)) if distances else None

def harmonic(a,b):
 a,b=str(a).upper(),str(b).upper()
 if not re.fullmatch(r'(?:[1-9]|1[0-2])[AB]',a) or not re.fullmatch(r'(?:[1-9]|1[0-2])[AB]',b):return None
 x,y=int(a[:-1]),int(b[:-1]);d=min(abs(x-y),12-abs(x-y))
 return 1 if a==b else .85 if (d==1 and a[-1]==b[-1]) or d==0 else .1

def listening_memory(rows):
 """Keep exact-plan reviews; positive examples support a pair, timing failures don't veto it."""
 seen=set();pairs={}
 for row in rows:
  try:
   preview=json.loads(row['preview_snapshot']);key=mix_auditions.review_key(preview)
   if key in seen:continue
   seen.add(key);s=preview['selection'];pair=(s['source_track'],s['target_track'])
   item={'review_id':row['id'],'vote':row['vote'],'note':row['note'],'tags':json.loads(row['tags']),
    'source_start_seconds':s['source_start_seconds'],'target_start_seconds':s['target_start_seconds'],
    'source_section':s['source_section'],'target_section':s['target_section'],'variant':preview.get('variant'),
    'preview_id':row['preview_id']}
   pairs.setdefault(pair,[]).append(item)
  except (KeyError,ValueError,TypeError):continue
 return pairs

def recommendations(source,tracks,feedback,query='',listening=None):
 byid={t['id']:t for t in tracks};ref=byid.get(source);out=[]
 votes={r['target']:r['vote'] for r in feedback if r['source']==source and r['kind']=='match'}
 terms=re.findall(r'\w+',query.casefold())
 if not ref and not terms:return []
 for t in tracks:
  if t['id']==source:continue
  reasons=[];score=0;weight=0
  if ref:
   bpm1,bpm2=ref.get('bpm',0),t.get('bpm',0)
   if bpm1 and bpm2:
    delta=min(abs(bpm2*m-bpm1)/bpm1 for m in (.5,1,2))
    score+=.35*max(0,1-delta/.15);weight+=.35
    if delta<.06:reasons.append('Compatible tempo'+(' (half/double)' if abs(bpm2-bpm1)/bpm1>.1 else ''))
   h=harmonic(ref.get('key'),t.get('key'))
   if h is not None:
    score+=.15*h;weight+=.15
    if h>=.8:reasons.append('Compatible key')
   sim=sound_similarity(source,t['id'])
   if sim is not None:
    score+=.4*sim;weight+=.4
    if sim>.65:reasons.append('Similar measured brightness, dynamics and event density')
   genre=ref.get('genre','').strip()
   if genre and genre.casefold()==t.get('genre','').strip().casefold():score+=.1;weight+=.1;reasons.append('Same genre tag')
  score=score/weight if weight else 0
  if terms:
   text=(t.get('title','')+' '+t.get('artist','')).casefold();hits=sum(term in text for term in terms)
   if not hits:continue
   score=.6*hits/len(terms)+.4*score;reasons.insert(0,'Request text match')
  vote=votes.get(t['id'],0)
  if vote:score+=.35*vote;reasons.insert(0,'You marked this pair '+('a match' if vote>0 else 'not a match'))
  evidence=(listening or {}).get((source,t['id']),[])
  if any(r['vote']>0 for r in evidence):
   score+=.1;reasons.append('You approved a transition using this pair')
  rejected=[r for r in evidence if r['vote']<0]
  if rejected:
   note=next((r['note'] for r in rejected if r['note']),None)
   reasons.append('Earlier transition review: '+note if note else 'An earlier transition was rejected; try different parts or timing')
  audio=FEATURES.get(str(t['id']),{});f=audio.get('features',{})
  out.append({'track':{k:t.get(k) for k in ('id','title','artist','bpm','key','genre','file_type')},'score':round(score,3),'reasons':reasons,'feedback':vote,'listening_evidence':evidence,'measured_audio':bool(audio),'sound':{'brightness_hz':f.get('brightness'),'dynamic_range_db':f.get('dynamic_range_db'),'event_density_per_min':f.get('hook_density_per_min')},'format':{'codec':audio.get('codec'),'bitrate_kbps':audio.get('bitrate_kbps')}})
 return sorted(out,key=lambda r:(-r['score'],r['track']['id']))[:8]

def section_suggestions(track_id,feedback):
 sections=FEATURES.get(str(track_id),{}).get('sections',[])
 votes={(r['target'],round(r['start'],2)):r['vote'] for r in feedback if r['kind']=='part'}
 out=[]
 for index,s in enumerate(sections):
  start,end=s.get('start_seconds'),s.get('end_seconds')
  if not numeric(start) or not numeric(end) or end-start<8:continue
  vote=votes.get((track_id,round(start,2)),0)
  lift=s.get('energy_change_db') or 0
  score=min(abs(lift),18)/18+vote*2
  out.append({'index':s.get('index',index),'start':start,'end':end,'label':s.get('label') or 'Measured section','key':s.get('key'),'bpm':s.get('tempo_bpm'),'energy_change_db':lift,'feedback':vote,'score':score})
 return sorted(out,key=lambda s:(-s['score'],s['start']))[:4]

class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,status,payload):
  body=json.dumps(payload,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
 def do_GET(self):
  try:
   u=urllib.parse.urlsplit(self.path);q=urllib.parse.parse_qs(u.query);route=u.path.removeprefix('/api/review/')
   if mix_auditions.serve(self,route,q,DB):return
   if mix_jobs.serve(self,route,q,ROOT):return
   with db() as c:
    if route=='health':return self.send(200,{'ok':True,'measured_tracks':len(FEATURES)})
    if route=='parts':
     try:
      source=int(q.get('source',['0'])[0]);section=int(q.get('section',['0'])[0]);mode=q.get('mode',['transition'])[0]
      return self.send(200,PART_INDEX.search(source,section,library(),mode))
     except ValueError as e:return self.send(400,{'error':str(e)})
    if route=='context':
     source=int(q.get('track',['0'])[0]);feedback=c.execute('SELECT * FROM feedback ORDER BY id').fetchall();queue=[dict(r) for r in c.execute("SELECT * FROM queue WHERE status='pending' ORDER BY id")]
     audition_reviews=c.execute('SELECT * FROM transition_reviews ORDER BY id DESC').fetchall();memory=listening_memory(audition_reviews)
     tracks=library();byid={t['id']:t for t in tracks}
     trained=None
     if PLAN_PATH.exists():
      plans=json.loads(PLAN_PATH.read_text())
      trained=next((p for p in plans.get('plans',[]) if p['track_id']==source),None)
     for item in queue:
      item['track']=byid.get(item['track_id']);item['candidates']=recommendations(source,tracks,feedback,item['text'],listening=memory)[:3] if not item['track_id'] else []
     return self.send(200,{'source':source,'similar':recommendations(source,tracks,feedback,listening=memory),'sections':section_suggestions(source,feedback),'queue':queue,'feedback_count':len(feedback),'listening_feedback_count':len(audition_reviews),'measured_tracks':len(FEATURES),'scope':'halo_library','experimental_transition_plan':trained})
   self.send(404,{'error':'Not found'})
  except Exception as e:self.send(500,{'error':str(e)})
 def do_POST(self):
  try:
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<=4096:return self.send(400,{'error':'Invalid request size'})
   body=json.loads(self.rfile.read(length));route=urllib.parse.urlsplit(self.path).path.removeprefix('/api/review/')
   if mix_auditions.serve(self,route,{},DB,body):return
   tracks={t['id']:t for t in library()}
   if mix_jobs.serve(self,route,{},ROOT,body=body,tracks=tracks):return
   with db() as c:
    if route=='feedback':
     source,target=int(body.get('source',0)),int(body['target']);kind=body.get('kind','match');vote=body['vote'];start,end=body.get('start',0),body.get('end',0)
     if target not in tracks or source not in tracks or kind not in ('match','part') or type(vote)!=int or vote not in (-1,1):return self.send(400,{'error':'Invalid feedback'})
     if not numeric(start) or not numeric(end) or start<0 or end<start:return self.send(400,{'error':'Invalid section'})
     c.execute('INSERT INTO feedback(source,target,kind,vote,start,end,created) VALUES(?,?,?,?,?,?,?)',(source,target,kind,vote,start,end,time.time()))
    elif route=='queue':
     text=str(body.get('text','')).strip();tid=body.get('track_id')
     if not text or len(text)>300 or (tid is not None and (type(tid)!=int or tid not in tracks)):return self.send(400,{'error':'Enter a song or artist, up to 300 characters'})
     if c.execute("SELECT count(*) FROM queue WHERE status='pending'").fetchone()[0]>=200:return self.send(400,{'error':'Queue is full'})
     c.execute('INSERT INTO queue(text,track_id,created) VALUES(?,?,?)',(text,tid,time.time()))
    elif route=='queue/status':
     status=body.get('status');ident=int(body['id'])
     if status not in ('done','pending','dismissed'):return self.send(400,{'error':'Invalid status'})
     c.execute('UPDATE queue SET status=? WHERE id=?',(status,ident))
    else:return self.send(404,{'error':'Not found'})
   self.send(200,{'ok':True})
  except (KeyError,ValueError,TypeError,json.JSONDecodeError) as e:self.send(400,{'error':'Invalid request'})
  except Exception as e:self.send(500,{'error':str(e)})

if __name__=='__main__':ThreadingHTTPServer((os.environ.get('DJ_REVIEW_HOST','127.0.0.1'),int(os.environ.get('DJ_REVIEW_PORT','9444'))),Handler).serve_forever()
