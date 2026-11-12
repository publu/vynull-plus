"""Offline section retrieval using the local analyzer's measured descriptors."""
import math,re

def camelot(key):
 key=str(key or '').strip().replace('♭','b').replace('♯','#')
 if re.fullmatch(r'(?:[1-9]|1[0-2])[ABab]',key):return int(key[:-1]),key[-1].upper()
 minor={'Abm':1,'G#m':1,'Ebm':2,'D#m':2,'Bbm':3,'A#m':3,'Fm':4,'Cm':5,'Gm':6,'Dm':7,'Am':8,'Em':9,'Bm':10,'F#m':11,'Gbm':11,'C#m':12,'Dbm':12}
 major={'B':1,'Cb':1,'F#':2,'Gb':2,'Db':3,'C#':3,'Ab':4,'G#':4,'Eb':5,'D#':5,'Bb':6,'A#':6,'F':7,'C':8,'G':9,'D':10,'A':11,'E':12,'Fb':12}
 return (minor[key],'A') if key in minor else (major[key],'B') if key in major else None

def key_relation(a,b):
 a,b=camelot(a),camelot(b)
 if not a or not b:return 'unknown'
 distance=min(abs(a[0]-b[0]),12-abs(a[0]-b[0]))
 return 'compatible' if a==b or distance==0 or (distance==1 and a[1]==b[1]) else 'different'

def vector(value):
 return isinstance(value,list) and bool(value) and all(isinstance(x,(float,int)) and math.isfinite(x) for x in value)

class SectionIndex:
 def __init__(self,features):
  self.features=features;self.rows=[]
  for tid,record in features.items():
   for section in record.get('sections',[]):
    if vector(section.get('core_vector')):self.rows.append(dict(section,track_id=int(tid)))
  self.core_dim=len(self.rows[0]['core_vector']) if self.rows else 0
  self.rows=[r for r in self.rows if len(r['core_vector'])==self.core_dim]
  full=[r['texture_vector'] for r in self.rows if vector(r.get('texture_vector'))]
  self.texture_dim=len(full[0]) if full else 0
  full=[v for v in full if len(v)==self.texture_dim]
  self.cm,self.cs=self.stats([r['core_vector'] for r in self.rows],self.core_dim)
  self.tm,self.ts=self.stats(full,self.texture_dim)
  weights=[.65 if i<24 else 1.2 if i<30 else .75 for i in range(self.core_dim)]
  self.cw=[math.sqrt(w) for w in weights]
  for r in self.rows:
   r['_core']=self.normalize(r['core_vector'],self.cm,self.cs,self.cw)
   r['_texture']=self.normalize(r['texture_vector'],self.tm,self.ts) if vector(r.get('texture_vector')) and len(r['texture_vector'])==self.texture_dim else None
  self.by_id={(r['track_id'],r['index']):r for r in self.rows}
 @staticmethod
 def stats(rows,n):
  if not rows:return [],[]
  means=[math.fsum(r[i] for r in rows)/len(rows) for i in range(n)]
  return means,[max(.05,math.sqrt(math.fsum((r[i]-means[i])**2 for r in rows)/len(rows))) for i in range(n)]
 @staticmethod
 def normalize(v,mean,scale,weights=None):
  return [max(-5,min(5,(x-m)/s))*(weights[i] if weights else 1) for i,(x,m,s) in enumerate(zip(v,mean,scale))]
 @staticmethod
 def distance(a,b):return math.fsum((x-y)**2 for x,y in zip(a,b))/len(a)
 def search(self,source,section_index,tracks,mode='transition',limit=8):
  if mode not in ('sound','transition'):raise ValueError('Unknown matching mode')
  query=self.by_id.get((source,section_index))
  if not query:raise ValueError('This part has no measured fingerprint')
  byid={t['id']:t for t in tracks};ref=byid.get(source,{})
  bpm=ref.get('bpm') or self.features[str(source)]['features'].get('tempo') or query.get('tempo_bpm') or 0
  key=query.get('key') or ref.get('key') or ''
  detailed_tracks={r['track_id'] for r in self.rows if r['track_id']!=source and r['_texture'] is not None and r['track_id'] in byid}
  detailed=bool(query['_texture'] is not None and len(detailed_tracks)>=3)
  ranked=[]
  for r in self.rows:
   tid=r['track_id'];t=byid.get(tid)
   if tid==source or not t or (detailed and r['_texture'] is None):continue
   tbpm=t.get('bpm') or self.features[str(tid)]['features'].get('tempo') or r.get('tempo_bpm') or 0
   factor=min((.5,1,2),key=lambda f:abs(tbpm*f-bpm)) if bpm and tbpm else 1
   delta=tbpm*factor-bpm if bpm and tbpm else None
   relationship=key_relation(key,r.get('key') or t.get('key'))
   if mode=='transition' and (delta is None or abs(delta)>8 or relationship!='compatible'):continue
   distance=self.distance(query['_core'],r['_core'])
   if detailed:distance=.55*distance+.45*self.distance(query['_texture'],r['_texture'])
   level=r.get('mean_rms_dbfs',0)-query.get('mean_rms_dbfs',0)
   if mode=='transition':distance+=.15*(delta/4)**2+.1*abs(level)/6
   ranked.append((distance,tid,r,delta,level,relationship,factor))
  matches=[];seen=set()
  for distance,tid,r,delta,level,relationship,factor in sorted(ranked,key=lambda x:(x[0],x[1],x[2]['index'])):
   if tid in seen:continue
   seen.add(tid);t=byid[tid]
   matches.append({'track':{k:t.get(k) for k in ('id','title','artist','bpm','key','genre')},
    'section':{k:v for k,v in r.items() if not k.startswith('_') and k not in ('core_vector','texture_vector','onset_phase')},
    'distance':round(math.sqrt(max(0,distance)),4),'tempo_difference_bpm':round(delta,2) if delta is not None else None,
    'tempo_interpretation_factor':factor,'energy_difference_db':round(level,2),'key_relationship':relationship})
   if len(matches)>=limit:break
  return {'source':source,'section_index':section_index,'query':{k:v for k,v in query.items() if not k.startswith('_') and k not in ('core_vector','texture_vector','onset_phase')},
   'mode':mode,'indexed_sections':len(self.rows),'feature_space':'MFCC / spectral contrast / groove + chroma / energy / brightness' if detailed else 'Chroma / energy / brightness',
   'matches':matches,'scope':'Measured section similarity; audition candidates to judge musical fit. This does not send deck commands.'}
