import tempfile,sqlite3,json,hashlib
from pathlib import Path
from part_preferences import load,effects

with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);db=root/'reviews.sqlite';binding=root/'bindings.json'
 c=sqlite3.connect(db);c.execute('CREATE TABLE transition_reviews(id INTEGER,vote INTEGER,tags TEXT,note TEXT,preview_snapshot TEXT)')
 def add(i,vote=1,a=10,b=20,tags=(),note='',source_sha='a',target_sha='b',loop=0,eq=True):
  p=dict(selection=dict(source_track=1,target_track=2,source_start_seconds=a,target_start_seconds=b,overlap_bars=8),
   source_audio_sha256=source_sha,target_audio_sha256=target_sha,engine=dict(loop={'bars':loop} if loop else None),eq_enabled=eq)
  text=json.dumps(p);c.execute('INSERT INTO transition_reviews VALUES(?,?,?,?,?)',(i,vote,json.dumps(tags),note,text));c.commit();return text
 add(1);add(2,eq=False)
 prefs,audit=load(db,binding,1,2,'a','b',8,0)
 assert len(prefs)==1 and prefs[0]['same_choice_review_ids']==[2,1]
 e=effects([[10,20],[12,20],[10,24],[30,40]],[2,2],prefs)
 assert e['liked']==[True,False,False,False] and not any(e['avoided'])
 assert not load(db,binding,2,1,'b','a',8,0)[0]
 assert not load(db,binding,1,2,'changed','b',8,0)[0]
 assert not load(db,binding,1,2,'a','b',4,0)[0]
 assert not load(db,binding,1,2,'a','b',8,4)[0]
 add(3,-1,note='beats clash')
 prefs,audit=load(db,binding,1,2,'a','b',8,0)
 assert prefs[0]['review_id']==2 and audit['notes'][0]['scope']=='alignment_or_loop'
 assert not any(effects([[10,20]],[2,2],prefs)['avoided'])
 add(31,-1,note='not on the same beat')
 prefs,audit=load(db,binding,1,2,'a','b',8,0)
 assert audit['notes'][0]['scope']=='alignment_or_loop' and prefs[0]['kind']=='liked'
 add(32,-1,note='almost but the fade out was too noticeable.')
 prefs,audit=load(db,binding,1,2,'a','b',8,0)
 assert audit['notes'][0]['scope']=='fade_envelope' and not any(effects([[10,20]],[2,2],prefs)['avoided'])
 add(4,-1,tags=['Wrong sections'])
 prefs,_=load(db,binding,1,2,'a','b',8,0)
 assert prefs[0]['kind']=='avoid_parts' and effects([[10,20]],[2,2],prefs)['avoided']==[True]
 add(5,1)
 prefs,_=load(db,binding,1,2,'a','b',8,0);assert prefs[0]['kind']=='liked'
 legacy=add(6,source_sha=None,target_sha=None,a=30,b=40)
 assert len(load(db,binding,1,2,'a','b',8,0)[0])==1
 proof=dict(review_id=6,verified=True,snapshot_sha256=hashlib.sha256(legacy.encode()).hexdigest(),bound_current_source_sha256='a',bound_current_target_sha256='b')
 binding.write_text(json.dumps(dict(records=[proof])))
 prefs,_=load(db,binding,1,2,'a','b',8,0)
 assert len(prefs)==2 and prefs[0]['binding']=='legacy_exact_render_replay'
 proof['snapshot_sha256']='wrong';binding.write_text(json.dumps(dict(records=[proof])))
 assert len(load(db,binding,1,2,'a','b',8,0)[0])==1
 narrow=[dict(kind='liked',review_id=1,source_start_seconds=10,target_start_seconds=20),dict(kind='avoid_parts',review_id=2,source_start_seconds=14,target_start_seconds=20)]
 e=effects([[10,20],[14,20]],[2,2],narrow)
 assert e['liked']==[True,False] and e['avoided']==[False,True]
 # Incoming-loop approval cannot become evidence for a straight playback path.
 snapshot=json.loads(add(100,a=70,b=80));snapshot['engine']['target_loop']={'bars':4}
 c.execute('UPDATE transition_reviews SET preview_snapshot=? WHERE id=100',(json.dumps(snapshot),));c.commit()
 prefs,audit=load(db,binding,1,2,'a','b',8,0)
 assert not any(p['source_start_seconds']==70 for p in prefs)
 assert any(r['review_id']==100 and 'Incoming-loop' in r['reason'] for r in audit['ignored'])
 c.close()
print('PASS: version-bound regional preferences, replay binding, configuration and direction isolation, latest opinion, variant deduplication, alignment feedback stays nonpunitive')
