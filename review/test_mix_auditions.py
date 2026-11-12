import tempfile,sqlite3,json,wave,io,hashlib
from pathlib import Path
import mix_auditions as m
class Handler:
 def __init__(self,headers=None):self.headers=headers or {};self.wfile=io.BytesIO();self.head={};self.status=None;self.payload=None
 def send(self,status,data):self.status=status;self.payload=data
 def send_response(self,status):self.status=status
 def send_header(self,k,v):self.head[k]=v
 def end_headers(self):pass
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);audio=root/'test.wav'
 with wave.open(str(audio),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(22050);f.writeframes(b'\0'*4000)
 m.ROOT=root
 item={'audio':{'path':str(audio)},'selection':{'source_track':5,'target_track':10},'engine':{'loop':None},'eq_enabled':True}
 manifest=root/'manifest.json';manifest.write_text(json.dumps({'renders':[item]}));identity=m.previews()[0]['preview_id']
 another=dict(item,selection={'source_track':5,'target_track':11})
 manifest.write_text(json.dumps({'renders':[item,another]}));assert m.previews()[0]['preview_id']==identity
 assert m.previews()[1]['preview_id']!=identity
 changed=dict(item,engine={'loop':{'bars':4}})
 manifest.write_text(json.dumps({'renders':[changed]}));assert m.previews()[0]['preview_id']!=identity
 preview={'preview_id':'test-version-0','manifest_sha256':'test-version','audio':{'path':str(audio)},'selection':{'source_track':5,'target_track':10},'engine':{'overlap_bars':8}}
 m.previews=lambda:[preview];database=root/'feedback.sqlite';m.install_database(database)
 h=Handler({'Range':'bytes=0-99'});assert m.serve(h,'audition-audio',{'id':['test-version-0']},database);assert h.status==206 and len(h.wfile.getvalue())==100 and h.head['Content-Range'].startswith('bytes 0-99/')
 h=Handler({'Range':'bytes=999999-'});m.serve(h,'audition-audio',{'id':['test-version-0']},database);assert h.status==416
 h=Handler();m.serve(h,'audition-vote',{},database,{'preview_id':'test-version-0','vote':1,'tags':['Good musical fit'],'note':'Test fixture only'});assert h.status==200
 with sqlite3.connect(database) as c:
  row=c.execute('SELECT vote,preview_snapshot FROM transition_reviews').fetchone();assert row[0]==1 and json.loads(row[1])==preview
 h=Handler();m.serve(h,'audition-data',{},database);assert h.payload['reviews']['test-version-0']['note']=='Test fixture only'
 changed_id=dict(preview,preview_id='new-manifest-0',manifest_sha256='new-manifest')
 assert m.latest_reviews(database,[changed_id])['new-manifest-0']['vote']==1
 different_plan=dict(changed_id,selection={'source_track':5,'target_track':11})
 assert m.latest_reviews(database,[different_plan])=={}
 h=Handler();m.serve(h,'audition-vote',{},database,{'preview_id':'stale','vote':-1});assert h.status==409
 with sqlite3.connect(database) as c:assert c.execute('SELECT count(*) FROM transition_reviews').fetchone()[0]==1
 digest=hashlib.sha256(audio.read_bytes()).hexdigest()
 bound=dict(preview,audio=dict(preview['audio'],sha256=digest))
 m.previews=lambda:[bound]
 h=Handler();m.serve(h,'audition-vote',{},database,{'preview_id':'test-version-0','vote':-1,'note':'Fade too noticeable'})
 same_audio=dict(bound,preview_id='same-bytes-other-metadata',engine={'overlap_bars':8,'phase_checked':True})
 saved=m.latest_reviews(database,[same_audio]);assert saved[same_audio['preview_id']]['vote']==-1 and saved[same_audio['preview_id']]['match']=='identical_audio'
 unheard=dict(same_audio,preview_id='new-audio',audio=dict(bound['audio'],sha256='f'*64))
 assert not m.latest_reviews(database,[unheard])
 print('PASS: audio ranges, versioned persistence, stale rejection, saved-rating restoration, changed-plan isolation')

with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);m.ROOT=root/'training/section_pair_v1/renders';m.ROOT.mkdir(parents=True)
 lab=root/'training/library_lab_v1';lab.mkdir()
 item=dict(preview_id='verified-audio',selection={'source_track':1,'target_track':2},engine={'strategy':'blend'},audio={'sha256':'a'*64},eq_enabled=True,comparison={'variant':'A'})
 entry=dict(audio_sha256='a'*64,reconstruction_pcm_max_difference=0,preview_snapshot_sha256=hashlib.sha256(m.review_key(item).encode()).hexdigest(),bars=[])
 report=dict(version='exact_overlap_librosa_v1',metric_version='simultaneous_frame_chroma_and_rendered_onsets_v2',all_reconstructions_bit_exact=True,previews={'a'*64:entry})
 path=lab/'overlap-analysis.json';path.write_text(json.dumps(report))
 assert m.overlap_analyses([item])=={'verified-audio':entry}
 assert not m.overlap_analyses([dict(item,engine={'strategy':'echo'})])
 assert not m.overlap_analyses([dict(item,audio={'sha256':'b'*64})])
 report['all_reconstructions_bit_exact']=False;path.write_text(json.dumps(report));assert not m.overlap_analyses([item])
 path.write_text('incomplete json');assert not m.overlap_analyses([item])
 report.update(version='deep_merge_librosa_v2',all_reconstructions_bit_exact=True)
 (lab/'deep-merge-analysis-v2.json').write_text(json.dumps(report))
 assert m.deep_merge_analyses([item])=={'verified-audio':entry}
 assert not m.deep_merge_analyses([dict(item,engine={'strategy':'cut'})])
 discovery=root/'training/discovery_rounds_v1';discovery.mkdir()
 discovery_entry=dict(entry,origin='discovery')
 discovery_report=dict(report,previews={'a'*64:discovery_entry})
 (discovery/'final-deep-analysis.json').write_text(json.dumps(discovery_report))
 assert m.deep_merge_analyses([item])['verified-audio']['origin']=='discovery'
 assert not m.deep_merge_analyses([dict(item,engine={'strategy':'cut'})])
 print('PASS: overlap diagnostics require verified identical audio/configuration; stale or incomplete analyses stay hidden.')

with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);m.ROOT=root/'training/section_pair_v1/renders';m.ROOT.mkdir(parents=True);folder=root/'training/library_search_v2';folder.mkdir()
 p=dict(preview_id='vocal-proof',selection={'source_track':1,'target_track':2},engine={'strategy':'blend'},audio={'sha256':'c'*64},eq_enabled=True,comparison={'variant':'A'})
 entry=dict(reconstruction_pcm_max_difference=0,preview_snapshot_sha256=hashlib.sha256(m.review_key(p).encode()).hexdigest(),status='vocal_overlap_watch')
 (folder/'public-vocal-audit.json').write_text(json.dumps(dict(version='vocal_overlap_v1',all_reconstructions_bit_exact=True,previews={'c'*64:entry})))
 assert m.vocal_analyses([p])['vocal-proof']['status']=='vocal_overlap_watch'
 assert not m.vocal_analyses([dict(p,engine={'strategy':'cut'})])
 print('PASS: vocal warnings stay bound to the actual rendered recipe and audio.')
