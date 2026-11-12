"""Append phrase-bridge trials and exact-bound diagnostics without changing history."""
from pathlib import Path
import json,time,copy,hashlib,fcntl,uuid
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent

def binding(p):return hashlib.sha256(json.dumps({k:p.get(k) for k in ['selection','engine','audio','eq_enabled','comparison']},sort_keys=True).encode()).hexdigest()
def write(path,data):
 temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp.json');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)
def append_report(report,entry,p):
 key=p['audio']['sha256'];e=copy.deepcopy(entry);e.update(analysis_input_snapshot_sha256=e['preview_snapshot_sha256'],preview_snapshot_sha256=binding(p));existing=report['previews'].get(key)
 if existing is None:report['previews'][key]=e
 else:
  candidates=existing if isinstance(existing,list) else [existing]
  if not any(v['preview_snapshot_sha256']==e['preview_snapshot_sha256'] for v in candidates):candidates.append(e)
  report['previews'][key]=candidates

def publish(entries):
 root=TRAIN/'section_pair_v1/renders';result=[]
 with (HERE/'publish.lock').open('a') as outer:
  fcntl.flock(outer,fcntl.LOCK_EX)
  with (root/'manifest.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);path=root/'manifest.json';manifest=json.loads(path.read_text());reports={}
   for field,filename,version in [('overlap','overlap.json','exact_overlap_librosa_v1'),('deep','deep.json','deep_merge_librosa_v2'),('vocal','vocals.json','vocal_overlap_v1'),('structure','structure.json','phrase_bridge_structure_v1')]:
    dest=HERE/filename;data=json.loads(dest.read_text()) if dest.exists() else dict(version=version,all_reconstructions_bit_exact=True,previews={})
    if field=='overlap':data['metric_version']='simultaneous_frame_chroma_and_rendered_onsets_v2'
    reports[field]=(dest,data)
   for row in entries:
    p=copy.deepcopy(row['preview']);existing=next((v for v in manifest['renders'] if v.get('job_id')==p['job_id']),None)
    if existing:result.append(existing);continue
    group=p['comparison']['group_id'];used={v['comparison']['variant'] for v in manifest['renders'] if v.get('comparison',{}).get('group_id')==group}
    if p['comparison']['variant'] in used:
     p['comparison']['variant']=next((chr(65+i) for i in range(26) if chr(65+i) not in used),'J'+p['job_id'][:8])
    for field,(dest,report) in reports.items():append_report(report,row[field],p)
    manifest['renders'].append(p);result.append(p)
   for dest,data in reports.values():data.update(updated=time.time());write(dest,data)
   write(path,manifest)
 return result

def main():
 entries=[]
 for pair in ['118-144','71-11']:entries.extend(json.loads((HERE/(pair+'-analysis.json')).read_text())['entries'])
 result=publish(entries);write(HERE/'published.json',dict(created=time.time(),previews=result,clips=len(result),original_audio_unchanged=True,deck_commands_sent=False,listening_quality_approved=False));print('PHRASE_BRIDGES_PUBLISHED',len(result),flush=True)
if __name__=='__main__':main()
