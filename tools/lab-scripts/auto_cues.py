#!/usr/bin/env python3
"""Create starter hot cues from the existing song-section analysis, preserving edits."""
import argparse,json,os,time,urllib.request
from pathlib import Path

NAMES={1:'INTRO',2:'UP',3:'DOWN',4:'BRIDGE',5:'CHORUS',6:'OUTRO'}
COLORS={1:10,2:33,3:2,4:54,5:37,6:62}

def plan_cues(phrases,beats,existing,duration_ms):
    free=[n for n in range(1,9) if not any(c['number']==n for c in existing)]
    downbeats=[float(b['time_ms']) for b in beats if b.get('beat_in_bar')==1]
    candidates=[]
    for phrase in sorted(phrases,key=lambda p:p['start_ms']):
        start=float(phrase['start_ms']);end=float(phrase.get('end_ms',duration_ms))
        if start<0 or start>=duration_ms or end-start<2000:continue
        position=start
        if downbeats:
            nearest=min(downbeats,key=lambda t:abs(t-start))
            if abs(nearest-start)<=1000:position=nearest
        position=round(position)
        if position<0 or position>=duration_ms:continue
        if any(abs(c['time_ms']-position)<500 for c in existing+candidates):continue
        kind=int(phrase.get('kind',0))
        candidates.append({'number':0,'type':'cue','time_ms':position,'loop_ms':-1,'color_id':COLORS.get(kind,22),'label':phrase.get('name') or NAMES.get(kind,'SECTION')})
    # Retain the first instance of each section (including OUTRO), then repeats.
    primary=[];secondary=[];seen=set()
    for cue in candidates:
        if cue['label'] in seen:secondary.append(cue)
        else:primary.append(cue);seen.add(cue['label'])
    selected=sorted((primary+secondary)[:len(free)],key=lambda c:c['time_ms'])
    return [dict(c,number=n) for n,c in zip(free,selected)]

def run(base,data_dir,track_ids=None):
    data=Path(data_dir);data.mkdir(parents=True,exist_ok=True)
    def api(route,payload=None):
        req=urllib.request.Request(base+'/'+route,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
    manifest_path=data/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tracks=api('tracks')
    if track_ids is not None:tracks=[t for t in tracks if t['id'] in track_ids]
    pending=[t for t in tracks if str(t['id']) not in manifest]
    if not pending:
        print('All library tracks already processed.',flush=True);return
    backup=data/('cues-before-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
    backup.write_text(json.dumps(api('cues'),indent=2))
    totals={'tracks':0,'cues':0,'errors':0}
    for track in pending:
        track_id=track['id'];key=str(track_id)
        try:
            analysis=api(f'analysis/{track_id}')
            phrases=api(f'tracks/{track_id}/phrases')
            if not phrases:
                print(f'No sections yet: {track_id}',flush=True);continue
            existing=api(f'tracks/{track_id}/cues')
            planned=plan_cues(phrases,analysis.get('beats',[]),existing,float(analysis['duration_sec'])*1000)
            written=[]
            for cue in planned:
                # Re-read before writing: never reuse a slot the user has just filled.
                current=api(f'tracks/{track_id}/cues')
                if any(c['number']==cue['number'] or abs(c['time_ms']-cue['time_ms'])<500 for c in current):continue
                api(f'tracks/{track_id}/cues',cue);written.append(cue)
            current=api(f'tracks/{track_id}/cues')
            if not all(any(c==saved for c in current) for saved in existing):
                raise RuntimeError('An existing cue changed during processing; original is backed up')
            if not all(any(c==saved for c in current) for saved in written):
                raise RuntimeError('Cue readback did not match saved points')
            manifest[key]={'title':track['title'],'created_at':time.time(),'generated':written,'preserved':existing,'backup':str(backup)}
            temporary=manifest_path.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(manifest_path)
            totals['tracks']+=1;totals['cues']+=len(written)
            print(f'{track_id}: {track["title"]}: added {len(written)}, preserved {len(existing)}',flush=True)
        except Exception as exc:
            totals['errors']+=1;print(f'ERROR {track_id}: {exc}',flush=True)
    print(json.dumps(totals),flush=True)
    if totals['errors']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default=os.environ.get('VYNULL_API','http://127.0.0.1:9443/api/').rstrip('/'))
    parser.add_argument('--data-dir',default=os.environ.get('AUTOCUES_DIR','autocues'))
    parser.add_argument('--track',type=int,action='append')
    args=parser.parse_args();run(args.base,args.data_dir,args.track)
