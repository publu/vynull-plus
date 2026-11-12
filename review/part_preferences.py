"""Version-bound listening preferences for local section-pair planning."""
import json,sqlite3,hashlib,math,re
from pathlib import Path

PART_REASONS={'wrong sections','clashing vocals or melodies'}
ALIGNMENT_REASONS={'beat alignment problem','unnatural loop','beats clash','not on the same beat','not on beat','off beat','out of sync'}

def feedback_scope(tags,note):
    if tags&ALIGNMENT_REASONS or note in ALIGNMENT_REASONS:return 'alignment_or_loop'
    if any('fade' in tag for tag in tags) or re.search(r'\bfad(?:e|ing)(?:[ -]?out)?\b',note):return 'fade_envelope'
    if 'level jump' in tags:return 'level_matching'
    return 'unspecified'

def load(database,bindings_path,source,target,source_sha,target_sha,bars,loop_bars):
    bindings={r['review_id']:r for r in json.loads(Path(bindings_path).read_text()).get('records',[])} if Path(bindings_path).exists() else {}
    if not Path(database).exists():return [],dict(available=False,reason='No listening history')
    with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True,timeout=5) as c:
        c.row_factory=sqlite3.Row;rows=[dict(r) for r in c.execute('SELECT id,vote,tags,note,preview_snapshot FROM transition_reviews ORDER BY id DESC LIMIT 5000')]
    choices={};notes=[];ignored=[]
    for row in rows:
        try:
            snapshot=json.loads(row['preview_snapshot']);s=snapshot['selection']
            if (s['source_track'],s['target_track'])!=(source,target):continue
            if snapshot.get('engine',{}).get('target_loop'):
                ignored.append(dict(review_id=row['id'],reason='Incoming-loop experiment is a different playback configuration'));continue
            loop=snapshot.get('engine',{}).get('loop');unit=loop.get('bars',0) if loop else 0
            if s['overlap_bars']!=bars or unit!=loop_bars:
                ignored.append(dict(review_id=row['id'],reason='Different overlap or loop configuration'));continue
            kind='stored_source_hashes'
            if snapshot.get('source_audio_sha256')!=source_sha or snapshot.get('target_audio_sha256')!=target_sha:
                proof=bindings.get(row['id'],{})
                digest=hashlib.sha256(row['preview_snapshot'].encode()).hexdigest()
                if not (proof.get('verified') and proof.get('snapshot_sha256')==digest and proof.get('bound_current_source_sha256')==source_sha and proof.get('bound_current_target_sha256')==target_sha):
                    ignored.append(dict(review_id=row['id'],reason='Audio identity is unverified or changed'));continue
                kind='legacy_exact_render_replay'
            a,b=float(s['source_start_seconds']),float(s['target_start_seconds'])
            if not math.isfinite(a) or not math.isfinite(b):raise ValueError('Invalid reviewed cue')
            tags={str(v).strip().casefold() for v in json.loads(row['tags'])};note=row['note'].strip().casefold()
            if row['vote']>0:decision='liked'
            elif tags&PART_REASONS or note in PART_REASONS:decision='avoid_parts'
            else:
                notes.append(dict(review_id=row['id'],scope=feedback_scope(tags,note),
                    source_start_seconds=a,target_start_seconds=b,note=row['note'],tags=sorted(tags)))
                continue
            key=(round(a,6),round(b,6))
            if key in choices:
                choices[key]['same_choice_review_ids'].append(row['id'])
                field='supporting_review_ids' if choices[key]['kind']==decision else 'superseded_review_ids'
                choices[key][field].append(row['id']);continue
            choices[key]=dict(kind=decision,review_id=row['id'],same_choice_review_ids=[row['id']],supporting_review_ids=[row['id']],superseded_review_ids=[],source_start_seconds=a,target_start_seconds=b,
                binding=kind,new_render_approved=False)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError):
            ignored.append(dict(review_id=row['id'],reason='Invalid review snapshot'))
    values=list(choices.values())
    version=hashlib.sha256(json.dumps(dict(choices=values,notes=notes,ignored=ignored),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return values,dict(available=True,version_sha256=version,choices=len(values),notes=notes,ignored=ignored,
        scope='Exact ordered pair, current audio identity, overlap and loop unit; no global taste-model fitting')

def effects(points,bars,preferences):
    """Return exact/snap-liked anchors and bounded explicit part rejections."""
    liked=[];avoided=[];strength=[];evidence=[]
    for point in points:
        positive=[];negative=[]
        for p in preferences:
            distance=float(max(abs(point[0]-p['source_start_seconds'])/bars[0],abs(point[1]-p['target_start_seconds'])/bars[1]))
            if p['kind']=='liked' and distance<=.5:positive.append((distance,p))
            if p['kind']=='avoid_parts' and distance<=4:negative.append((distance,p))
        # A more recent explicit decision wins if neighborhood opinions overlap.
        newest_positive=max((p['review_id'] for _,p in positive),default=-1)
        newest_negative=max((p['review_id'] for d,p in negative if d<=.5),default=-1)
        blocked=bool(negative) and (not positive or newest_negative>newest_positive)
        if positive and not blocked:
            distance,p=min(positive,key=lambda pair:(pair[0],-pair[1]['review_id']))
            liked.append(True);strength.append(1/(1+distance));evidence.append(dict(p,snap_distance_bars=distance))
        else:liked.append(False);strength.append(0.);evidence.append(None)
        avoided.append(blocked)
    return dict(liked=liked,avoided=avoided,strength=strength,evidence=evidence)
