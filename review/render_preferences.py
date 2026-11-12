"""Replay only explicit, version-bound approvals of controlled render settings."""
from pathlib import Path
import sqlite3,json,math

def select(database,selection,source_sha,target_sha,loop_bars,enabled=True):
    if not enabled:return dict(enabled=False,applied=False,reason='Listening feedback disabled')
    if not Path(database).exists():return dict(enabled=True,applied=False,reason='No listening history')
    with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True,timeout=5) as c:
        c.row_factory=sqlite3.Row;rows=[dict(r) for r in c.execute('SELECT id,vote,preview_snapshot FROM transition_reviews ORDER BY id DESC LIMIT 5000')]
    seen=set();ignored=[]
    for row in rows:
        try:
            p=json.loads(row['preview_snapshot']);s=p['selection'];engine=p['engine'];comparison=p.get('comparison',{})
            if comparison.get('kind') not in ['fade_exit','beat_alignment'] and not s.get('render_preference',{}).get('applied'):continue
            if p.get('source_audio_sha256')!=source_sha or p.get('target_audio_sha256')!=target_sha:continue
            if (s['source_track'],s['target_track'])!=(selection['source_track'],selection['target_track']):continue
            if s['overlap_bars']!=selection['overlap_bars'] or (engine.get('loop') or {}).get('bars',0)!=loop_bars:continue
            if not p.get('eq_enabled') or abs(float(s['mix_bpm'])-float(selection['mix_bpm']))>1e-6:continue
            if abs(float(s.get('target_tempo_factor',1))-float(selection.get('target_tempo_factor',1)))>1e-6:continue
            fade=engine.get('fade_envelope',{});handoff=fade.get('handoff_bars',s['overlap_bars'])
            nudge=engine.get('target_alignment',{}).get('cue_nudge',{});frames=nudge.get('shift_frames',0)
            if type(handoff)!=int or handoff not in [1,2,4,8] or handoff>s['overlap_bars']:continue
            if type(frames)!=int or abs(frames)>round(2*60/s['mix_bpm']*48000):continue
            incoming_gain=engine.get('incoming_level_gain');output_gain=p.get('audio',{}).get('gain_applied')
            if incoming_gain is not None and (not isinstance(incoming_gain,(float,int)) or not math.isfinite(incoming_gain) or not .5<=incoming_gain<=2):continue
            if output_gain is not None and (not isinstance(output_gain,(float,int)) or not math.isfinite(output_gain) or not 0<output_gain<=1):continue
            delta=frames/48000;base_target=s.get('base_target_start_seconds',float(s['target_start_seconds'])-delta)
            values=[float(s['source_start_seconds']),float(base_target)]
            if not all(math.isfinite(v) for v in values):continue
            if abs(values[0]-selection['source_start_seconds'])>.02 or abs(values[1]-selection['target_start_seconds'])>.02:continue
            key=(handoff,frames,round(incoming_gain,12) if incoming_gain is not None else None,round(output_gain,12) if output_gain is not None else None)
            if key in seen:continue
            seen.add(key)
            if row['vote']<=0:
                ignored.append(dict(review_id=row['id'],reason='This exact render configuration was rejected'));continue
            return dict(enabled=True,applied=True,review_id=row['id'],handoff_bars=handoff,target_cue_nudge_frames=frames,
                incoming_level_gain=incoming_gain,output_gain=output_gain,
                source_start_seconds=values[0],base_target_start_seconds=values[1],source_audio_sha256=source_sha,target_audio_sha256=target_sha,
                scope='Previously approved settings for these exact parts, song versions, tempo, overlap and loop. Newly rendered audio is still unrated.',new_render_approved=False)
        except (ValueError,TypeError,KeyError,json.JSONDecodeError):continue
    return dict(enabled=True,applied=False,reason='No current approved controlled-render configuration',ignored=ignored)
