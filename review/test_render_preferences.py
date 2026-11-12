import tempfile,sqlite3,json
from pathlib import Path
from render_preferences import select
with tempfile.TemporaryDirectory() as folder:
    db=Path(folder)/'reviews.sqlite';c=sqlite3.connect(db);c.execute('CREATE TABLE transition_reviews(id INTEGER,vote INTEGER,preview_snapshot TEXT)')
    selection=dict(source_track=5,target_track=126,source_start_seconds=100.,target_start_seconds=20.,mix_bpm=120.,overlap_bars=8)
    def add(ident,vote=1,handoff=2,frames=0,source_sha='a',target_sha='b',loop=4,base=20.):
        p=dict(selection=dict(selection,target_start_seconds=base+frames/48000),source_audio_sha256=source_sha,target_audio_sha256=target_sha,eq_enabled=True,
            engine=dict(loop={'bars':loop} if loop else None,fade_envelope=dict(handoff_bars=handoff),target_alignment=dict(cue_nudge=dict(shift_frames=frames))),comparison=dict(kind='fade_exit'))
        c.execute('INSERT INTO transition_reviews VALUES(?,?,?)',(ident,vote,json.dumps(p)));c.commit()
    add(1);p=select(db,selection,'a','b',4);assert p['applied'] and p['handoff_bars']==2 and not p['new_render_approved']
    assert not select(db,selection,'changed','b',4)['applied'];assert not select(db,selection,'a','b',0)['applied'];assert not select(db,dict(selection,overlap_bars=4),'a','b',4)['applied']
    assert not select(db,dict(selection,target_start_seconds=40),'a','b',4)['applied'];assert not select(db,selection,'a','b',4,enabled=False)['applied']
    assert not select(db,dict(selection,target_tempo_factor=.98),'a','b',4)['applied']
    add(2,-1);assert not select(db,selection,'a','b',4)['applied']
    add(3,handoff=8,frames=36000);p=select(db,selection,'a','b',4);assert p['applied'] and p['target_cue_nudge_frames']==36000 and p['base_target_start_seconds']==20
    add(4,frames=99999999);assert select(db,selection,'a','b',4)['review_id']==3
    add(5,handoff=8,frames=36000,base=40);assert select(db,selection,'a','b',4)['review_id']==3
    c.close()
print('PASS: explicit approved render controls, source/version/parts/configuration isolation, rejection and feedback-off handling')
