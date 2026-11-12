import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json,hashlib,time,sys
import numpy as np,librosa
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN/'djmix_grid_v2'));from source_grid import fit_grid
sys.path.insert(0,str(TRAIN/'semantic_pair_v1'));from audio_context import extract
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))
def main():
    s=read(HERE/'selection.json');assert all(sha(p)==h for p,h in s['code_sha256'].items());assert sha(s['tempo_prediction_path'])==s['tempo_prediction_sha256']
    pred=read(s['tempo_prediction_path']);assert pred['author_tempo_loaded'] is False;byid={r['id']:r for r in pred['rows']};assert set(byid)==set(s['inputs'])
    out=HERE/'features';out.mkdir(exist_ok=True);rows=[];started=time.time()
    for tid,item in s['inputs'].items():
        assert set(item)=={'id','member','path','sha256'} and sha(item['path'])==item['sha256'];r=byid[tid];assert r['source_sha256']==item['sha256']
        y,sr=librosa.load(item['path'],sr=22050,mono=True);duration=len(y)/sr;hint=r['predictions']['acoustic_plus_prior_gate'];assert 80<=hint<=180
        onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=512);_,frames=librosa.beat.beat_track(onset_envelope=onset,sr=sr,hop_length=512,bpm=hint,tightness=100,trim=True)
        down=librosa.frames_to_time(frames,sr=sr,hop_length=512)[::4];times,bpm,report=fit_grid(down,hint,duration,refine=True)
        raw=extract(item['path'],times,bpm,duration);dest=out/(tid+'.npz');assert not dest.exists();np.savez_compressed(dest,raw=raw,times=times,downbeats=times,bpm=bpm,duration=duration,audio_sha256=item['sha256'],grid_stride_bars=1)
        rows.append(dict(id=tid,member=item['member'],status='ready',source_sha256=item['sha256'],features_sha256=sha(dest),bpm=bpm,duration=duration,grid=report))
        write(HERE/'audio-progress.json',dict(records=rows,seconds=time.time()-started));print('EXCERPT_GRID_FEATURES',len(rows),'of19',tid,bpm,flush=True)
    write(HERE/'audio-complete.json',dict(created=time.time(),records=rows,seconds=time.time()-started,selection_sha256=sha(HERE/'selection.json'),author_annotations_or_tempo_read=False))
if __name__=='__main__':main()
