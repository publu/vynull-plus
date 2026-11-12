"""Frozen music and matching acoustic contexts for the actual Halo library."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
from pathlib import Path
import json,time,hashlib,subprocess,urllib.request,fcntl,argparse
import numpy as np,torch
from huggingface_hub import snapshot_download
from transformers import ClapAudioModelWithProjection,ClapFeatureExtractor
from audio_context import extract

HERE=Path(__file__).parent;OUT=HERE/'halo_features';OUT.mkdir(exist_ok=True)
def api(route):
    with urllib.request.urlopen(os.environ.get('VYNULL_API','http://127.0.0.1:9443/api/')+route,timeout=60) as r:return json.load(r)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ids',default='');args=parser.parse_args()
    wanted={int(x) for x in args.ids.split(',') if x};suffix='-selected' if wanted else ''
    lock=(HERE/'halo-extract.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    torch.set_num_threads(4);source=json.load(open(HERE/'encoder.json'))
    snapshot=snapshot_download(source['model'],revision=source['revision'],local_files_only=os.environ.get('HF_HUB_OFFLINE')=='1',
        allow_patterns=['config.json','preprocessor_config.json',source['weights']],cache_dir=os.environ.get('VYNULL_HF_CACHE') or None)
    processor=ClapFeatureExtractor.from_pretrained(snapshot)
    model=ClapAudioModelWithProjection.from_pretrained(snapshot);model.requires_grad_(False);model.eval()
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    tracks=api('tracks');priorities=[5,126,10,101,3,16,146,163]
    if wanted:tracks=[t for t in tracks if t['id'] in wanted]
    tracks.sort(key=lambda t:(priorities.index(t['id']) if t['id'] in priorities else len(priorities),t['id']))
    ready={};failed={};started=time.time()
    for t in tracks:
        tid=t['id']
        try:
            analysis=api(f'analysis/{tid}');bpm=float(analysis['bpm'])
            downbeats=np.array([b['time_ms']/1000 for b in analysis['beats'] if b['beat_in_bar']==1])
            times=downbeats[::4];path=Path(t['file_path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)]))
            times=times[times<duration]
            if len(times)<4:raise ValueError('Too few bar candidates')
            config={'revision':source['revision'],'audio_sha256':digest,'bpm':bpm,'duration':duration,'version':'halo_context244_clap512_v1'}
            fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()+times.tobytes()).hexdigest()
            target=OUT/f'{tid}.npz'
            cached=False
            if target.exists():
                d=np.load(target);cached=str(d['fingerprint'])==fingerprint and np.isfinite(d['raw']).all() and np.isfinite(d['embeddings']).all()
            if not cached:
                raw=extract(path,times,bpm,duration)
                decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar','48000','-f','f32le','pipe:1'],timeout=120)
                y=np.frombuffer(decoded,'<f4');values=[]
                for i in range(0,len(times),8):
                    clips=[]
                    for stamp in times[i:i+8]:
                        begin=round(max(0,float(stamp)-5)*48000);clip=y[begin:begin+480000].copy()
                        if len(clip)<480000:clip=np.pad(clip,(0,480000-len(clip)))
                        clips.append(clip)
                    inputs=processor(clips,sampling_rate=48000,return_tensors='pt',padding='pad',truncation='rand_trunc')
                    with torch.inference_mode():
                        e=model(**{k:v.to(device) for k,v in inputs.items()}).audio_embeds
                        values.append(torch.nn.functional.normalize(e.float(),dim=-1).cpu().numpy())
                embeddings=np.concatenate(values)
                if embeddings.shape!=(len(times),512) or not np.isfinite(embeddings).all():raise ValueError('Invalid music embeddings')
                temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,times=times,raw=raw,embeddings=embeddings,
                    fingerprint=np.array(fingerprint),config=np.array(json.dumps(config)),downbeats=downbeats)
                temp.replace(target)
            ready[str(tid)]={'title':t['title'],'windows':len(times),'fingerprint':fingerprint}
            print('HALO_MUSIC_FEATURES',tid,t['title'],len(times),'cached',cached,flush=True)
        except Exception as e:failed[str(tid)]=repr(e);print('HALO_FEATURE_ERROR',tid,repr(e),flush=True)
        progress={'ready':ready,'failed':failed,'total':len(tracks),'seconds':time.time()-started,'device':device,'encoder_revision':source['revision']}
        temp=OUT/('progress'+suffix+'.tmp.json');temp.write_text(json.dumps(progress,indent=2));temp.replace(OUT/('progress'+suffix+'.json'))
    (OUT/('complete'+suffix+'.json')).write_text(json.dumps(progress,indent=2));print('HALO_MUSIC_FEATURES_COMPLETE',len(ready),len(failed),flush=True)

if __name__=='__main__':main()
