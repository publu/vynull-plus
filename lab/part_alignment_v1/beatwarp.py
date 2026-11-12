"""Pitch-preserving, bounded piecewise beat alignment for offline mix rendering."""
from pathlib import Path
import subprocess, tempfile, warnings
import numpy as np
from scipy.io import wavfile

def warp(audio, sample_rate, source_seconds, target_seconds, output_seconds,
         minimum_tempo=.90, maximum_tempo=1.10):
    y=np.asarray(audio,np.float32)
    if y.ndim not in (1,2) or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected finite nonempty mono or stereo audio')
    source=np.rint(np.asarray(source_seconds)*sample_rate).astype(np.int64)
    target=np.rint(np.asarray(target_seconds)*sample_rate).astype(np.int64)
    frames=round(output_seconds*sample_rate)
    if source.ndim!=1 or source.shape!=target.shape or len(source)<2:
        raise ValueError('Matching source/target anchor arrays are required')
    if source[0]!=0 or target[0]!=0 or source[-1]!=len(y) or target[-1]!=frames:
        raise ValueError('Maps must include exact clip start and end frames')
    if np.any(np.diff(source)<=0) or np.any(np.diff(target)<=0):
        raise ValueError('Beat anchors must be strictly increasing')
    tempo=np.diff(source)/np.diff(target)
    if np.min(tempo)<minimum_tempo or np.max(tempo)>maximum_tempo:
        raise ValueError('Local tempo correction exceeds bounds')
    with tempfile.TemporaryDirectory(prefix='halo-beatwarp-') as folder:
        p=Path(folder);wavfile.write(p/'input.wav',sample_rate,y)
        # The CLI/library already imply (0,0) and the total-duration endpoint.
        # R3 computes its initial ratio from the first supplied map point;
        # supplying (0,0) causes a 0/0 reset and can shorten the output.
        interior=np.c_[source[1:-1],target[1:-1]]
        command=['rubberband','--fine','--centre-focus','--quiet','--time',str(frames/len(y))]
        if len(interior):
            np.savetxt(p/'map.txt',interior,fmt='%d');command+=['--timemap',str(p/'map.txt')]
        process=subprocess.run(command+[str(p/'input.wav'),str(p/'output.wav')],check=True,timeout=120,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if b'NaN or Inf' in process.stderr:raise ValueError('Time stretcher rejected the beat map')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=wavfile.WavFileWarning,message='Chunk.*not understood.*')
            sr,result=wavfile.read(p/'output.wav')
        attempts=[{'engine':'R3','output_frames':len(result),'duration_error_frames':len(result)-frames}]
    if result.dtype.kind in 'iu':result=result.astype(np.float32)/np.iinfo(result.dtype).max
    result=np.asarray(result,np.float32)
    if sr!=sample_rate or result.shape[1:]!=y.shape[1:] or not np.isfinite(result).all():
        raise ValueError('Invalid time-stretched output')
    # Permit only sample rounding; never synthesize a missing musical tail.
    raw_frames=len(result);tail=frames-raw_frames
    if abs(tail)>1:raise ValueError(f'Unexpected warped duration: {raw_frames} != {frames}')
    if tail>0:
        shape=(-1,)+(1,)*(result.ndim-1)
        extension=np.linspace(1,0,tail+1,dtype=np.float32)[1:].reshape(shape)*result[-1]
        result=np.concatenate([result,extension])
    elif tail<0:result=result[:frames]
    return result,{'engine':'Rubber Band R3 key-frame time map','engine_attempts':attempts,'pitch_shift_semitones':0,
        'origin_anchor_omitted':True,'supplied_interior_anchors':len(interior),
        'source_frames':len(y),'target_frames':frames,'raw_engine_output_frames':raw_frames,
        'tail_duration_correction_frames':tail,'anchor_count':len(source),
        'local_tempo_factor_min':float(np.min(tempo)),'local_tempo_factor_max':float(np.max(tempo)),
        'source_anchor_seconds':(source/sample_rate).tolist(),
        'target_anchor_seconds':(target/sample_rate).tolist()}

def repeat_loop(unit, repeats, edge_ms=3, sample_rate=22050):
    """Retain exact period; edge tapers do not shorten or overlap whole-bar units."""
    if type(repeats)!=int or repeats<1:raise ValueError('A positive integer repeat count is required')
    unit=np.array(unit,np.float32,copy=True)
    n=min(round(edge_ms*sample_rate/1000),len(unit)//4)
    if n:
        shape=(-1,)+(1,)*(unit.ndim-1)
        unit[:n]*=np.linspace(0,1,n).reshape(shape)
        unit[-n:]*=np.linspace(1,0,n).reshape(shape)
    return np.concatenate([unit]*repeats,axis=0)
