"""Inference counterpart of the original 244-dimensional training features."""
import subprocess
import numpy as np
from scipy.signal import stft
from scipy.fft import dct

def extract(path,times,bpm,duration,loops=None,return_audio=False):
    times=np.asarray(times,float)
    if not len(times) or bpm<=0 or duration<=0:raise ValueError('Valid candidate times, BPM and duration required')
    sr=16000
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(sr),'-f','f32le','pipe:1'],timeout=120)
    y=np.frombuffer(raw,np.float32)
    if not len(y) or not np.isfinite(y).all():raise ValueError('Invalid decoded audio')
    freq,frame_t,z=stft(y,sr,nperseg=1024,noverlap=768,boundary=None,padded=False)
    power=(abs(z)**2).astype(np.float32);del z
    total=power.sum(axis=0)+1e-12;log_rms=np.log(np.maximum(total,1e-12))
    centroid=(power*freq[:,None]).sum(axis=0)/total/8000
    flatness=np.exp(np.log(power+1e-12).mean(axis=0))/(power.mean(axis=0)+1e-12)
    norm=power/total;flux=np.r_[0,np.maximum(np.diff(norm,axis=1),0).sum(axis=0)]
    edges=np.geomspace(30,8000,25)
    bands=np.array([np.log(power[(freq>=lo)&(freq<hi)].sum(axis=0)+1e-12) for lo,hi in zip(edges,edges[1:])])
    mfcc=dct(bands,type=2,axis=0,norm='ortho')[:8]
    midi=np.zeros(len(freq));valid=freq>0;midi[valid]=69+12*np.log2(freq[valid]/440);pc=np.rint(midi).astype(int)%12
    chroma=np.array([power[(pc==p)&(freq>=50)&(freq<5000)].sum(axis=0) for p in range(12)])
    chroma/=chroma.sum(axis=0)+1e-12
    frame=np.vstack([log_rms,centroid,flatness,flux,bands,mfcc,chroma]).T.astype(np.float32)
    del power,norm
    sums=np.vstack([np.zeros((1,frame.shape[1])),np.cumsum(frame,axis=0,dtype=np.float64)])
    squares=np.vstack([np.zeros((1,frame.shape[1])),np.cumsum(frame.astype(np.float64)**2,axis=0)])
    def aggregate(a,b):
        i,j=np.searchsorted(frame_t,[max(0,a),b]);i=min(max(0,int(i)),len(frame)-1);j=min(len(frame),max(i+1,int(j)))
        mean=(sums[j]-sums[i])/(j-i);variance=np.maximum(0,(squares[j]-squares[i])/(j-i)-mean**2)
        return mean,np.sqrt(variance)
    bar=240/bpm;rows=[]
    for t in times:
        pre,_=aggregate(t-8*bar,t);mid,spread=aggregate(t-bar/2,t+bar/2);post,_=aggregate(t,t+8*bar)
        rows.append(np.r_[t/duration,t/bar,(duration-t)/bar,bpm,pre,mid,post,spread,post-pre])
    x=np.asarray(rows,np.float32)
    if x.shape!=(len(times),244) or not np.isfinite(x).all():raise ValueError('Invalid acoustic context features')
    if loops is None:return x
    loop_rows=[]
    for start,bars in loops:
        end=start+bars*bar
        if not 0<=start<end<=duration:raise ValueError('Loop lies outside the recording')
        begin,_=aggregate(start,start+bar);finish,_=aggregate(end-bar,end);inside,spread=aggregate(start,end)
        loop_rows.append(np.r_[bars,start/duration,begin,finish,inside,spread,abs(begin-finish)])
    result=(x,np.asarray(loop_rows,np.float32))
    return result+(y,sr) if return_audio else result
