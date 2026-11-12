"""Source-only metrical evidence. No annotation or source metadata inputs."""
import numpy as np
import librosa
from scipy.signal import stft, fftconvolve

SR=16000
HOP=256
RATIOS=(.5,2/3,1.,1.5,2.)
LAGS=(.25,1/3,.5,2/3,1.,4/3,1.5,2.,3.,4.,8.,16.)

def fold(bpm):
    if not np.isfinite(bpm) or bpm<=0:raise ValueError('Invalid automatic tempo')
    while bpm<80:bpm*=2
    while bpm>180:bpm/=2
    return float(bpm)

def candidates(baseline):
    return np.array([baseline*r for r in RATIOS if 80<=baseline*r<=180],float)

def onset_bands(y):
    if len(y)<SR*30 or not np.isfinite(y).all():raise ValueError('Insufficient finite audio')
    f,_,z=stft(y,fs=SR,nperseg=1024,noverlap=1024-HOP,boundary=None,padded=False)
    mag=np.log1p(100*np.abs(z));flux=np.maximum(0,np.diff(mag,axis=1))
    bands=[]
    for low,high in [(20,8001),(20,200),(200,2000),(2000,8001)]:
        v=flux[(f>=low)&(f<high)].mean(axis=0)
        scale=np.quantile(v,.95)
        bands.append(np.minimum(v/max(scale,1e-8),4))
    return np.array(bands)

def acf(v):
    x=v-np.mean(v);den=np.dot(x,x)
    if den<1e-12:return np.zeros(len(x))
    a=fftconvolve(x,x[::-1],mode='full')[len(x)-1:]/den
    return a*np.minimum(2,len(x)/np.maximum(1,len(x)-np.arange(len(x))))

def from_bands(excerpts):
    if len(excerpts)!=3:raise ValueError('Three source excerpts required')
    envelope=np.concatenate([x[0] for x in excerpts])
    baseline=fold(float(librosa.feature.tempo(onset_envelope=envelope,sr=SR,hop_length=HOP,start_bpm=120)[0]))
    bpms=candidates(baseline);correlations=[[acf(v) for v in e] for e in excerpts];rows=[]
    for bpm in bpms:
        descriptors=[]
        for cor in correlations:
            row=[]
            for a in cor:
                lag=np.asarray(LAGS)*60/bpm*SR/HOP
                row.extend(np.interp(lag,np.arange(len(a)),a,left=0,right=0))
            descriptors.append(row)
        d=np.asarray(descriptors);rows.append(np.r_[baseline/120,bpm/120,bpm/baseline,np.log(bpm/baseline),d.mean(axis=0),d.std(axis=0)])
    x=np.array(rows)
    if not np.isfinite(x).all():raise ValueError('Nonfinite metrical features')
    return baseline,bpms,x

def reference_tempo(times,labels):
    times=np.asarray(times,float);labels=np.asarray(labels,int)
    if len(times)<128 or len(times)!=len(labels) or not np.isfinite(times).all():raise ValueError('Insufficient manual beats')
    intervals=np.diff(times);period=float(np.median(intervals))
    if period<=0 or np.quantile(abs(intervals-period),.95)>.02*period:raise ValueError('Nonconstant manual beat tempo')
    if not np.all((labels[1:]-labels[:-1])%4==1):raise ValueError('Nonconsecutive four-beat labels')
    bpm=60/period
    if not 80<=bpm<=180:raise ValueError('Manual tempo outside declared domain')
    return bpm

def target_index(bpms,reference):
    indices=np.flatnonzero(abs(np.asarray(bpms)/reference-1)<=.03)
    if len(indices)>1:raise ValueError('Ambiguous metrical target')
    return int(indices[0]) if len(indices) else -1
