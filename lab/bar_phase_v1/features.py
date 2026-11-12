"""Phase-equivariant local acoustic features on an existing beat grid."""
import numpy as np

SELECT=np.r_[np.arange(4),np.arange(28,48)]
def descriptors(raw):
    if raw.ndim!=2 or raw.shape[1]!=244 or not np.isfinite(raw).all():raise ValueError('Expected finite 244-dimensional contexts')
    mid=raw[:,52:100][:,SELECT];previous=np.vstack([mid[:1],mid[:-1]])
    return np.c_[mid,raw[:,148:152],abs(mid-previous),raw[:,196:244][:,SELECT]].astype(np.float32)

def rotations(values):
    if values.shape!=(32,76) or not np.isfinite(values).all():raise ValueError('Expected 32 beats of 76 acoustic descriptors')
    center=np.median(values,axis=0);scale=np.maximum(.01,np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0))
    z=np.clip((values-center)/scale,-5,5)
    pools=np.array([np.r_[z[p::4].mean(axis=0),z[p::4].std(axis=0)] for p in range(4)])
    result=np.array([np.roll(pools,-p,axis=0).ravel() for p in range(4)],np.float32)
    if result.shape!=(4,608) or not np.isfinite(result).all():raise ValueError('Invalid phase rotations')
    return result

def label_windows(auto,manual_times,manual_labels):
    auto=np.asarray(auto);mt=np.asarray(manual_times);labels=np.asarray(manual_labels)
    period=float(np.median(np.diff(auto)));pos=np.searchsorted(mt,auto)
    lo=np.clip(pos-1,0,len(mt)-1);hi=np.clip(pos,0,len(mt)-1)
    nearest=np.where(abs(mt[lo]-auto)<=abs(mt[hi]-auto),lo,hi)
    valid=abs(mt[nearest]-auto)<=min(.08,.15*period)
    windows=[];targets=[]
    for start in range(0,len(auto)-31,16):
        sl=slice(start,start+32);ix=nearest[sl];ml=labels[ix];steps=np.diff(auto[sl])
        if not valid[sl].all() or not np.all(np.diff(ix)==1):continue
        if not np.array_equal(ml,(np.arange(32)+ml[0]-1)%4+1):continue
        if not np.all((steps>=.8*period)&(steps<=1.2*period)):continue
        windows.append(start);targets.append(int(np.flatnonzero(ml[:4]==1)[0]))
    return np.asarray(windows,dtype=int),np.asarray(targets,dtype=int)

def probabilities(logits,temperature):
    z=np.asarray(logits)/temperature;z=z-z.max(axis=1,keepdims=True);v=np.exp(z)
    return v/v.sum(axis=1,keepdims=True)
