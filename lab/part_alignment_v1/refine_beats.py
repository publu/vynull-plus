"""Conservative local onset tracking around an existing beat grid, preserving bar IDs."""
import numpy as np
from scipy.signal import stft

def refine(audio,sample_rate,beats,onset_bias=0):
    y=np.asarray(audio,np.float32)
    if y.ndim==2:y=y.mean(axis=1)
    beats=np.asarray(beats,float)
    if y.ndim!=1 or not np.isfinite(y).all() or len(beats)<8 or not np.isfinite(beats).all() or np.any(np.diff(beats)<=0):
        raise ValueError('Finite audio and at least eight ordered beat timestamps required')
    if beats[0]<0 or beats[-1]>len(y)/sample_rate:raise ValueError('Beats outside decoded audio')
    period=float(np.median(np.diff(beats)));limit=min(.1,.25*period)
    offsets=np.linspace(-limit,limit,2*int(np.ceil(limit/.004))+1)
    freq,t,z=stft(y,sample_rate,nperseg=1024,noverlap=896)
    magnitude=np.log1p(100*np.abs(z));novelties=[]
    for lo,hi in [(40,8000),(35,200)]:
        value=np.maximum(0,np.diff(magnitude[(freq>=lo)&(freq<=hi)],axis=1)).mean(axis=0)
        scale=max(float(np.quantile(value,.9)),1e-10)
        novelties.append(np.clip(value/scale,0,5))
    novelty=.8*novelties[0]+.2*novelties[1];times=t[1:]
    evidence=np.array([np.interp(beat+offsets+onset_bias,times,novelty,left=0,right=0) for beat in beats])
    # Smooth phase trajectory: at most 12ms change per beat, with a small-change prior.
    delta=offsets[:,None]-offsets[None,:]
    transition=.2*(delta/.008)**2;transition[abs(delta)>.0121]=np.inf
    score=evidence[0]-.05*(offsets/limit)**2;previous=[]
    for row in evidence[1:]:
        paths=score[:,None]-transition
        selected=np.argmax(paths,axis=0);previous.append(selected)
        score=paths[selected,np.arange(len(offsets))]+row-.05*(offsets/limit)**2
    path=[int(np.argmax(score))]
    for selected in reversed(previous):path.append(int(selected[path[-1]]))
    path=np.array(path[::-1]);proposed=offsets[path]
    chosen=evidence[np.arange(len(beats)),path];baseline=evidence[:,len(offsets)//2]
    local_best=np.max(evidence,axis=1)
    support=float(np.mean(chosen/np.maximum(local_best,.2)))
    evidence_ratio=float(np.mean(chosen)/max(float(np.mean(evidence)),1e-8))
    baseline_ratio=float(np.mean(baseline)/max(float(np.mean(chosen)),1e-8))
    confident=bool(np.max(abs(y))>1e-5 and support>=.5 and evidence_ratio>=1.5 and np.max(abs(proposed))<limit*.95)
    # Do not move an already plausible grid to chase every syncopated onset.
    applied=bool(confident and baseline_ratio<.55 and np.median(abs(proposed))>=.005)
    result=beats+proposed if applied else beats.copy()
    if np.any(np.diff(result)<=0):raise ValueError('Refinement broke beat ordering')
    return result,{'accepted_evidence':confident,'correction_applied':applied,
        'median_proposed_shift_ms':float(np.median(proposed)*1000),
        'proposed_offsets_ms':(proposed*1000).tolist(),'onset_bias_ms':onset_bias*1000,
        'evidence_ratio':evidence_ratio,'supported_fraction':support,'original_strength_ratio':baseline_ratio,
        'beat_period_ms':period*1000,'mean_candidate_strength':float(np.mean(chosen)),
        'mean_original_strength':float(np.mean(baseline)),
        'maximum_offset_ms':limit*1000,'bar_ids_preserved':True,
        'scope':'Local phase/drift refinement; does not establish a correct downbeat or musical compatibility'}
