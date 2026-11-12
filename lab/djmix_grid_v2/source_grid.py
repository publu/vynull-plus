"""Fixed source-only grid fit; never accepts annotations or reference tempo."""
import numpy as np
def fit_grid(downbeats,bpm,duration,refine=False):
    down=np.asarray(downbeats,float);period=240/float(bpm);phase=float(down[0]);report=dict(refinement_requested=refine,refinement_accepted=False,original_bpm=float(bpm))
    if len(down)<2 or duration<=0 or not np.isfinite(down).all() or np.any(np.diff(down)<=0):raise ValueError('Invalid detected grid')
    if refine and len(down)>=32:
        count=np.r_[0,np.cumsum(np.maximum(1,np.rint(np.diff(down)/period)).astype(int))];keep=np.ones(len(down),bool)
        for _ in range(5):
            slope,intercept=np.linalg.lstsq(np.c_[count[keep],np.ones(keep.sum())],down[keep],rcond=None)[0]
            residual=down-(intercept+slope*count);center=np.median(residual[keep]);mad=np.median(abs(residual[keep]-center))
            keep=abs(residual-center)<=max(.08*period,3*mad)
            if keep.sum()<2:break
        p95=float(np.quantile(abs(residual),.95)/period);candidate=240/slope if slope>0 else 0
        accepted=bool(keep.mean()>=.8 and p95<=.15 and candidate>0 and abs(candidate/bpm-1)<=.06)
        report.update(retained_fraction=float(keep.mean()),p95_residual_bars=p95,fitted_bpm=float(candidate),refinement_accepted=accepted)
        if accepted:period=float(slope);phase=float(intercept)
    start=int(np.ceil((0-phase)/period));end=int(np.ceil((duration-.1-phase)/period))
    times=phase+np.arange(start,end)*period;times=times[(times>=0)&(times<duration-.1)]
    if len(times)<8:raise ValueError('Insufficient full-source grid')
    report.update(bpm=240/period,phase_seconds=phase,grid_nodes=len(times),original_first_downbeat=float(down[0]),original_last_downbeat=float(down[-1]),
        inferred_prefix_nodes=int(sum(times<down[0]-1e-6)),inferred_tail_nodes=int(sum(times>down[-1]+1e-6)),audio_samples_added=False)
    return times,240/period,report
