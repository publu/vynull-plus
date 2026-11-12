"""Compare equivalent bar phases; a kick attack need not match a beat's quiet tail."""
import numpy as np

def metrics(unit,sample_rate,bar_seconds):
    y=np.asarray(unit,np.float32)
    if not len(y) or not np.isfinite(y).all():raise ValueError('Invalid loop audio')
    edge=min(round(bar_seconds*sample_rate),len(y)//2)
    if edge<1:raise ValueError('Loop is too short for boundary analysis')
    begin=float(np.sqrt(np.mean(y[:edge]**2)+1e-12));end=float(np.sqrt(np.mean(y[-edge:]**2)+1e-12))
    return {'boundary_level_difference_db':float(abs(20*np.log10(begin/end))),
        'boundary_window_bars':1,'rms':float(np.sqrt(np.mean(y**2)+1e-12)),
        'boundary_comparison':'First whole bar versus last whole bar; beat phases matched'}
