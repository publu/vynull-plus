"""Fade controls used only by explicitly approved preview render settings."""
import numpy as np
def gains(frames,total_bars,handoff_bars=None):
    if frames<2 or total_bars<=0:raise ValueError('Invalid blend duration')
    t=np.linspace(0,1,frames)
    if handoff_bars is not None:
        if type(handoff_bars)!=int or handoff_bars not in [1,2,4,8] or handoff_bars>total_bars:raise ValueError('Invalid fade handoff')
        if handoff_bars!=total_bars:t=np.clip((t-(1-handoff_bars/total_bars))/(handoff_bars/total_bars),0,1)
    smooth=t*t*(3-2*t)
    return np.cos(smooth*np.pi/2)[:,None],np.sin(smooth*np.pi/2)[:,None]
