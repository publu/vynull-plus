import numpy as np
from preview_fades import gains
for n in [100,307200,614400]:
    t=np.linspace(0,1,n);s=t*t*(3-2*t);a,b=gains(n,8);np.testing.assert_array_equal(a,np.cos(s*np.pi/2)[:,None]);np.testing.assert_array_equal(b,np.sin(s*np.pi/2)[:,None])
    x,y=gains(n,8,8);np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
    x,y=gains(n,8,2);np.testing.assert_allclose(x*x+y*y,1,atol=1e-12);assert np.all(x[:n//2]==1) and np.all(y[:n//2]==0)
print('PASS: exact default fade parity and approved late-handoff envelope bounds')
