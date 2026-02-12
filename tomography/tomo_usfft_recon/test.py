import cupy as cp
from tomo import Tomo

obj = cp.load('obj.npy')+2*1j*cp.load('obj.npy').swapaxes(1,2)
[nz,n,n] = obj.shape

ntheta = 3*n//4
theta = cp.linspace(0,cp.pi,ntheta).astype('float32')
cl = Tomo(n,theta,mask_r=1,raxis=None) # mask_r - circle radius for the mask, raxis - rotation axis (None if at the middle)

cp.random.seed(seed=0)
a = cp.random.random([nz,n,n]).astype('float32')
b = cp.random.random([nz,ntheta,n]).astype('float32')
Ra = cl.fwd_tomo(a)
RTb = cl.adj_tomo(b)
assert int(cp.real(cp.sum(a*RTb.conj()))) == 178253
assert int(cp.real(cp.sum(Ra*b.conj()))) == 178253

print("Test Complete")
