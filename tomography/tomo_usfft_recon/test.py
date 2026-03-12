import cupy as cp
from tomo import Tomo

def test_proj():
    obj = cp.load('obj.npy') + 2*1j*cp.load('obj.npy').swapaxes(1,2)
    [nz, n, n] = obj.shape

    ntheta = 3*n//4
    theta = cp.linspace(0, cp.pi, ntheta).astype('float32')
    cl = Tomo(n, theta, mask_r=1, raxis=None)

    cp.random.seed(seed=0)
    a = cp.random.random([nz, n, n]).astype('float32')
    b = cp.random.random([nz, ntheta, n]).astype('float32')
    Ra = cl.fwd_tomo(a)
    RTb = cl.adj_tomo(b)

    lhs = float(cp.real(cp.sum(a * RTb.conj())))
    rhs = float(cp.real(cp.sum(Ra * b.conj())))

    # Both sides should agree with each other (adjoint test)
    cp.testing.assert_allclose(lhs, rhs, rtol=1e-2,
        err_msg=f"Adjoint mismatch: <a, R^Tb>={lhs:.1f} vs <Ra, b>={rhs:.1f}")

    # Each side should be close to the NVIDIA reference value (178253)
    # rtol=0.005 = 0.5% tolerance, comfortably covers the observed 0.23% AMD/NVIDIA gap
    reference = 178253
    cp.testing.assert_allclose(lhs, reference, rtol=5e-3,
        err_msg=f"Value {lhs:.1f} deviates too far from reference {reference}")
