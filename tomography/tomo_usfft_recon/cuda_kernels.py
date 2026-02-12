import cupy as cp

gather_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void gather(float2* g, float2* f, float* theta, int m, float* mu,
                                  int n, int ntheta, int nz, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= ntheta || tz >= nz) return;

    float M_PI = 3.141592653589793238f;
    float2 g0, g0t;
    float w, coeff0;
    float w0, w1, x0, y0, coeff1;
    int ell0, ell1, g_ind, f_ind, f_indx, f_indy;

    g_ind = tx + ty * n + tz * n * ntheta;  

    if (dir == 0) g0 = {};
    else g0 = g[g_ind];

    coeff0 = M_PI / mu[0];
    coeff1 = -M_PI * M_PI / mu[0];

    x0 = (tx - n / 2) / (float)n * __cosf(theta[ty]);
    y0 = -(tx - n / 2) / (float)n * __sinf(theta[ty]);

    for (int i1 = 0; i1 < 2 * m + 1; i1++)
    {
        ell1 = floorf(2 * n * y0) - m + i1;
        for (int i0 = 0; i0 < 2 * m + 1; i0++)
        {
            ell0 = floorf(2 * n * x0) - m + i0;

            w0 = ell0 / (float)(2 * n) - x0;
            w1 = ell1 / (float)(2 * n) - y0;

            w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));
            
            f_indx = (n + ell0 + 2 * n) % (2 * n);
            f_indy = (n + ell1 + 2 * n) % (2 * n);

            f_ind = f_indx + (2 * n) * f_indy + tz * (2 * n) * (2 * n);

            if (dir == 0)
            {
                g0.x += w * f[f_ind].x;
                g0.y += w * f[f_ind].y;
            }
            else
            {
                atomicAdd(&(f[f_ind].x), w * g0.x);
                atomicAdd(&(f[f_ind].y), w * g0.y);
            }
        }
    }

    if (dir == 0)
    {
        g[g_ind].x = g0.x / n;
        g[g_ind].y = g0.y / n;
    }
}
""",
    "gather",
)
