import cupy as cp
import numpy as np
from tomo import Tomo


def get_gpu_info():
    device = cp.cuda.Device()
    return cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()


def gpu_timer(func, *args, warmup=3, repeats=20):
    for _ in range(warmup):
        func(*args)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        func(*args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    times = np.array(times)
    return {
        'mean_ms': times.mean(),
        'median_ms': np.median(times),
    }


def run_benchmark(n=512, nz=32, warmup=3, repeats=20):
    gpu = get_gpu_info()
    ntheta = 1000

    print(f"\n{'='*60}")
    print(f"  GPU      : {gpu}")
    print(f"  n={n}, nz={nz}, ntheta={ntheta}")
    print(f"  warmup={warmup}, repeats={repeats}")
    print(f"{'='*60}")

    theta = cp.linspace(0, cp.pi, ntheta).astype('float32')
    cl = Tomo(n, theta, mask_r=1, raxis=None)

    cp.random.seed(0)
    a = cp.random.random([nz, n, n]).astype('float32')
    b = cp.random.random([nz, ntheta, n]).astype('float32')

    fwd_stats = gpu_timer(cl.fwd_tomo, a, warmup=warmup, repeats=repeats)
    adj_stats = gpu_timer(cl.adj_tomo, b, warmup=warmup, repeats=repeats)

    for name, stats in [
        ('fwd_tomo', fwd_stats),
        ('adj_tomo', adj_stats),
    ]:
        print(
            f"  {name:<12} ---"
            f" {stats['mean_ms']:>8.4f} -- {stats['median_ms']:>8.4f}"
        )

    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_benchmark(n=3150,  nz=32)
