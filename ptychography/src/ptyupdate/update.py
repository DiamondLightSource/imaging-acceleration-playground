import numpy as np
from . import load_kernel

class Adict(object):
    def __init__(self):
        pass

class BaseKernel(object):

    def __init__(self):
        self.verbose = False
        self.npy = Adict()
        self.benchmark = {}

    def log(self, x):
        if self.verbose:
            print(x)


class PoUpdateCPUKernel(BaseKernel):

    def __init__(self):

        super(PoUpdateCPUKernel, self).__init__()
        self.kernels = [
            'pr_update',
            'ob_update',
        ]

    def allocate(self):
        pass

    def ob_update(self, addr, ob, obn, pr, ex):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            obn[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                (pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]).real
        return

    def pr_update(self, addr, pr, prn, ob, ex):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            prn[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                (ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols]).real
        return


class PoUpdateGPUKernel(PoUpdateCPUKernel):

    def __init__(self, queue_thread=None,
                 math_type='float', accumulator_type='float'):
        super(PoUpdateGPUKernel, self).__init__()
        # and now initialise the cuda
        if math_type not in ['double', 'float']:
            raise ValueError(
                'only float and double are supported for math_type')
        if accumulator_type not in ['double', 'float']:
            raise ValueError(
                'only float and double are supported for accumulator_type')
        self.math_type = math_type
        self.accumulator_type = accumulator_type
        self.queue = queue_thread
        self.norm = None
        self.ob_update_cuda = load_kernel("ob_update", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.ob_update2_cuda = None 
        self.pr_update_cuda = load_kernel("pr_update", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.pr_update2_cuda = None

    def ob_update(self, addr, ob, obn, pr, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        if obn.dtype != np.float32:
            raise ValueError(
                "Denominator must be float32 in current implementation")

        if self.queue is not None:
            self.queue.use()
        if atomics:
            if addr.shape[3] != 3 or addr.shape[2] != 5:
                raise ValueError(
                    'Address not in required shape for atomics ob_update')
            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.ob_update_cuda(grid=(int(num_pods), 1, 1),
                                block=(32, 32, 1),
                                args=(ex, num_pods, prsh[1], prsh[2],
                                      pr, prsh[0], prsh[1], prsh[2],
                                      ob, obsh[0], obsh[1], obsh[2],
                                      addr,
                                      obn))
        else:
            if addr.shape[0] != 5 or addr.shape[1] != 3:
                raise ValueError(
                    'Address not in required shape for tiled ob_update')
            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.ob_update2_cuda:
                self.ob_update2_cuda = load_kernel("ob_update2", {
                    "NUM_MODES": obsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16,
                    'IN_TYPE': 'float',
                    'OUT_TYPE': 'float',
                    'MATH_TYPE': self.math_type,
                    'ACC_TYPE': self.accumulator_type
                })

            grid = [int((x+15)//16) for x in ob.shape[-2:]]
            grid = (grid[1], grid[0], int(1))
            self.ob_update2_cuda(grid=grid,
                                 block=(16, 16, 1),
                                 args=(prsh[-1], obsh[0], num_pods, obsh[-2], obsh[-1],
                                       prsh[0],
                                       np.int32(ex.shape[0]),
                                       np.int32(ex.shape[1]),
                                       np.int32(ex.shape[2]),
                                       ob, obn, pr, ex, addr))

    def pr_update(self, addr, pr, prn, ob, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        if prn.dtype != np.float32:
            raise ValueError(
                "Denominator must be float32 in current implementation")
        if self.queue is not None:
            self.queue.use()
        if atomics:
            if addr.shape[3] != 3 or addr.shape[2] != 5:
                raise ValueError(
                    'Address not in required shape for atomics pr_update')

            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.pr_update_cuda(grid=(int(num_pods), 1, 1),
                                block=(32, 32, 1),
                                args=(ex, num_pods, prsh[1], prsh[2],
                                      pr, prsh[0], prsh[1], prsh[2],
                                      ob, obsh[0], obsh[1], obsh[2],
                                      addr,
                                      prn))
        else:
            if addr.shape[0] != 5 or addr.shape[1] != 3:
                raise ValueError(
                    'Address not in required shape for tiled pr_update')

            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.pr_update2_cuda:
                self.pr_update2_cuda = load_kernel("pr_update2", {
                    "NUM_MODES": prsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16,
                    'IN_TYPE': 'float',
                    'OUT_TYPE': 'float',
                    'MATH_TYPE': self.math_type,
                    'ACC_TYPE': self.accumulator_type
                })

            grid = [int((x+15)//16) for x in pr.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.pr_update2_cuda(grid=grid,
                                 block=(16, 16, 1),
                                 args=(prsh[-1], obsh[-2], obsh[-1],
                                       prsh[0], obsh[0], num_pods,
                                       pr, prn, ob, ex, addr))
