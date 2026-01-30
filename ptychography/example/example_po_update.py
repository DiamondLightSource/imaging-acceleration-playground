import h5py
import numpy as np
import matplotlib.pyplot as plt
from ptyupdate.update import PoUpdateCPUKernel

def plot(o, p, e, on, pn, title=""):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10,2))
    fig.suptitle(title)
    axes[0].imshow(np.angle(o[0]))
    axes[1].imshow(np.abs(p[0]))
    axes[2].imshow(np.abs(e[0]))
    axes[3].imshow(np.abs(on[0]))
    axes[4].imshow(np.abs(pn[0]), vmax=1.1*pn.mean())
    plt.show()

# Load example data
fname = "/dls/science/groups/imaging/ptypy_tutorials/data.h5"
fname = "/dls/science/groups/imaging/ptypy_tutorials/data_modes.h5"
with h5py.File(fname, "r") as f:
    ob = f["ob"][:]
    pr = f["pr"][:]
    ex = f["ex"][:]
    addr = f["addr"][:]

# Create denominator arrays
obn = np.ones(ob.shape, dtype="float")
prn = np.ones(pr.shape, dtype="float")

# Plotting
plot(ob, pr, ex, obn, prn, title="Before")

# Setup CPU kernel
POUK = PoUpdateCPUKernel()

# Object update
POUK.ob_update(addr, ob, obn, pr, ex)
ob /= obn

# Probe update
POUK.pr_update(addr, pr, prn, ob, ex)
pr /= prn

# Plotting
plot(ob, pr, ex, obn, prn, title="After")

