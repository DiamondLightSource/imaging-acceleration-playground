#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from astraOP import AstraTools
from BackprojCtypes import get_backproj_RB_ctypes
import timeit

def rmse(im1, im2):
    a, b = im1.shape
    rmse = np.sqrt(np.sum((im1 - im2) ** 2) / float(a * b))    
    return rmse

def anorm(x):
    '''Calculate L2 norm over the last array dimention'''
    return np.sqrt((x*x).sum(-1))

class rotbased:
    """
    2D rotation-based (parallel beam) projector class
    """
    def __init__(self, proj_angles, dimDetect):
        self.proj_angles = proj_angles # in radians
        self.dimDetect = dimDetect # detectors dimension
        self.dimAngles = len(proj_angles) #angles number
    def backprojection(self, sinogram):
        if np.size(sinogram,0) != self.dimAngles:
            raise TypeError('The sinogram first dimension is not the angular one')
        bp_image = np.zeros((self.dimDetect,self.dimDetect),'float32') # backprojected image
        __cuda_bckprj = get_backproj_RB_ctypes()
        #Running CUDA backprojection kernel here
        __cuda_bckprj(sinogram, bp_image, self.proj_angles, self.dimDetect, self.dimAngles)
        return bp_image

dimAngles = 1500 # total number of angles
dimDetect = 1134 # horizontal detector size

input_file = open('InputData.bin','rb')
sinogram = np.fromfile(input_file, dtype='float32', sep="")
sinogram = sinogram.reshape([dimAngles,dimDetect])
print("Input projection data (sinogram) shape is {}".format(np.shape(sinogram)))
input_file.close()

theta = np.linspace(0, 179.9, num=dimAngles, dtype = 'float32')
angles_rad =theta*(np.pi/180) 
#######################################################################
tic=timeit.default_timer()
Atools = AstraTools(dimDetect, angles_rad, dimDetect, 'gpu') # initiate a class object
imageBP_astra = Atools.backproj(sinogram) # generate backprojection (A'b)
toc=timeit.default_timer()
run_time_bpASTRA = toc - tic
print("The elapsed time of ASTRA back-projector is {} seconds".format(run_time_bpASTRA))

L2_norm_ASTRA = (sum(anorm(imageBP_astra))/(dimAngles*dimDetect)) # 2597.9282
print("L2 norm for ASTRA back-projection equals to {} seconds".format(L2_norm_ASTRA))
assert L2_norm_ASTRA == 2597.9282

plt.figure(1) 
plt.imshow(imageBP_astra, cmap="BuPu")
plt.title('Backprojected (ASTRA) SL Phantom')
#######################################################################
RB_proj = rotbased(-angles_rad,dimDetect) # initiate a projector class

tic=timeit.default_timer()
imageBP = RB_proj.backprojection(np.float32(sinogram))
toc=timeit.default_timer()
run_time_bpRB = toc - tic
print("The elapsed time of rotation based back-projector is {} seconds".format(run_time_bpRB))

L2_norm_ROTBASED = (sum(anorm(imageBP))/(dimAngles*dimDetect))
print("L2 norm for rotation-based back-projection equals to {} seconds".format(L2_norm_ROTBASED))

assert L2_norm_ROTBASED == 2597.9304

plt.figure(2)
plt.imshow(imageBP, cmap="BuPu")
plt.title('Backprojected (RB) SL Phantom')

#outfile = open('Rot_based_recon.bin','wb')
#outfile.write(imageBP)
#outfile.close()
#%%
input_file = open('InputData.bin','rb')
input_data = np.fromfile(input_file, dtype='float32', sep="")
input_data = input_data.reshape([dimAngles,dimDetect])
input_file.close()

input_file = open('Rot_based_recon.bin','rb')
rot_based_rec = np.fromfile(input_file, dtype='float32', sep="")
rot_based_rec = rot_based_rec.reshape([dimDetect,dimDetect])
input_file.close()

plt.figure(3)
plt.subplot(121)
plt.imshow(input_data)
plt.title('Sinogram')
plt.subplot(122)
plt.imshow(rot_based_rec)
plt.title('Rot-based backprojection image')
#%%

