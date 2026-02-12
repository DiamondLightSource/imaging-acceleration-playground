# Description: 

Rotation-based backprojection as an alternative to ray-tracing that is used in ASTRA-toolbox. For parallel-beam geometry you can rotate an object by means of interpolation and sum the lines for each angle, which is equivalent to forward projection. For inversion (the current code) you extract a 1D projection from the 2D sinogram (extractProj_kernel) and smear it by interpolation (rotate kernel) onto a 2D grid to obtain the backprojected image. This operation is performed in a loop over all detector positions.

# Installation:

You will need the nvcc compiler to build CUDA kernels provided with BackProjRB_GPU2D_core.cu. Then you build a shared object with: 
nvcc -Xcompiler -fPIC -shared -o BackProjRB_GPU2D_core.so BackProjRB_GPU2D_core.cu

Run DemoProj_bench.py to perform calculations and check the assertions of the L2 norm results both ASTRA and rotation-based backprojector. You can comment out astra projector if you want to it is there just for a reference. 

# Code origin / acknowledgment:

Provided by Daniil Kazantsev. 



