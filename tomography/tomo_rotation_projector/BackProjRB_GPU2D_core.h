#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#ifndef _BP_
#define _BP_

extern "C" void BackProjGPU(float *sino_in, float *image_out, float *proj_angles, int dimDetect, int dimAngles);   

#endif 
