#include "BackProjRB_GPU2D_core.h"

/* 
 * nvcc -Xcompiler -fPIC -shared -o BackProjRB_GPU2D_core.so BackProjRB_GPU2D_core.cu
 */

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define BLKXSIZE2D 16
#define BLKYSIZE2D 32


#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
struct square { __host__ __device__ float operator()(float x) { return x * x; } };

__device__ int wrapIndex(int index, int size)
{
    int wrapped = index % size;
    return (wrapped < 0) ? wrapped + size : wrapped;
}

__device__ float bilinearSampleWrap(const float *img, float x, float y, int dimDetect)
{
    const int x0 = (int)floorf(x);
    const int y0 = (int)floorf(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = x - (float)x0;
    const float ty = y - (float)y0;

    const int x0Wrapped = wrapIndex(x0, dimDetect);
    const int x1Wrapped = wrapIndex(x1, dimDetect);
    const int y0Wrapped = wrapIndex(y0, dimDetect);
    const int y1Wrapped = wrapIndex(y1, dimDetect);

    const float topLeft = img[y0Wrapped * dimDetect + x0Wrapped];
    const float topRight = img[y0Wrapped * dimDetect + x1Wrapped];
    const float bottomLeft = img[y1Wrapped * dimDetect + x0Wrapped];
    const float bottomRight = img[y1Wrapped * dimDetect + x1Wrapped];

    const float top = topLeft + tx * (topRight - topLeft);
    const float bottom = bottomLeft + tx * (bottomRight - bottomLeft);
    return top + ty * (bottom - top);
}

/************************************************/
__global__ void extractProj_kernel(float *sino, float *img, int yIndex, int dimDetect, int dimAngles)
{ 
	int i;
    //calculate each thread global index
    const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;       
    
    int index = xIndex + dimDetect*yIndex; 
    
    if ((xIndex < dimDetect) && (yIndex < dimAngles)) {
		
		 for(i=0; i<dimDetect; i++) {
			img[xIndex + dimDetect*i] = sino[index];
		 }
    }    
}

__global__ void rotateIm_kernel(float *img_out, float* img_in, float theta, int dimDetect)
{ 

	// Calculate normalized texture coordinates 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ((x < dimDetect) && (y < dimDetect)) {

    float u = (float)x - (float)dimDetect/2 + 0.5f; 
    float v = (float)y - (float)dimDetect/2 + 0.5f; 
    float tu = u*cosf(theta) - v*sinf(theta); 
    float tv = v*cosf(theta) + u*sinf(theta); 

    float sampleX = u*cosf(theta) - v*sinf(theta) + (float)dimDetect/2 - 0.5f;
    float sampleY = v*cosf(theta) + u*sinf(theta) + (float)dimDetect/2 - 0.5f;

    img_out[y * dimDetect + x] += bilinearSampleWrap(img_in, sampleX, sampleY, dimDetect);
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
////////////MAIN HOST FUNCTION ///////////////
extern "C" void BackProjGPU(float *sino_in, float *image_out, float *proj_angles, int dimDetect, int dimAngles)
{
    int deviceCount = -1; // number of devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return;
    }    
		const int blocksizeVec = 16;
		int gridSizeVec; 
		gridSizeVec = idivup(dimDetect,blocksizeVec);
		
		int SinoSize = dimAngles*dimDetect;    
		int ImSize = dimDetect*dimDetect;    
		float *sino_in_d, *image_out_d=NULL, *image_rot_d=NULL;
        float* img_in;
   
		dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
		dim3 dimGrid(idivup(dimDetect,BLKXSIZE2D), idivup(dimDetect,BLKYSIZE2D));

        // Allocate memory to hold the sinogram that gets given to the backprojector
		cudaMalloc(&img_in, sizeof(float) * dimDetect * dimDetect);
		
		/*allocate space for images on device*/
		checkCudaErrors( cudaMalloc((void**)&sino_in_d,SinoSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&image_out_d,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&image_rot_d,ImSize*sizeof(float)) );		
    
        checkCudaErrors( cudaMemcpy(sino_in_d,sino_in,SinoSize*sizeof(float),cudaMemcpyHostToDevice));
        cudaMemset(image_out_d, 0, ImSize*sizeof(float));
        cudaMemset(image_rot_d, 0, ImSize*sizeof(float));        
        
        /********************** Run CUDA 2D kernels here ********************/
        for (int k=0; k < dimAngles; k++) {                
        /* extract 1D projection and propagate it into 2D image */
        extractProj_kernel<<<gridSizeVec,blocksizeVec>>>(sino_in_d, image_rot_d, k, dimDetect, dimAngles);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );
        
        // Copy sinogram generated by forward projector into global memory
        checkCudaErrors( cudaMemcpy(img_in, image_rot_d, dimDetect*dimDetect*sizeof(float),cudaMemcpyDeviceToDevice) );
        
        /* rotate image and accumulate the result */
        rotateIm_kernel<<<dimGrid,dimBlock>>>(image_out_d, img_in, proj_angles[k], dimDetect);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaPeekAtLastError() );   
		}
        /***************************************************************/    
        
        // Copy the result from device to host memory
        cudaMemcpy(image_out,image_out_d,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

        cudaFree(sino_in_d);
        cudaFree(img_in);
        cudaFree(image_out_d);
        cudaFree(image_rot_d);
}
