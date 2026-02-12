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

__global__ void rotateIm_kernel(float *img_out, cudaTextureObject_t texObj, float theta, int dimDetect)
{ 

	// Calculate normalized texture coordinates 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ((x < dimDetect) && (y < dimDetect)) {

    float u = (float)x - (float)dimDetect/2 + 0.5f; 
    float v = (float)y - (float)dimDetect/2 + 0.5f; 
    float tu = u*cosf(theta) - v*sinf(theta); 
    float tv = v*cosf(theta) + u*sinf(theta); 

    tu /= (float)dimDetect; 
    tv /= (float)dimDetect; 

    // read from texture and write to global memory
    img_out[y * dimDetect + x] += tex2D<float>(texObj, tu+0.5f, tv+0.5f);
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
   
		dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
		dim3 dimGrid(idivup(dimDetect,BLKXSIZE2D), idivup(dimDetect,BLKYSIZE2D));

        	// Allocate CUDA array in device memory
		cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaArray* cuArray;
		cudaMallocArray(&cuArray, &channelDesc, dimDetect, dimDetect);
		// Copy to device memory some data located at address h_data
		// in host memory
		//cudaMemcpyToArray(cuArray, 0, 0, image_out, dimDetect*dimDetect*sizeof(float), cudaMemcpyHostToDevice);
		
		cudaMemcpy2DToArray(cuArray, 0, 0,  image_out, sizeof(image_out) * dimDetect*dimDetect, 1, 1, cudaMemcpyHostToDevice);
	
		// Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		// Create texture object
		cudaTextureObject_t texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
		
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
        
        // Bind the array to the texture
        cudaMemcpyToArray(cuArray, 0, 0, image_rot_d, dimDetect*dimDetect*sizeof(float),cudaMemcpyDeviceToDevice);        
        //cudaMemcpy2DToArray(cuArray, 0, 0,  image_rot_d, sizeof(image_rot_d) * dimDetect*dimDetect, 1 , 1, cudaMemcpyDeviceToDevice);
        
        /* rotate image and accumulate the result */
        rotateIm_kernel<<<dimGrid,dimBlock>>>(image_out_d, texObj, proj_angles[k], dimDetect);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaPeekAtLastError() );   
		}
        /***************************************************************/    
        
        // Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);    
        
        // Copy the result from device to host memory
        cudaMemcpy(image_out,image_out_d,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

        cudaFree(sino_in_d);
        cudaFree(image_out_d);
        cudaFree(image_rot_d);
}
