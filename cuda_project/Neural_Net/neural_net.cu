/*
 * neural_net.c
 *
 *  Created on: Apr 29, 2018
 *      Author: sandile
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
// #include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// // CUDA helper functions
// #include <helper_cuda.h>
__const__ int len = 3;
__global__ void compute_layer(float* instances, const int len_instance, float* weights, float* out, int out_offset){
	__shared__ float instance[len];
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	int tdim = blockDim.x;

	for(unsigned int i = bidx + tidx; i < len_instance; i+= tdim){
			instance[i] = instances[i];
	}
	__syncthreads();
	//All threads have read instance data into memory
	float val = 0.0;
  //dot product
	for(unsigned int i = 0; i < len_instance; i++){
			val += instance[i] * weights[tidx + i*tdim];
	}

	 //apply sigmoid and write output
	out[bidx*out_offset + tidx ] = val ; //1.0/(1+exp(-val));

}
int main(){

	float* instances = (float*) malloc(6*sizeof(float));
	float* weights = (float*) malloc(6*sizeof(float));
	float* outs = (float*)malloc(4*sizeof(float));

	float* d_instances = 0;
	float* d_weights = 0;
	float* d_out = 0;

	cudaMalloc((void**)&d_instances, 6*sizeof(float));
	cudaMalloc((void**)&d_weights, 6*sizeof(float));
	cudaMalloc((void**)&d_out, 4*sizeof(float));

	for(int i = 0; i < 6; i++){
		instances[i] = 1;
	}
	for(int i = 0; i < 6; i++){
		weights[i] = 1;
	}
	cudaMemcpy(d_instances, instances, 6*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, 6*sizeof(float), cudaMemcpyHostToDevice);
 // const int len = 3;
	compute_layer<<<3,2>>>(d_instances, 3, d_weights, d_out,2);

	cudaMemcpy(outs, d_out, 4*sizeof(float), cudaMemcpyDeviceToHost);


	for(int i = 0; i < 4; i++){
		printf("%f ", outs[i]);
	}
	printf("\n");

	return 0;
}
