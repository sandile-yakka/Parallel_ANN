/*
 * neural_net.c
 *
 *  Created on: Apr 29, 2018
 *      Author: sandile
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

__const__ int len = 3;


//grid dim == num of input d_instances
//block dim == num of nodes in layer
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
	out[bidx*tdim + tidx ] = val ; //1.0/(1+exp(-val));
}
// calculate the delta for the outputs
//nb == num of input instances
//npb == num of output nodes

__global__ void delta_j(float* outputs, float* targets, float* deltaJ){

	int tidx = threadIdx.x;
	int bidx = blockIdx.x;
	int tdim = blockDim.x;
	int i = bidx*tdim + tidx;

		deltaJ[i] = outputs[i]*(1-outputs[i])*(targets[i]-outputs[i]);
}
//calculates the delta for any hidden layer
//num blocks == num instances
//threads per block == num nodes in layer
__global__ void delta_k(float* layer_outs ,float* deltaJ, int num_outs,
	float* nxt_weights, float* deltaK ){

	int tidx = threadIdx.x;
	int bidx = blockIdx.x;
	int tdim = blockDim.x;
	int idx = bidx*tdim + tidx;

	float sum = 0.0;
	for(int i = 0; i < num_outs; i++){
		sum += nxt_weights[tidx*num_outs + i]*deltaJ[bidx*num_outs + i];
	}
	deltaK[idx] = layer_outs[idx]*(1 - layer_outs[idx])*sum;
}
//grid dim == number of instances
//threads per block == the total number of weights in the network
__global__ void errDerivates(float* deltaJ, int dj_c, float* deltaK, int dk_c,
		float* in_lay1, int in1_size, float* in_lay2, int in2_size, float* output){

		int tidx = threadIdx.x;
		int bidx = blockIdx.x;
		int tdim = blockDim.x;
		int idx = bidx*tdim + tidx;

		if(tidx < in1_size*in2_size){ // if weight is in hidden layer
			//loop through to find the corresponding delta value
				// for(int i = 1; i <= in2_size; i++){
				// 	if((tidx+1)%i == 0){
				// 		idx_deltak = i-1;
				// 	} paused, gonna explore different logic
				// }
			//
			//so to get corresponding input value if we sayy
			// so if i say the index of the corresponding input is
			int idx_deltak = tidx % in2_size;
			int in_idx = floorf(tidx/in2_size);
			output[idx] = deltaK[bidx*tdim + idx_deltak] * in_lay1[bidx*tdim + in_idx];
		}
		else{
			 //last layer calculations
			 int prev_layer = in1_size * in2_size;
			 int tmp_tidx = tidx - prev_layer;
			 output[idx] = deltaJ[0] * in_lay2[bidx*in2_size + tmp_tidx];
		}
}
//grid dim == num of instances
//threads pb == number of weights
__global__ void reduction_kernel(float* errDerivates, float* output){
	int tidx = threadIdx.x;
	int bidx = blockIdx.x;
	int tdim = blockDim.x;

	atomicAdd(&output[tidx], errDerivates[tidx]);

}
//grid dim == 1 block
//num threads per block == total number of weights in network

__global__ void update_kernel(float* weights, float* new_weights, int lrate, float* deltas){
	int tidx = threadIdx.x;
	int bidx = blockIdx.x;
	int tdim = blockDim.x;
	int idx = bidx*tdim + tidx;

	new_weights[idx] = weights[idx] + lrate*deltas[idx];

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
	compute_layer<<<3,2>>>(d_instances, 3, d_weights, d_out,2);

	cudaMemcpy(outs, d_out, 4*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < 4; i++){
		printf("%f ", outs[i]);
	}
	printf("\n");

	return 0;
}
