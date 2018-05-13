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


//grid dim == num of input instances
//block dim == num of nodes in layer
__global__ void compute_layer(float* instances, const int len_instance, float* weights, float* out){
	__shared__ float instance[len];
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	int tdim = blockDim.x;

	for(unsigned int i = tidx; i < len_instance; i+= tdim){
			instance[i] = instances[ bidx*len_instance + i];
	}
	__syncthreads();
	//All threads have read instance data into memory
	float val = 0.0;
  //dot product
	for(unsigned int i = 0; i < len_instance; i++){
			val += instance[i] * weights[tidx + i*tdim];
	}
	 //apply sigmoid and write output
	out[bidx*tdim + tidx] = val ; //1.0/(1+exp(-val));
}
// calculate the delta for the outputs
//nb == num of input instances
//npb == num of output nodes

__global__ void delta_j(float* outputs, float* targets, float* deltaJ){

	int tidx = threadIdx.x;
	int bidx = blockIdx.x;
	int tdim = blockDim.x;
	int i = bidx*tdim + tidx;
	printf("%f hello \n", outputs[i]*(1-outputs[i])*(targets[i]-outputs[i]) );
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
	printf("%f \n", layer_outs[idx]*(1 - layer_outs[idx])*sum );
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
			output[idx] = deltaK[bidx*tdim + idx_deltak] * in_lay1[bidx*in1_size + in_idx];
			// printf("%f \n",deltaK[bidx*dk_c + idx_deltak] * in_lay1[bidx*in1_size + in_idx]);
		}
		else{
			 //last layer calculations
			 int prev_layer = in1_size * in2_size;
			 int tmp_tidx = tidx - prev_layer;
			 output[idx] = deltaJ[bidx] * in_lay2[bidx*in2_size + tmp_tidx];
			printf("%f \n", deltaJ[bidx] * in_lay2[bidx*in2_size + tmp_tidx]);
		}
}
//grid dim == num of instances
//threads pb == number of weights
__global__ void reduction_kernel(float* errDerivates, float* output){
	int tidx = threadIdx.x;
	int bidx = blockIdx.x;
	int tdim = blockDim.x;

	atomicAdd(&output[ tidx], errDerivates[bidx*tdim +  tidx]);

}
//grid dim == 1 block
//num threads per block == total number of weights in network
__global__ void update_kernel(float* weights, int tw_lay1,float* weights2,float* new_w1,
	 float* new_w2, int lrate, float* deltas){
	int tidx = threadIdx.x;
	int bidx = blockIdx.x;
	int tdim = blockDim.x;
	int idx = bidx*tdim + tidx;

	if(idx < tw_lay1){
		new_w1[idx] = weights[idx] + lrate*deltas[idx];
		printf("%f \n", weights[idx] + lrate*deltas[idx] );
	}
	else{
		new_w2[idx] = weights2[idx-tw_lay1] + lrate*deltas[idx];
		printf("%f lay2 \n", weights2[idx-tw_lay1] + lrate*deltas[idx] );
	}
}

int main(){

	float* instances = (float*) malloc(6*sizeof(float));
	float* weights = (float*) malloc(9*sizeof(float));
	float* outs = (float*)malloc(6*sizeof(float));

	float* d_instances = 0;
	float* d_weights = 0;
	float* d_out = 0;

	cudaMalloc((void**)&d_instances, 6*sizeof(float));
	cudaMalloc((void**)&d_weights, 9*sizeof(float));
	cudaMalloc((void**)&d_out, 6*sizeof(float));

	for(int i = 0; i < 6; i++){
		instances[i] = 1;
		if(i >= 3){
			instances[i] = 0.5;
		}
	}

	for(int i = 0; i < 9; i++){
		weights[i] = 0.5;
		if(i%3 == 0) weights[i] = 1;
	}

	cudaMemcpy(d_instances, instances, 6*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, 9*sizeof(float), cudaMemcpyHostToDevice);
	compute_layer<<<2,3>>>(d_instances, 3, d_weights, d_out);

	cudaMemcpy(outs, d_out, 6*sizeof(float), cudaMemcpyDeviceToHost);
	printf("\n");
	for(int i = 0; i < 6; i++){
		printf("%f ", outs[i]);
	}
	printf("\n");

	// this is for the new layer
	printf("SECOND LAYER ******************\n");

	float* weights2 = (float*) malloc(3*sizeof(float));
	float* outs2 = (float*)malloc(2*sizeof(float));

	for(int i = 0; i < 3; i++) weights2[i] = 1;
	float* dn_instances = 0;
	float* dn_weights = 0;
	float* d_out2 = 0;
	cudaMalloc((void**)&dn_instances, 6*sizeof(float));
	cudaMalloc((void**)&dn_weights, 3*sizeof(float));
	cudaMalloc((void**)&d_out2, 2*sizeof(float));

	cudaMemcpy(dn_instances, outs, 6*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dn_weights, weights2, 3*sizeof(float), cudaMemcpyHostToDevice);
	compute_layer<<<2,1>>>(dn_instances, 3, dn_weights, d_out2);

	cudaMemcpy(outs2, d_out2, 2*sizeof(float), cudaMemcpyDeviceToHost);
	printf("\n");
	for(int i = 0; i < 2; i++){
		printf("%f ", outs2[i]);
	}
	printf("\n");

	float* targs = 0;
	float* dj = 0;
	cudaMalloc((void**)&targs, 2*sizeof(float));
	cudaMalloc((void**)&dj, 2*sizeof(float));
	float* targets = (float*) malloc(2*sizeof(float));
	targets[0] = 7;
	targets[1] = 4;

	cudaMemcpy(targs, targets, 2*sizeof(float), cudaMemcpyHostToDevice);

	delta_j<<<2,1>>>(d_out2, targs,dj);
	cudaDeviceSynchronize();
  float* dk = 0;
  cudaMalloc((void**)&dk, 6*sizeof(float));
	delta_k<<<2,3>>>(d_out, dj, 1, dn_weights , dk);
  cudaDeviceSynchronize();

	float* errd = 0;
	cudaMalloc((void**)&errd, 12*sizeof(float));
	printf("The error derivatives ********* \n");
  errDerivates<<<2,12>>>(dj, 2, dk, 3, d_instances, 3, d_out, 3, errd );

	float* tErros  = 0;
	cudaMalloc((void**)&tErros, 12*sizeof(float));

	reduction_kernel<<<2,12>>>(errd, tErros);

	float* errtd = (float*) malloc(12*sizeof(float));
	cudaMemcpy(errtd, tErros, 12*sizeof(float), cudaMemcpyDeviceToHost);

	printf("The error derivatives reduced \n");
	for(int i=0; i < 12; i++){
		printf("%f --- \n", errtd[i]);
	}


	float* n_weights = (float*) malloc(12*sizeof(float));

	float* lay1_w = 0;
	float* lay2_w = 0;
	cudaMalloc((void**)&lay1_w, 9*sizeof(float));
	cudaMalloc((void**)&lay2_w, 3*sizeof(float));
	update_kernel<<<1, 12>>>(d_weights, 9, dn_weights,  lay1_w, lay2_w, 1, tErros);

	cudaDeviceSynchronize();


	return 0;
}
