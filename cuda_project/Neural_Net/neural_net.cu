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
	out[bidx*tdim + tidx] = 1.0/(1+exp(-val));
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

		if(tidx < in1_size*in2_size){
			// if weight is in hidden layer
			//loop through to find the corresponding delta value
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
		// printf("%f \n", weights[idx] + lrate*deltas[idx] );
	}
	else{
		new_w2[idx] = weights2[idx-tw_lay1] + lrate*deltas[idx];
		// printf("%f lay2 \n", weights2[idx-tw_lay1] + lrate*deltas[idx] );
	}
}

void read_function(float* targets, float* dataPoints, int max, char* name){

	char const* const fileName = name; /* should check that argc > 1 */
    FILE* file = fopen(fileName, "r"); /* should check the result */
    char line[1024];
    char* token;
    float temp_f = 00.0f;
    int i = 0; int j = 0;

    while (fgets(line, sizeof(line), file) && i < max) {
			j = 0;
			token = strtok(line, " ");
      while( token != NULL ) {
				    temp_f = atof(token);
						if(j == 0){
							targets[i] = temp_f;
							// printf("first token %f\n", atof(token) );
						}
						else{
							dataPoints[i*18 + j-1] = temp_f;
							// printf("%f %d \n",temp_f, j );
						}
            // printf( " %f", temp_f )
						// printf("****\n" );
            token = strtok(NULL, " ");
						j++;
    }
			i++;
		}
    fclose(file);
    // printf("%f %f %f\n", dataPoints[1500][613], dataPoints[1500][614], dataPoints[1500][615] );
}


int main(){

	int num_instances = 40000;
	int num_nodes = 5;

	float* targets = (float*) malloc(num_instances*sizeof(float));
	float* dataset = (float*) malloc(num_instances*18*sizeof(float));

	read_function(targets, dataset, num_instances, "SUSY.txt");

	float* w_weights = (float*) malloc(18 * num_nodes*sizeof(float));
//initialize random weights for the first layer
  for(int i = 0; i < 18 * num_nodes ; i++ ){
		w_weights[i] = (float)rand()/(float)(RAND_MAX/1);
	}
	float* u_weights = (float*) malloc(num_nodes*sizeof(float));
	//initialize random weights for the second layer
	for(int i = 0; i < num_nodes ; i++){
		u_weights[i] = (float)rand()/(float)(RAND_MAX/1);
	}

  float* outs = (float*)malloc(num_instances*sizeof(float));
	float* d_instances = 0;
	float* d_weights = 0;
	float* du_weights = 0;
	float* d_out = 0;

	cudaMalloc((void**)&d_instances, num_instances*18*sizeof(float));
	cudaMalloc((void**)&d_weights, num_nodes*18*sizeof(float));
	cudaMalloc((void**)&d_out, num_instances*num_nodes*sizeof(float));
	cudaMalloc((void**)&du_weights, num_nodes*sizeof(float));

	cudaMemcpy(d_instances, dataset, num_instances*18*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, w_weights, num_nodes*18*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(du_weights, u_weights, num_nodes*sizeof(float), cudaMemcpyHostToDevice);
	//put a for loop here
	// for(int epoch = 1; epoch < 100; epoch++)
	//begin feedforward
	for(int epoch = 0; epoch < 100; epoch++){
  //layer ____1____
	compute_layer<<<num_instances, num_nodes>>>(d_instances, 18, d_weights, d_out);

	float* layer2_out = 0;
	cudaMalloc((void**)&layer2_out, num_instances*sizeof(float));
	//layer ____2_____
	compute_layer<<<num_instances, 1>>>(d_out, num_nodes, du_weights, layer2_out);
	/*
	Feedforward operation is done now to backpropagate
	*/
  float* targs = 0;
	float* dj = 0;
	cudaMalloc((void**)&targs, num_instances*sizeof(float));
	cudaMalloc((void**)&dj, num_instances*sizeof(float));

	cudaMemcpy(targs, targets, num_instances*sizeof(float), cudaMemcpyHostToDevice);
	//
	delta_j<<<num_instances,1>>>(layer2_out, targs,dj);
	cudaDeviceSynchronize();
  float* dk = 0;
  cudaMalloc((void**)&dk, num_instances*num_nodes*sizeof(float));
	delta_k<<<num_instances, num_nodes>>>(d_out, dj, 1, du_weights , dk);
  cudaDeviceSynchronize();
	//
	float* errd = 0;
	int errd_size = (18*num_nodes + num_nodes);
	cudaMalloc((void**)&errd, num_instances*errd_size*sizeof(float));
  errDerivates<<<num_instances, errd_size>>>(dj, num_instances, dk, num_nodes, d_instances, 18, d_out, num_nodes, errd);
	float* tErros  = 0;
	cudaMalloc((void**)&tErros, errd_size*sizeof(float));

	reduction_kernel<<<num_instances,errd_size>>>(errd, tErros);

	float* lay1_w = 0;
	float* lay2_w = 0;
	cudaMalloc((void**)&lay1_w, 18*num_nodes*sizeof(float));
	cudaMalloc((void**)&lay2_w, num_nodes*sizeof(float));
	update_kernel<<<1, num_instances*errd_size>>>(w_weights, 18*num_nodes, du_weights, lay1_w, lay2_w, 0.5, tErros);
	cudaMemcpy(d_weights, lay1_w, 18*num_nodes*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(du_weights, lay2_w, num_nodes*sizeof(float), cudaMemcpyDeviceToHost);

	}
	// cudaDeviceSynchronize();
	return 0;
}
