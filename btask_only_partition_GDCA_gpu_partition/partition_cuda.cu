#include <cuda.h>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ot/timer/timer.hpp>

#define BLOCK_SIZE 512 

namespace ot {

// ------------------------------------------------------------------------------------------------

// where should I put it if I don't want to modify headers in timer.hpp?
void checkError_t(cudaError_t error, std::string msg) {
    if (error != cudaSuccess) {
        printf("%s: %d\n", msg.c_str(), error);
        std::exit(1);
    }
}

__global__ void check_result_gpu(int* d_topo_result_gpu, int num_nodes, uint32_t* write_size) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid == 0) {
    printf("after pushed, d_topo_result_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_topo_result_gpu[i]);
    }
    printf("\n");
    printf("write_size = %d\n", *write_size);
  }
}

// all the read and write will be perform straightly on d_topo_result_gpu
__global__ void partition_gpu_atomic_centric_vector(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size, int* d_dep_size,
  int* d_topo_result_gpu,
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int partition_size,
  uint32_t* max_partition_id,
  int read_offset, uint32_t read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
  uint32_t* write_size
) {

  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < read_size) {
    int cur_id = d_topo_result_gpu[read_offset + tid]; // get current task id
    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      if(atomicSub(&d_dep_size[neighbor_id], 1) == 1) {
        int position = atomicAdd(write_size, 1); // no need to atomic here...
        d_topo_result_gpu[read_offset + read_size + position] = neighbor_id;        
        // also check if this partition id is full, if not, set this neighbor with the same id as its parent, if so, start a new partition id for this neighbor
        int cur_partition = d_partition_result_gpu[cur_id]; // get partition id of current task
        if(atomicAdd(&d_partition_counter_gpu[cur_partition], 1) < partition_size) { // there could be multiple threads with same cur_partition access this 
                                                                                     // notice partition_counter will be 5 when it is full(cuz we still add 1 in if condition) 
          d_partition_result_gpu[neighbor_id] = cur_partition; // no need to atomic here cuz only one thread will access this neighbor here
        }
        /*
        else {
          uint32_t new_partition_id = atomicAdd(max_partition_id, 1) + 1; // now we have new partition when we find cur_partition is full
                                                                          // we need to store this new partition id locally to the thread
          d_partition_result_gpu[neighbor_id] = new_partition_id;
          d_partition_counter_gpu[new_partition_id]++;  
        }
        */
      }
    }
  }
}

void Timer::call_cuda_partition() {
  
  /*
   * partition gpu version
   */
  unsigned num_nodes = _adjp.size();
  unsigned num_edges = _adjncy.size();

  std::vector<int> source;
  for(unsigned id=0; id<_adjp.size(); id++) {
    if(_vivekDAG._vtask_ptrs[id]->_fanin.size() == 0) {
      source.push_back(id);
    }
  }

  int* d_adjp; 
  int* d_adjncy; 
  int* d_adjncy_size;
  int* d_dep_size;
  int* d_topo_result_gpu;
  int* d_partition_result_gpu;
  int* d_partition_counter_gpu;
  int read_offset = 0;
  uint32_t read_size = source.size();
  uint32_t* write_size;
  // reshape _topo_result_gpu
  _topo_result_gpu.resize(num_nodes);

  // initialize partition with each source node as 1 partition
  _partition_result_gpu.resize(num_nodes, -1);
  int source_partition_id = 0;
  for(unsigned i=0; i<source.size(); i++) {
    _partition_result_gpu[source[i]] = source_partition_id;
    source_partition_id++;
  }

  // also initialize _partition_counter_gpu
  // _partition_counter_gpu[i] means the number of nodes in partition i 
  // the number of partition = (num_nodes + partition_size - 1) / partition_size
  uint32_t* max_partition_id; // max_partition id we have currently, initially is source.size() - 1
  _partition_counter_gpu.resize((num_nodes + partition_size - 1) / partition_size, 0);
  for(unsigned i=0; i<source.size(); i++) { // at the beginning, each source corresponds to one partition
    _partition_counter_gpu[i]++;
  }
  
  checkError_t(cudaMalloc(&d_adjp, sizeof(int)*num_nodes), "d_adjp allocation failed");
  checkError_t(cudaMalloc(&d_adjncy, sizeof(int)*num_edges), "d_adjncy allocation failed");
  checkError_t(cudaMalloc(&d_adjncy_size, sizeof(int)*num_nodes), "d_adjncy_size allocation failed");
  checkError_t(cudaMalloc(&d_dep_size, sizeof(int)*num_nodes), "d_dep_size allocation failed");
  checkError_t(cudaMalloc(&d_topo_result_gpu, sizeof(int)*num_nodes), "d_topo_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_result_gpu, sizeof(int)*num_nodes), "d_partition_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_counter_gpu, sizeof(int)*_partition_counter_gpu.size()), "d_partition_counter_gpu allocation failed");
  checkError_t(cudaMalloc(&write_size, sizeof(uint32_t)), "write_size allocation failed");
  checkError_t(cudaMalloc(&max_partition_id, sizeof(uint32_t)), "max_partition_id allocation failed");

  auto start = std::chrono::steady_clock::now();
  checkError_t(cudaMemcpy(d_adjp, _adjp.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjp memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy, _adjncy.data(), sizeof(int)*num_edges, cudaMemcpyHostToDevice), "d_adjncy memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy_size, _adjncy_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjncy_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_dep_size, _dep_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_dep_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_topo_result_gpu, source.data(), sizeof(int)*source.size(), cudaMemcpyHostToDevice), "d_topo_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_result_gpu, _partition_result_gpu.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_partition_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_counter_gpu, _partition_counter_gpu.data(), sizeof(int)*_partition_counter_gpu.size(), cudaMemcpyHostToDevice), "d_partition_counter_gpu memcpy failed"); 
  checkError_t(cudaMemset(write_size, 0, sizeof(uint32_t)), "write_size memset failed");
  checkError_t(cudaMemset(max_partition_id, source.size() - 1, sizeof(uint32_t)), "max_partition_id memset failed");

  // invoke kernel
  unsigned num_block;
  while(read_size > 0) { 
    // num_block = (num_nodes + read_size - 1) / read_size;
    num_block = (read_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if(num_block * BLOCK_SIZE < read_size) {
      std::cerr << "num of threads: " << num_block * BLOCK_SIZE << "\n";
      std::cerr << "read_size: " << read_size << "\n";
      std::cerr << "threads resource cannot handle one level of BFS node.\n";
      std::exit(EXIT_FAILURE);
    }

    // use kernal as global synchronization for one level. results are stored in d_topo_result_gpu 
    partition_gpu_atomic_centric_vector<<<num_block, BLOCK_SIZE>>>(
      d_adjp, d_adjncy, d_adjncy_size, d_dep_size,
      d_topo_result_gpu,
      d_partition_result_gpu,
      d_partition_counter_gpu,
      partition_size,
      max_partition_id,
      read_offset, read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
      write_size
    );
   
    // check_result_gpu<<<1, 1>>>(d_topo_result_gpu, read_offset + read_size, write_size);

    // calculate where to read for next iteration
    read_offset += read_size;
    checkError_t(cudaMemcpy(&read_size, write_size, sizeof(uint32_t), cudaMemcpyDeviceToHost), "queue_size memcpy failed");

    // set write_size = 0 for next iteration 
    checkError_t(cudaMemset(write_size, 0, sizeof(uint32_t)), "write_size rewrite failed");
  }

  checkError_t(cudaMemcpy(_topo_result_gpu.data(), d_topo_result_gpu, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost), "_topo_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(_partition_result_gpu.data(), d_partition_result_gpu, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost), "_partition_result_gpu memcpy failed"); 
 
  auto end = std::chrono::steady_clock::now();
  GPU_topo_runtime += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

  // print gpu topo result
  // std::cout << "_topo_result_gpu = [";
  // for(auto id : _topo_result_gpu) {
  //   std::cout << id << " ";
  // }
  // std::cout << "]\n"; 

  checkError_t(cudaFree(d_adjp), "d_adjp free failed");
  checkError_t(cudaFree(d_adjncy), "d_adjncy free failed");
  checkError_t(cudaFree(d_adjncy_size), "d_adjncy_size free failed");
  checkError_t(cudaFree(d_dep_size), "d_dep_size free failed");
  checkError_t(cudaFree(d_topo_result_gpu), "d_topo_result_gpu free failed");
  checkError_t(cudaFree(d_partition_result_gpu), "d_partition_result_gpu free failed");
  checkError_t(cudaFree(d_partition_counter_gpu), "d_partition_counter_gpu free failed");
  checkError_t(cudaFree(write_size), "write_size free failed");
  checkError_t(cudaFree(max_partition_id), "max_partition_id free failed");
}

};  // end of namespace ot. -----------------------------------------------------------------------



















