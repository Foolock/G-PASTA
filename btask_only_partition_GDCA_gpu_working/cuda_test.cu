#include <cuda.h>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ot/timer/timer.hpp>

#define BLOCK_SIZE 128 

namespace ot {

// ------------------------------------------------------------------------------------------------

void checkError(cudaError_t error, std::string msg) {
    if (error != cudaSuccess) {
        printf("%s: %d\n", msg.c_str(), error);
        std::exit(1);
    }
}

__global__ void test_kernel() {

  int num = threadIdx.x + 1; // a number between 1 to 8  
  int result = 1; // factorial result of the number

  for(int i=1; i<=num; i++) {
    result = result * i;
  }
  std::printf("%d!=%d\n", num, result);

}

__global__ void bfs_gpu_atomic_queue(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size,
  int* d_distance, int* d_parent,
  int* read_queue, int* write_queue,
  int rqueue_size, int* wqueue_size,
  int level
) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x; 

  if(tid < rqueue_size) {
    int cur_id = read_queue[tid];

    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      // unfinished.. cuz it only supports bfs instead of topological order
    }

  }
}

__global__ void topo_gpu_atomic_2queue(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size, int* d_dep_size,
  int* read_queue, int* write_queue,
  int rqueue_size, int* wqueue_size
) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < rqueue_size) {
    int cur_id = read_queue[tid];
    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      atomicSub(&d_dep_size[neighbor_id], 1);
      if(atomicMax(&d_dep_size[neighbor_id], 0) == 0) {
        int position = atomicAdd(wqueue_size, 1);
        write_queue[position] = neighbor_id;
      }
    }
  }
}

// all the read and write will be perform straightly on d_topo_result_gpu
__global__ void topo_gpu_atomic_centric_vector(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size, int* d_dep_size,
  int* d_topo_result_gpu,
  int read_offset, int read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
  int* write_size
) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < read_size) {
    int cur_id = d_topo_result_gpu[read_offset + tid];
    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      if(atomicSub(&d_dep_size[neighbor_id], 1) == 1) {
        int position = atomicAdd(write_size, 1); // no need to atomic here...
        d_topo_result_gpu[read_offset + read_size + position] = neighbor_id;        
      }
    }
  }
}

__global__ void check_d_topo_result_gpu(int* d_topo_result_gpu, int num_nodes) {
  
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(tid == 0) {
    printf("after pushed, d_topo_result_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_topo_result_gpu[i]);
    }
    printf("\n");
  }
}

void Timer::bfs_cpu(std::vector<int>& distance, std::vector<int>& parent) {

  // find roots
  std::queue<int> to_visit;
  for(int id=0; id<_adjp.size(); id++) {
    if(_vivekDAG._vtask_ptrs[id]->_fanin.size() == 0) {
      distance[id] = 0;
      to_visit.push(id);
    }
  }

  while(!to_visit.empty()) {
    int cur_id = to_visit.front();
    to_visit.pop();
    for(int offset=_adjp[cur_id]; offset<_adjp[cur_id] + _adjncy_size[cur_id]; offset++) {
      int neighbor_id = _adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      if(distance[neighbor_id] < distance[cur_id] + 1) {
        distance[neighbor_id] = distance[cur_id] + 1;
        parent[neighbor_id] = cur_id;
        to_visit.push(neighbor_id);
      }
    }
  }
}

void Timer::topo_cpu(std::vector<int>& dep_size) {
  
  // find roots
  std::queue<int> to_visit;
  for(int id=0; id<_adjp.size(); id++) {
    if(_vivekDAG._vtask_ptrs[id]->_fanin.size() == 0) {
      _topo_result_cpu.push_back(id);
      to_visit.push(id);
    }
  }
  
  auto start = std::chrono::steady_clock::now();
  while(!to_visit.empty()) {
    int cur_id = to_visit.front();
    to_visit.pop();
    for(int offset=_adjp[cur_id]; offset<_adjp[cur_id] + _adjncy_size[cur_id]; offset++) {
      int neighbor_id = _adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      dep_size[neighbor_id] --;
      if(dep_size[neighbor_id] == 0) {
        _topo_result_cpu.push_back(neighbor_id);
        to_visit.push(neighbor_id);
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  CPU_topo_runtime += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
}

void Timer::call_cuda_topo_2queue() {

  /*
  std::vector<int> parent(num_nodes); // used to store topological order(is it correct)
  std::vector<int> distance(num_nodes); // critical path of each node in DAG
  std::fill(distance.begin(), distance.end(), -1);
  std::fill(parent.begin(), parent.end(), -1);
  bfs_cpu(distance, parent);
  */

  /*
   * topological order cpu version
   */
  std::vector<int> dep_size = _dep_size;
  topo_cpu(dep_size);
  // print cpu topo result
  std::cout << "_topo_result_cpu = [";
  for(auto id : _topo_result_cpu) {
    std::cout << id << " ";
  }
  std::cout << "]\n";

  /*
   * topological order gpu version
   */
  unsigned num_nodes = _adjp.size();
  unsigned num_edges = _adjncy.size();

  std::vector<int> source;
  for(int id=0; id<_adjp.size(); id++) {
    if(_vivekDAG._vtask_ptrs[id]->_fanin.size() == 0) {
      source.push_back(id);
      _topo_result_gpu.push_back(id);
    }
  }

  int* d_adjp; 
  int* d_adjncy; 
  int* d_adjncy_size;
  int* d_dep_size;
  int* read_queue;
  int* write_queue;
  int rqueue_size;
  int* wqueue_size;
  
  checkError(cudaMalloc(&d_adjp, sizeof(int)*num_nodes), "d_adjp allocation failed");
  checkError(cudaMalloc(&d_adjncy, sizeof(int)*num_edges), "d_adjncy allocation failed");
  checkError(cudaMalloc(&d_adjncy_size, sizeof(int)*num_nodes), "d_adjncy_size allocation failed");
  checkError(cudaMalloc(&d_dep_size, sizeof(int)*num_nodes), "d_dep_size allocation failed");
  checkError(cudaMalloc(&read_queue, sizeof(int)*num_nodes), "read_queue allocation failed");
  checkError(cudaMalloc(&write_queue, sizeof(int)*num_nodes), "write_queue allocation failed");
  checkError(cudaMalloc(&wqueue_size, sizeof(int)), "wqueue_size allocation failed");
  checkError(cudaMemset(wqueue_size, 0, sizeof(int)), "wqueue_size memset failed");

  checkError(cudaMemcpy(d_adjp, _adjp.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjp memcpy failed"); 
  checkError(cudaMemcpy(d_adjncy, _adjncy.data(), sizeof(int)*num_edges, cudaMemcpyHostToDevice), "d_adjncy memcpy failed"); 
  checkError(cudaMemcpy(d_adjncy_size, _adjncy_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjncy_size memcpy failed"); 
  checkError(cudaMemcpy(d_dep_size, _dep_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_dep_size memcpy failed"); 
  checkError(cudaMemcpy(read_queue, source.data(), sizeof(int)*source.size(), cudaMemcpyHostToDevice), "read_queue memcpy failed"); 

  // invoke kernel
  rqueue_size = source.size();
  unsigned num_block; 
  while(rqueue_size > 0) {
    num_block = (num_nodes + rqueue_size - 1) / rqueue_size; 

    // use kernal as global synchronization for one level. results are stored in write_queue
    topo_gpu_atomic_2queue<<<num_block, BLOCK_SIZE>>>(
      d_adjp, d_adjncy, d_adjncy_size, d_dep_size,
      read_queue, write_queue,
      rqueue_size, wqueue_size
      );

    checkError(cudaMemcpy(&rqueue_size, wqueue_size, sizeof(int), cudaMemcpyDeviceToHost), "queue_size memcpy failed");

    // write results to _topo_gpu_result
    std::vector<int> tmp_result(rqueue_size);
    checkError(cudaMemcpy(tmp_result.data(), write_queue, sizeof(int)*rqueue_size, cudaMemcpyDeviceToHost), "write back result failed");
    _topo_result_gpu.insert(_topo_result_gpu.end(), tmp_result.begin(), tmp_result.end());

    std::cout << "rqueue_size" << rqueue_size << "\n";

    checkError(cudaMemset(wqueue_size, 0, sizeof(int)), "wqueue_size rewrite failed"); 
   
    // swap write_queue and read_queue
    std::swap(read_queue, write_queue);
  }

  // print gpu topo result
  std::cout << "_topo_result_gpu = [";
  for(auto id : _topo_result_gpu) {
    std::cout << id << " ";
  }
  std::cout << "]\n";

  cudaFree(&d_adjp);
  cudaFree(&d_adjncy);
  cudaFree(&d_adjncy_size);
  cudaFree(&d_dep_size);
  cudaFree(&read_queue);
  cudaFree(&write_queue);
  cudaFree(&wqueue_size);
 
  /*
  checkError(cudaFree(&d_adjp), "d_adjp free failed");
  checkError(cudaFree(&d_adjncy), "d_adjncy free failed");
  checkError(cudaFree(&d_adjncy_size), "d_adjncy_size free failed");
  checkError(cudaFree(&d_dep_size), "d_dep_size free failed");
  checkError(cudaFree(&read_queue), "read_queue free failed");
  checkError(cudaFree(&write_queue), "write_queue free failed");
  checkError(cudaFree(&wqueue_size), "wqueue_size free failed");
  */
}

void Timer::call_cuda_topo_centric_vector() {
  
  /*
   * topological order cpu version
   */
  std::vector<int> dep_size = _dep_size;
  topo_cpu(dep_size);
  // print cpu topo result
  // std::cout << "_topo_result_cpu = [";
  // for(auto id : _topo_result_cpu) {
  //   std::cout << id << " ";
  // }
  // std::cout << "]\n";
  
  /*
   * topological order gpu version
   */
  unsigned num_nodes = _adjp.size();
  unsigned num_edges = _adjncy.size();

  std::vector<int> source;
  for(int id=0; id<_adjp.size(); id++) {
    if(_vivekDAG._vtask_ptrs[id]->_fanin.size() == 0) {
      source.push_back(id);
    }
  }

  int* d_adjp; 
  int* d_adjncy; 
  int* d_adjncy_size;
  int* d_dep_size;
  int* d_topo_result_gpu;
  int read_offset = 0;
  int read_size = source.size();
  int* write_size;
  // reshape _topo_result_gpu
  _topo_result_gpu.resize(num_nodes);
  
  checkError(cudaMalloc(&d_adjp, sizeof(int)*num_nodes), "d_adjp allocation failed");
  checkError(cudaMalloc(&d_adjncy, sizeof(int)*num_edges), "d_adjncy allocation failed");
  checkError(cudaMalloc(&d_adjncy_size, sizeof(int)*num_nodes), "d_adjncy_size allocation failed");
  checkError(cudaMalloc(&d_dep_size, sizeof(int)*num_nodes), "d_dep_size allocation failed");
  checkError(cudaMalloc(&d_topo_result_gpu, sizeof(int)*num_nodes), "d_topo_result_gpu allocation failed");
  checkError(cudaMalloc(&write_size, sizeof(int)), "write_size allocation failed");

  auto start = std::chrono::steady_clock::now();
  checkError(cudaMemcpy(d_adjp, _adjp.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjp memcpy failed"); 
  checkError(cudaMemcpy(d_adjncy, _adjncy.data(), sizeof(int)*num_edges, cudaMemcpyHostToDevice), "d_adjncy memcpy failed"); 
  checkError(cudaMemcpy(d_adjncy_size, _adjncy_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjncy_size memcpy failed"); 
  checkError(cudaMemcpy(d_dep_size, _dep_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_dep_size memcpy failed"); 
  checkError(cudaMemcpy(d_topo_result_gpu, source.data(), sizeof(int)*source.size(), cudaMemcpyHostToDevice), "d_topo_result_gpu memcpy failed"); 
  checkError(cudaMemset(write_size, 0, sizeof(int)), "write_size memset failed");

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
    topo_gpu_atomic_centric_vector<<<num_block, BLOCK_SIZE>>>(
      d_adjp, d_adjncy, d_adjncy_size, d_dep_size,
      d_topo_result_gpu,
      read_offset, read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
      write_size
    );
   
    // calculate where to read for next iteration
    read_offset += read_size;
    checkError(cudaMemcpy(&read_size, write_size, sizeof(int), cudaMemcpyDeviceToHost), "queue_size memcpy failed");

    // set write_size = 0 for next iteration 
    checkError(cudaMemset(write_size, 0, sizeof(int)), "write_size rewrite failed");

    // check_d_topo_result_gpu<<<1, 1>>>(d_topo_result_gpu, read_offset + read_size);
  }

  checkError(cudaMemcpy(_topo_result_gpu.data(), d_topo_result_gpu, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost), "_topo_result_gpu memcpy failed"); 
 
  auto end = std::chrono::steady_clock::now();
  GPU_topo_runtime += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

  // print gpu topo result
  // std::cout << "_topo_result_gpu = [";
  // for(auto id : _topo_result_gpu) {
  //   std::cout << id << " ";
  // }
  // std::cout << "]\n"; 

  checkError(cudaFree(d_adjp), "d_adjp free failed");
  checkError(cudaFree(d_adjncy), "d_adjncy free failed");
  checkError(cudaFree(d_adjncy_size), "d_adjncy_size free failed");
  checkError(cudaFree(d_dep_size), "d_dep_size free failed");
  checkError(cudaFree(d_topo_result_gpu), "d_topo_result_gpu free failed");
  checkError(cudaFree(write_size), "write_size free failed");
}

};  // end of namespace ot. -----------------------------------------------------------------------



















