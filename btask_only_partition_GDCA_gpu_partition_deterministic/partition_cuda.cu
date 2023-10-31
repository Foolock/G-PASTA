#include <cuda.h>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits.h>
#include <ot/timer/timer.hpp>
#include "moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "moderngpu/src/moderngpu/kernel_mergesort.hxx"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

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

__device__ int binarySearch(int* segment_heads, int tid, int seg_heads_size) {
  int low = 0;
  int high = seg_heads_size - 1;
  
  while (low <= high) {
      int mid = low + (high - low) / 2;
      
      if (tid < segment_heads[mid]) {
          high = mid - 1;
      } else if (tid >= segment_heads[mid] && (mid == seg_heads_size - 1 || tid < segment_heads[mid + 1])) {
          return mid;
      } else {
          low = mid + 1;
      }
  }
  
  return -1; // tid is not within any segment
}

void Timer::partition_cpu_revised(
  std::vector<int>& dep_size, 
  std::vector<int>& partition_result_cpu, 
  std::vector<int>& partition_counter_cpu, 
  std::vector<int>& fu_partition, // the desired partition id for this node, initialized as -1 for each node
  int* max_partition_id
) {

  // find roots
  std::queue<int> to_visit;
  for(unsigned id=0; id<_adjp.size(); id++) {
    if(_vivekDAG._vtask_ptrs[id]->_fanin.size() == 0) {
      _topo_result_cpu.push_back(id);
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
      if(fu_partition[neighbor_id] < partition_result_cpu[cur_id]) {
        fu_partition[neighbor_id] = partition_result_cpu[cur_id];
      }
      dep_size[neighbor_id] --;
      if(dep_size[neighbor_id] == 0) {
        _topo_result_cpu.push_back(neighbor_id);
        to_visit.push(neighbor_id);
        int cur_partition = fu_partition[neighbor_id];
        if(partition_counter_cpu[cur_partition] < partition_size) {
          partition_result_cpu[neighbor_id] = cur_partition;
          partition_counter_cpu[cur_partition]++;
        }
        else {
          (*max_partition_id)++;
          int new_partition_id = *max_partition_id; 
          partition_result_cpu[neighbor_id] = new_partition_id;
          partition_counter_cpu[new_partition_id]++;
        }
      }
    }
  }
}

void Timer::partition_cpu(std::vector<int>& dep_size, std::vector<int>& partition_result_cpu, std::vector<int>& partition_counter_cpu, int* max_partition_id) {

  // find roots
  std::queue<int> to_visit;
  for(unsigned id=0; id<_adjp.size(); id++) {
    if(_vivekDAG._vtask_ptrs[id]->_fanin.size() == 0) {
      _topo_result_cpu.push_back(id);
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
      dep_size[neighbor_id] --;
      if(dep_size[neighbor_id] == 0) {
        _topo_result_cpu.push_back(neighbor_id);
        to_visit.push(neighbor_id);
        int cur_partition = partition_result_cpu[cur_id];
        if(partition_counter_cpu[cur_partition] < partition_size) {
          partition_result_cpu[neighbor_id] = cur_partition;
          partition_counter_cpu[cur_partition]++;
        }
        else {
          (*max_partition_id)++;
          int new_partition_id = *max_partition_id; 
          partition_result_cpu[neighbor_id] = new_partition_id;
          partition_counter_cpu[new_partition_id]++;
        }
      }
    }
  }
}

__global__ void check_result_gpu(
  int* d_topo_result_gpu, 
  int* d_topo_fu_partition_result_gpu,
  int* d_seg_heads,
  int* d_reduce_value,
  int* d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
  int* d_new_partition, // decide whether this node should apply d_incre_max_partition_id to get new partition id
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int* max_partition_id,
  int* max_incre_value_partition_id,
  int num_nodes, uint32_t* write_size,
  int* d_distance,
  int* d_level
) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid == 0) {
    printf("checking ---------------------------------------\n");
    printf("after pushed, d_topo_result_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_topo_result_gpu[i]);
    }
    printf("\n");
    printf("after pushed, d_topo_fu_partition_result_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_topo_fu_partition_result_gpu[i]);
    }
    printf("\n");
    printf("after pushed, d_partition_result_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_partition_result_gpu[i]);
    }
    printf("\n");
    printf("after pushed, d_seg_heads(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_seg_heads[i]);
    }
    printf("\n");
    printf("after pushed, d_incre_max_partition_id(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_incre_max_partition_id[i]);
    }
    printf("\n");
    printf("after pushed, d_new_partition(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_new_partition[i]);
    }
    printf("\n");
    printf("after pushed, d_partition_counter_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_partition_counter_gpu[i]);
    }
    printf("\n");
    printf("after pushed, max_partition_id= %d\n", *max_partition_id);
    printf("after pushed, max_incre_value_partition_id= %d\n", *max_incre_value_partition_id);
    printf("\n");
    printf("\n");
  }
}

__global__ void decide_partition_id( // decide partition id for each node 
  int* d_topo_result_gpu,
  int* d_topo_fu_partition_result_gpu, // future partition result stored in topoligical order
  int* d_partition_counter_gpu,
  int partition_size,
  int* max_partition_id,
  int* max_incre_value_partition_id, // maximum incremental value for max_partition_id
  int read_offset, uint32_t read_size,
  int* d_seg_heads, int num_seg_heads,
  int* d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
  int* d_new_partition // decide whether this node should apply d_incre_max_partition_id to get new partition id
) {
  
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < read_size) {
    // check which segments this tid belongs to
    int loc_seg = binarySearch(d_seg_heads, tid, num_seg_heads);
    // check how many nodes can still be merged in this partition
    int left_over = partition_size - d_partition_counter_gpu[d_topo_fu_partition_result_gpu[read_offset + tid]]; 
    if(tid < left_over + d_seg_heads[loc_seg]) {
      d_new_partition[read_offset + tid] = 0; // false
    }
    else {
      d_new_partition[read_offset + tid] = 1; // true
      d_incre_max_partition_id[read_offset + tid] = 1; // this value will be scanned later to finalize d_incre_max_partition_id
    }
  }
}

__global__ void assign_partition_id(
  int* d_topo_result_gpu,
  int* d_topo_fu_partition_result_gpu, // future partition result stored in topoligical order
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int* max_partition_id,
  int* max_incre_value_partition_id, // maximum incremental value for max_partition_id
  int read_offset, uint32_t read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
  int* d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
  int* d_new_partition // decide whether this node should apply d_incre_max_partition_id to get new partition id
) {

  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < read_size) {
    int tid_index = read_offset + tid; // the entry that tid is handling
    int cur_id = d_topo_result_gpu[tid_index]; // get current task id

    /*
     * assign partition id to cur nodes according to d_incre_max_partition_id and d_new_partition and d_topo_fu_partition_result_gpu 
     */
    if(d_new_partition[tid_index] == 1) {
      int new_partition_id = *max_partition_id + d_incre_max_partition_id[tid_index];
      d_partition_result_gpu[cur_id] = new_partition_id;
      atomicAdd(&d_partition_counter_gpu[new_partition_id], 1);
      atomicMax(max_incre_value_partition_id, d_incre_max_partition_id[tid_index]);
    }
    else {
      d_partition_result_gpu[cur_id] = d_topo_fu_partition_result_gpu[tid_index];
      atomicAdd(&d_partition_counter_gpu[d_topo_fu_partition_result_gpu[tid_index]], 1);
    }
  }
}

__global__ void partition_gpu_deterministic(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size, int* d_dep_size,
  int* d_topo_result_gpu,
  int* d_topo_fu_partition_result_gpu, // future partition result stored in topoligical order
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int partition_size,
  int* max_partition_id,
  int* max_incre_value_partition_id, // maximum incremental value for max_partition_id
  int read_offset, uint32_t read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
  uint32_t* write_size,
  int* d_distance,
  int* d_level,
  int* d_fu_partition, // the future partition this node will be assigned to(if partition not full)
  int* d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
  int* d_new_partition // decide whether this node should apply d_incre_max_partition_id to get new partition id
) {

  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid == 0) {
    *max_partition_id = *max_partition_id + *max_incre_value_partition_id;
    *max_incre_value_partition_id = 0;
  }

  if(tid < read_size) {
    int tid_index = read_offset + tid; // the entry that tid is handling
    int cur_id = d_topo_result_gpu[tid_index]; // get current task id

    // /*
    //  * assign partition id to cur nodes according to d_incre_max_partition_id and d_new_partition and d_topo_fu_partition_result_gpu 
    //  */
    // if(d_new_partition[tid_index] == 1) {
    //   int new_partition_id = *max_partition_id + d_incre_max_partition_id[tid_index];
    //   d_partition_result_gpu[cur_id] = new_partition_id;
    //   d_partition_counter_gpu[new_partition_id] ++;
    //   atomicMax(max_incre_value_partition_id, d_incre_max_partition_id[tid_index]);
    // }
    // else {
    //   d_partition_result_gpu[cur_id] = d_topo_fu_partition_result_gpu[tid_index];
    //   d_partition_counter_gpu[d_topo_fu_partition_result_gpu[tid_index]] ++;
    // }

    /*
     * then release dependents for its neighbor
     */
    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      
      atomicMax(&d_fu_partition[neighbor_id], d_partition_result_gpu[cur_id]);

      if(atomicSub(&d_dep_size[neighbor_id], 1) == 1) {
        int position = atomicAdd(write_size, 1); // no need to atomic here...
        d_topo_result_gpu[read_offset + read_size + position] = neighbor_id;        
        d_topo_fu_partition_result_gpu[read_offset + read_size + position] = d_fu_partition[neighbor_id];        
      }
    }
  }
}

__global__ void partition_gpu(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size, int* d_dep_size,
  int* d_topo_result_gpu,
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int partition_size,
  int* max_partition_id,
  int read_offset, uint32_t read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
  uint32_t* write_size,
  int* d_distance,
  int* d_level,
  int* d_fu_partition // the future partition this node will be assigned to(if partition not full)
) {

  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < read_size) {
    int cur_id = d_topo_result_gpu[read_offset + tid]; // get current task id
    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      
      atomicMax(&d_fu_partition[neighbor_id], d_partition_result_gpu[cur_id]);

      if(atomicSub(&d_dep_size[neighbor_id], 1) == 1) {
        int position = atomicAdd(write_size, 1); // no need to atomic here...
        d_topo_result_gpu[read_offset + read_size + position] = neighbor_id;        
        int cur_partition = d_fu_partition[neighbor_id]; // get the partition id of the parent that gives this neighbor the shortest path
        if(atomicAdd(&d_partition_counter_gpu[cur_partition], 1) < partition_size) { 
          d_partition_result_gpu[neighbor_id] = cur_partition; // no need to atomic here cuz only one thread will access this neighbor here
        }
        else {
          int new_partition_id = atomicAdd(max_partition_id, 1) + 1; // now we have new partition when we find cur_partition is full
                                                                     // we need to store this new partition id locally to the thread
          d_partition_result_gpu[neighbor_id] = new_partition_id;
          d_partition_counter_gpu[new_partition_id]++;  
        }
      }
    }
  }
}

void Timer::call_cuda_partition() {

  /*
   * modern gpu usage
   */
  mgpu::standard_context_t context(false);

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
  // std::cout << "num of source = " << source.size() << "\n";

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
  int* max_partition_id; // max_partition id we have currently, initially is source.size() - 1

  std::vector<int> distance(num_nodes, INT_MAX);
  // initialize distance
  for(unsigned i=0; i<source.size(); i++) {
    distance[source[i]] = 0;
  }
  int* d_distance; // d_distance[i] stores the shortest distance to node i from source in DAG
 
  std::vector<int> level(num_nodes, -1);
  for(unsigned i=0; i<source.size(); i++) {
    level[source[i]] = 0;
  }
  int* d_level;

  std::vector<int> fu_partition(num_nodes, -1);
  int* d_fu_partition;

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
  // _partition_counter_gpu.resize((num_nodes + partition_size - 1) / partition_size, 0);
  _partition_counter_gpu.resize(num_nodes, 0);
  for(unsigned i=0; i<source.size(); i++) { // at the beginning, each source corresponds to one partition
    _partition_counter_gpu[i]++;
  }
 
  // for testing_kernel_2
  std::vector<int> atomic_partition_counter_node(num_nodes, 0);
  for(auto id : source) {
    atomic_partition_counter_node[id] = 1;
  }

  checkError_t(cudaMalloc(&d_adjp, sizeof(int)*num_nodes), "d_adjp allocation failed");
  checkError_t(cudaMalloc(&d_adjncy, sizeof(int)*num_edges), "d_adjncy allocation failed");
  checkError_t(cudaMalloc(&d_adjncy_size, sizeof(int)*num_nodes), "d_adjncy_size allocation failed");
  checkError_t(cudaMalloc(&d_dep_size, sizeof(int)*num_nodes), "d_dep_size allocation failed");
  checkError_t(cudaMalloc(&d_topo_result_gpu, sizeof(int)*num_nodes), "d_topo_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_result_gpu, sizeof(int)*num_nodes), "d_partition_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_counter_gpu, sizeof(int)*num_nodes), "d_partition_counter_gpu allocation failed");
  checkError_t(cudaMalloc(&write_size, sizeof(uint32_t)), "write_size allocation failed");
  checkError_t(cudaMalloc(&max_partition_id, sizeof(int)), "max_partition_id allocation failed");
  checkError_t(cudaMalloc(&d_distance, sizeof(int)*num_nodes), "d_distance allocation failed");
  checkError_t(cudaMalloc(&d_level, sizeof(int)*num_nodes), "d_level allocation failed");
  checkError_t(cudaMalloc(&d_fu_partition, sizeof(int)*num_nodes), "d_fu_partition allocation failed");

  auto start = std::chrono::steady_clock::now();
  checkError_t(cudaMemcpy(d_adjp, _adjp.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjp memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy, _adjncy.data(), sizeof(int)*num_edges, cudaMemcpyHostToDevice), "d_adjncy memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy_size, _adjncy_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjncy_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_dep_size, _dep_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_dep_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_topo_result_gpu, source.data(), sizeof(int)*source.size(), cudaMemcpyHostToDevice), "d_topo_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_result_gpu, _partition_result_gpu.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_partition_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_counter_gpu, _partition_counter_gpu.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_partition_counter_gpu memcpy failed"); 
  checkError_t(cudaMemset(write_size, 0, sizeof(uint32_t)), "write_size memset failed");
  int max_partition_id_cpu = source.size() - 1;
  // checkError_t(cudaMemset(max_partition_id, 0x00000001, sizeof(int)), "max_partition_id memset failed");
  checkError_t(cudaMemcpy(max_partition_id, &max_partition_id_cpu, sizeof(int), cudaMemcpyHostToDevice), "max_partition_id memcpy failed"); 
  checkError_t(cudaMemcpy(d_distance, distance.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_distance memcpy failed"); 
  checkError_t(cudaMemcpy(d_level, level.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_level memcpy failed"); 
  checkError_t(cudaMemcpy(d_fu_partition, fu_partition.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_fu_partition memcpy failed"); 


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

    partition_gpu<<<num_block, BLOCK_SIZE>>>(
      d_adjp, d_adjncy, d_adjncy_size, d_dep_size,
      d_topo_result_gpu,
      d_partition_result_gpu,
      d_partition_counter_gpu,
      partition_size,
      max_partition_id,
      read_offset, read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
      write_size,
      d_distance,
      d_level,
      d_fu_partition
    );
   
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
    }

    /*
    check_result_gpu<<<1, 1>>>(
      d_topo_result_gpu,
      d_partition_result_gpu,
      d_partition_counter_gpu,
      max_partition_id,
      num_nodes, write_size,
      d_distance,
      d_level
    );
    */

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
  checkError_t(cudaMemcpy(&max_partition_id_cpu, max_partition_id, sizeof(int), cudaMemcpyDeviceToHost), "max_partition_id_cpu memcpy failed"); 
  _total_num_partitions_gpu = max_partition_id_cpu + 1;

  // print gpu topo result
  // std::cout << "_topo_result_gpu = [";
  // for(auto id : _topo_result_gpu) {
  //   std::cout << id << " ";
  // }
  // std::cout << "]\n"; 
  // std::cout << "_total_num_partitions_gpu = " << _total_num_partitions_gpu << "\n";

  // check partition result
  // for(auto partition : _partition_result_gpu) {
  //   if(partition == -1) {
  //     std::cerr << "something's wrong.\n";
  //     std::exit(EXIT_FAILURE);
  //   }
  // }
 
  checkError_t(cudaFree(d_adjp), "d_adjp free failed");
  checkError_t(cudaFree(d_adjncy), "d_adjncy free failed");
  checkError_t(cudaFree(d_adjncy_size), "d_adjncy_size free failed");
  checkError_t(cudaFree(d_dep_size), "d_dep_size free failed");
  checkError_t(cudaFree(d_topo_result_gpu), "d_topo_result_gpu free failed");
  checkError_t(cudaFree(d_partition_result_gpu), "d_partition_result_gpu free failed");
  checkError_t(cudaFree(d_partition_counter_gpu), "d_partition_counter_gpu free failed");
  checkError_t(cudaFree(write_size), "write_size free failed");
  checkError_t(cudaFree(max_partition_id), "max_partition_id free failed");
  checkError_t(cudaFree(d_distance), "d_distance free failed");
  checkError_t(cudaFree(d_level), "level free failed");
  checkError_t(cudaFree(d_fu_partition), "fu_partition free failed");
  
  /*
   * partition cpu version
   */

  std::vector<int> dep_size = _dep_size;
  _partition_result_cpu.resize(num_nodes, -1);
  int source_partition_id_cpu = 0;
  for(unsigned i=0; i<source.size(); i++) {
    _partition_result_cpu[source[i]] = source_partition_id_cpu;
    source_partition_id_cpu++;
  }
  _partition_counter_cpu.resize(num_nodes, 0);
  for(unsigned i=0; i<source.size(); i++) { // at the beginning, each source corresponds to one partition
    _partition_counter_cpu[i]++;
  }
  max_partition_id_cpu = source.size() - 1;
  std::vector<int> fu_partition_cpu(num_nodes, -1);
  partition_cpu_revised(dep_size, _partition_result_cpu, _partition_counter_cpu, fu_partition_cpu, &max_partition_id_cpu);
  _total_num_partitions_cpu = max_partition_id_cpu + 1;
 
  
  // std::cout << "_partition_result_cpu = [";
  // for(auto partition : _partition_result_cpu) {
  //   std::cout << partition << " ";
  // }
  // std::cout << "]\n"; 
  std::ofstream ofs("parts");
  for(auto partition : _partition_result_gpu) {
    ofs << partition << "\n";
  }
}

void Timer::call_cuda_partition_deterministic() {

  /*
   * modern gpu usage
   */
  mgpu::standard_context_t context(false);

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
  // std::cout << "num of source = " << source.size() << "\n";

  int* d_adjp; 
  int* d_adjncy; 
  int* d_adjncy_size;
  int* d_dep_size;
  int* d_topo_result_gpu;
  int* d_topo_fu_partition_result_gpu;
  int* d_partition_result_gpu;
  int* d_partition_counter_gpu;
  int read_offset = 0;
  uint32_t read_size = source.size();
  uint32_t* write_size;
  int* max_partition_id; // max_partition id we have currently, initially is source.size() - 1
  int* max_incre_value_partition_id; // maximum incremental value for max_partition_id, initially is 0

  std::vector<int> distance(num_nodes, INT_MAX);
  // initialize distance
  for(unsigned i=0; i<source.size(); i++) {
    distance[source[i]] = 0;
  }
  int* d_distance; // d_distance[i] stores the shortest distance to node i from source in DAG
 
  std::vector<int> level(num_nodes, -1);
  for(unsigned i=0; i<source.size(); i++) {
    level[source[i]] = 0;
  }
  int* d_level;

  std::vector<int> fu_partition(num_nodes, -1);
  int source_partition_id = 0;
  for(unsigned i=0; i<source.size(); i++) {
    fu_partition[source[i]] = source_partition_id;
    source_partition_id++;
  }
  int* d_fu_partition;

  std::vector<int> topo_fu_partition(num_nodes, -1);
  for(unsigned i=0; i<source.size(); i++) {
    topo_fu_partition[i] = fu_partition[source[i]];
  }

  std::vector<int> reduce_value(num_nodes, 1); // used for reduce by key
  int* d_reduce_value; // input values of reduce by key
  int* d_output_keys; // output keys of reduce by key
  int* d_seg_heads; // output values of reduce by key

  // reshape _topo_result_gpu
  _topo_result_gpu.resize(num_nodes, -1);

  // initialize partition with each source node as 1 partition
  _partition_result_gpu.resize(num_nodes, -1);

  // also initialize _partition_counter_gpu
  // _partition_counter_gpu[i] means the number of nodes in partition i 
  // the number of partition = (num_nodes + partition_size - 1) / partition_size
  // _partition_counter_gpu.resize((num_nodes + partition_size - 1) / partition_size, 0);
  _partition_counter_gpu.resize(num_nodes, 0);
 
  // for testing_kernel_2
  std::vector<int> atomic_partition_counter_node(num_nodes, 0);
  for(auto id : source) {
    atomic_partition_counter_node[id] = 1;
  }
  
  int* d_incre_max_partition_id; // the incremental value to max_partition_id for each node  
  int* d_new_partition; // decide whether this node should apply d_incre_max_partition_id to get new partition id
  std::vector<int> incre_max_partition_id(num_nodes, 0);
  std::vector<int> new_partition(num_nodes, 0);

  checkError_t(cudaMalloc(&d_adjp, sizeof(int)*num_nodes), "d_adjp allocation failed");
  checkError_t(cudaMalloc(&d_adjncy, sizeof(int)*num_edges), "d_adjncy allocation failed");
  checkError_t(cudaMalloc(&d_adjncy_size, sizeof(int)*num_nodes), "d_adjncy_size allocation failed");
  checkError_t(cudaMalloc(&d_dep_size, sizeof(int)*num_nodes), "d_dep_size allocation failed");
  checkError_t(cudaMalloc(&d_topo_result_gpu, sizeof(int)*num_nodes), "d_topo_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_topo_fu_partition_result_gpu, sizeof(int)*num_nodes), "d_topo_fu_partition_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_result_gpu, sizeof(int)*num_nodes), "d_partition_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_counter_gpu, sizeof(int)*num_nodes), "d_partition_counter_gpu allocation failed");
  checkError_t(cudaMalloc(&write_size, sizeof(uint32_t)), "write_size allocation failed");
  checkError_t(cudaMalloc(&max_partition_id, sizeof(int)), "max_partition_id allocation failed");
  checkError_t(cudaMalloc(&max_incre_value_partition_id, sizeof(int)), "max_incre_value_partition_id allocation failed");
  checkError_t(cudaMalloc(&d_distance, sizeof(int)*num_nodes), "d_distance allocation failed");
  checkError_t(cudaMalloc(&d_level, sizeof(int)*num_nodes), "d_level allocation failed");
  checkError_t(cudaMalloc(&d_fu_partition, sizeof(int)*num_nodes), "d_fu_partition allocation failed");
  checkError_t(cudaMalloc(&d_reduce_value, sizeof(int)*num_nodes), "d_reduce_value allocation failed");
  checkError_t(cudaMalloc(&d_output_keys, sizeof(int)*num_nodes), "d_output_keys allocation failed");
  checkError_t(cudaMalloc(&d_seg_heads, sizeof(int)*num_nodes), "d_seg_heads allocation failed");
  checkError_t(cudaMalloc(&d_incre_max_partition_id, sizeof(int)*num_nodes), "d_incre_max_partition_id allocation failed");
  checkError_t(cudaMalloc(&d_new_partition, sizeof(int)*num_nodes), "d_incre_max_partition_id allocation failed");

  auto start = std::chrono::steady_clock::now();
  checkError_t(cudaMemcpy(d_adjp, _adjp.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjp memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy, _adjncy.data(), sizeof(int)*num_edges, cudaMemcpyHostToDevice), "d_adjncy memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy_size, _adjncy_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjncy_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_dep_size, _dep_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_dep_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_topo_result_gpu, source.data(), sizeof(int)*source.size(), cudaMemcpyHostToDevice), "d_topo_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_topo_fu_partition_result_gpu, topo_fu_partition.data(), sizeof(int)*source.size(), cudaMemcpyHostToDevice), "d_topo_fu_partition_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_result_gpu, _partition_result_gpu.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_partition_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_counter_gpu, _partition_counter_gpu.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_partition_counter_gpu memcpy failed"); 
  checkError_t(cudaMemset(write_size, 0, sizeof(uint32_t)), "write_size memset failed");
  int max_partition_id_cpu = source.size() - 1;
  int max_incre_value_partition_id_cpu = 0;
  // checkError_t(cudaMemset(max_partition_id, 0x00000001, sizeof(int)), "max_partition_id memset failed");
  checkError_t(cudaMemcpy(max_partition_id, &max_partition_id_cpu, sizeof(int), cudaMemcpyHostToDevice), "max_partition_id memcpy failed"); 
  checkError_t(cudaMemcpy(max_incre_value_partition_id, &max_incre_value_partition_id_cpu, sizeof(int), cudaMemcpyHostToDevice), "max_incre_value_partition_id memcpy failed"); 
  checkError_t(cudaMemcpy(d_distance, distance.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_distance memcpy failed"); 
  checkError_t(cudaMemcpy(d_level, level.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_level memcpy failed"); 
  checkError_t(cudaMemcpy(d_fu_partition, fu_partition.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_fu_partition memcpy failed"); 
  checkError_t(cudaMemcpy(d_reduce_value, reduce_value.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_reduce_value memcpy failed"); 
  checkError_t(cudaMemcpy(d_incre_max_partition_id, incre_max_partition_id.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_incre_max_partition_id memcpy failed"); 
  checkError_t(cudaMemcpy(d_new_partition, new_partition.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_new_partition memcpy failed"); 

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
      
    assign_partition_id<<<num_block, BLOCK_SIZE>>>(
      d_topo_result_gpu,
      d_topo_fu_partition_result_gpu, // future partition result stored in topoligical order
      d_partition_result_gpu,
      d_partition_counter_gpu,
      max_partition_id,
      max_incre_value_partition_id, // maximum incremental value for max_partition_id
      read_offset, read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
      d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
      d_new_partition // decide whether this node should apply d_incre_max_partition_id to get new partition id
    );

    partition_gpu_deterministic<<<num_block, BLOCK_SIZE>>>(
      d_adjp, d_adjncy, d_adjncy_size, d_dep_size,
      d_topo_result_gpu,
      d_topo_fu_partition_result_gpu, // future partition result stored in topoligical order
      d_partition_result_gpu,
      d_partition_counter_gpu,
      partition_size,
      max_partition_id,
      max_incre_value_partition_id, // maximum incremental value for max_partition_id
      read_offset, read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
      write_size,
      d_distance,
      d_level,
      d_fu_partition,
      d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
      d_new_partition // decide whether this node should apply d_incre_max_partition_id to get new partition id
    );
   
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
    }

    // calculate where to read for next iteration
    read_offset += read_size;
    checkError_t(cudaMemcpy(&read_size, write_size, sizeof(uint32_t), cudaMemcpyDeviceToHost), "queue_size memcpy failed");

    // set write_size = 0 for next iteration 
    checkError_t(cudaMemset(write_size, 0, sizeof(uint32_t)), "write_size rewrite failed");

    if(read_size == 0) {
      continue;
    }
    /*
     * to make partition result deterministic
     * 1. sort d_topo_result[read_offset, read_offset + read_size - 1] by using d_fu_partition as their key
     * 2. reduce by key to get segment head for segmented sort
     * 3. segmented sort d_topo_result[read_offset, read_offset + read_size - 1] by using the d_topo_result[i].value() 
     */

    mgpu::mergesort(d_topo_fu_partition_result_gpu+read_offset, d_topo_result_gpu+read_offset, read_size, []MGPU_DEVICE(int a, int b){return a < b;}, context);
    thrust::pair<int*,int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, d_topo_fu_partition_result_gpu+read_offset, d_topo_fu_partition_result_gpu+read_offset+read_size, d_reduce_value+read_offset, d_output_keys, d_seg_heads); // no offset for output_keys and seg_heads(output_values), so they will be overwritten.  
    int num_seg_heads = static_cast<int>(new_end.second - (d_seg_heads));
    thrust::exclusive_scan(thrust::device, d_seg_heads, d_seg_heads + num_seg_heads, d_seg_heads, 0); // in-place scan for seg_heads(before seg_head[i] is length of each segments)
    mgpu::segmented_sort(d_topo_result_gpu+read_offset, read_size, d_seg_heads, num_seg_heads, []MGPU_DEVICE(int a, int b){return a < b;}, context);

    decide_partition_id<<<num_block, BLOCK_SIZE>>>( // decide partition id for each node 
      d_topo_result_gpu,
      d_topo_fu_partition_result_gpu, // future partition result stored in topoligical order
      d_partition_counter_gpu,
      partition_size,
      max_partition_id,
      max_incre_value_partition_id, // maximum incremental value for max_partition_id
      read_offset, read_size,
      d_seg_heads, num_seg_heads,
      d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
      d_new_partition // decide whether this node should apply d_incre_max_partition_id to get new partition id
    ); 

    thrust::inclusive_scan(thrust::device, d_incre_max_partition_id+read_offset, d_incre_max_partition_id+read_offset+read_size, d_incre_max_partition_id+read_offset);
 
    // std::cout << "read_offset = " << read_offset << "\n";
    // std::cout << "read_size = " << read_size << "\n";
    // check_result_gpu<<<1, 1>>>(
    //   d_topo_result_gpu,
    //   d_topo_fu_partition_result_gpu,
    //   d_seg_heads,
    //   d_reduce_value,
    //   d_incre_max_partition_id, // the incremental value to max_partition_id for each node  
    //   d_new_partition, // decide whether this node should apply d_incre_max_partition_id to get new partition id
    //   d_partition_result_gpu,
    //   d_partition_counter_gpu,
    //   max_partition_id,
    //   max_incre_value_partition_id, // maximum incremental value for max_partition_id
    //   num_nodes, write_size,
    //   d_distance,
    //   d_level
    // );
  }

  checkError_t(cudaMemcpy(_topo_result_gpu.data(), d_topo_result_gpu, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost), "_topo_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(_partition_result_gpu.data(), d_partition_result_gpu, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost), "_partition_result_gpu memcpy failed"); 
  auto end = std::chrono::steady_clock::now();
  GPU_topo_runtime += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  checkError_t(cudaMemcpy(&max_partition_id_cpu, max_partition_id, sizeof(int), cudaMemcpyDeviceToHost), "max_partition_id_cpu memcpy failed"); 
  std::cout << "max_partition_id_cpu = " << max_partition_id_cpu << "\n";
  _total_num_partitions_gpu = max_partition_id_cpu + 1;

  // print gpu topo result
  // std::cout << "_topo_result_gpu = [";
  // for(auto id : _topo_result_gpu) {
  //   std::cout << id << " ";
  // }
  // std::cout << "]\n"; 
  // std::cout << "_total_num_partitions_gpu = " << _total_num_partitions_gpu << "\n";

  // check partition result
  // for(auto partition : _partition_result_gpu) {
  //   if(partition == -1) {
  //     std::cerr << "something's wrong.\n";
  //     std::exit(EXIT_FAILURE);
  //   }
  // }
 
  checkError_t(cudaFree(d_adjp), "d_adjp free failed");
  checkError_t(cudaFree(d_adjncy), "d_adjncy free failed");
  checkError_t(cudaFree(d_adjncy_size), "d_adjncy_size free failed");
  checkError_t(cudaFree(d_dep_size), "d_dep_size free failed");
  checkError_t(cudaFree(d_topo_result_gpu), "d_topo_result_gpu free failed");
  checkError_t(cudaFree(d_topo_fu_partition_result_gpu), "d_topo_fu_partition_result_gpu free failed");
  checkError_t(cudaFree(d_partition_result_gpu), "d_partition_result_gpu free failed");
  checkError_t(cudaFree(d_partition_counter_gpu), "d_partition_counter_gpu free failed");
  checkError_t(cudaFree(write_size), "write_size free failed");
  checkError_t(cudaFree(max_partition_id), "max_partition_id free failed");
  checkError_t(cudaFree(max_incre_value_partition_id), "max_incre_value_partition_id free failed");
  checkError_t(cudaFree(d_distance), "d_distance free failed");
  checkError_t(cudaFree(d_level), "level free failed");
  checkError_t(cudaFree(d_fu_partition), "fu_partition free failed");
  checkError_t(cudaFree(d_reduce_value), "d_reduce_value free failed");
  checkError_t(cudaFree(d_output_keys), "d_output_keys free failed");
  checkError_t(cudaFree(d_seg_heads), "d_seg_heads free failed");
  checkError_t(cudaFree(d_incre_max_partition_id), "d_incre_max_partition_id free failed");
  checkError_t(cudaFree(d_new_partition), "d_incre_max_partition_id free failed");
  
  /*
   * partition cpu version
   */

  std::vector<int> dep_size = _dep_size;
  _partition_result_cpu.resize(num_nodes, -1);
  int source_partition_id_cpu = 0;
  for(unsigned i=0; i<source.size(); i++) {
    _partition_result_cpu[source[i]] = source_partition_id_cpu;
    source_partition_id_cpu++;
  }
  _partition_counter_cpu.resize(num_nodes, 0);
  for(unsigned i=0; i<source.size(); i++) { // at the beginning, each source corresponds to one partition
    _partition_counter_cpu[i]++;
  }
  max_partition_id_cpu = source.size() - 1;
  std::vector<int> fu_partition_cpu(num_nodes, -1);
  partition_cpu_revised(dep_size, _partition_result_cpu, _partition_counter_cpu, fu_partition_cpu, &max_partition_id_cpu);
  _total_num_partitions_cpu = max_partition_id_cpu + 1;
 
  
  // std::cout << "_partition_result_cpu = [";
  // for(auto partition : _partition_result_cpu) {
  //   std::cout << partition << " ";
  // }
  // std::cout << "]\n"; 
  std::ofstream ofs("parts");
  for(auto partition : _partition_result_gpu) {
    ofs << partition << "\n";
  }
}

};  // end of namespace ot. -----------------------------------------------------------------------



















