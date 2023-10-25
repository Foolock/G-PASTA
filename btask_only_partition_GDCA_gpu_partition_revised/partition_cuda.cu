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

  // std::cout << "_topo_result_cpu = [";
  // for(auto task : _topo_result_cpu) {
  //   std::cout << task << " ";
  // }
  // std::cout << "]\n";
  // std::cout << "corresponding partition result = [";
  // for(auto task : _topo_result_cpu) {
  //   std::cout << partition_result_cpu[task] << " ";
  // }
  // std::cout << "]\n";

}

__global__ void check_result_gpu(
  int* d_topo_result_gpu, 
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int* max_partition_id,
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
    printf("after pushed, d_partition_result_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_partition_result_gpu[i]);
    }
    printf("\n");
    printf("after pushed, d_partition_counter_gpu(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_partition_counter_gpu[i]);
    }
    printf("\n");
    printf("write_size = %d\n", *write_size);
    printf("max_partition_id = %d\n", *max_partition_id);
    printf("after pushed, d_distance(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_distance[i]);
    }
    printf("\n");
    printf("after pushed, d_level(threadId = %d) = ", tid);
    for(int i=0; i<num_nodes; i++) {
      printf("%d ", d_level[i]);
    }
    printf("\n");
    printf("\n");
    printf("\n");
  }
}

__global__ void testing_kernel(
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
      
      /*
       * rule for merge a node:  
       * 1. only check adjacent level parents
       * 2. compare distance for its parents, choose the parents with shortest distance
       * 3. when same distance, choose the parent with largest partition
       */

      // if(atomicMax(&d_level[neighbor_id], d_level[cur_id]+1) < d_level[cur_id]+1) { // only happen for one thread during a kernel
      //   d_distance[neighbor_id] = INT_MAX; // reset distance
      //   d_fu_partition[neighbor_id] = -1; // reset future partition
      // }

      // if(atomicMin(&d_distance[neighbor_id], d_distance[cur_id]+1) >= d_distance[cur_id]+1) { // happen for multiple threads
      //   atomicMax(&d_fu_partition[neighbor_id], d_partition_result_gpu[cur_id]);
      // }
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

/*
__global__ void partition_gpu_atomic_centric_vector(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size, int* d_dep_size,
  int* d_topo_result_gpu,
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int partition_size,
  int* max_partition_id,
  int read_offset, uint32_t read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
  uint32_t* write_size,
  int* d_parent
) {

  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < read_size) {
    int cur_id = d_topo_result_gpu[read_offset + tid]; // get current task id
    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      if(neighbor_id == -1) { // if _adjncy[offset] = -1, it means it has no fanout
        continue;
      }
      // for each neighbor, assign d_distance for it
      // if(atomicMin(&d_distance[neighbor_id], d_distance[cur_id]+1) >= d_distance[cur_id]+1) { // if neighbor_id is updated, store this into parents  
        // atomicMin(&d_parent[neighbor_id], cur_id);
      // }
      if(atomicSub(&d_dep_size[neighbor_id], 1) == 1) {
        atomicMin(&d_parent[neighbor_id], d_partition_result_gpu[cur_id]);
        int position = atomicAdd(write_size, 1); // no need to atomic here...
        d_topo_result_gpu[read_offset + read_size + position] = neighbor_id;        
        // also check if this partition id is full, if not, set this neighbor with the same id as its parent, if so, start a new partition id for this neighbor
        // int cur_partition = d_partition_result_gpu[cur_id]; // get partition id of current task
        // int cur_partition = d_partition_result_gpu[d_parent[neighbor_id]]; // get the partition id of the parent that gives this neighbor the shortest path
        int cur_partition = d_parent[neighbor_id]; // get the partition id of the parent that gives this neighbor the shortest path
        if(atomicAdd(&d_partition_counter_gpu[cur_partition], 1) < partition_size) { // there could be multiple threads with same cur_partition access this 
                                                                                     // notice partition_counter will be 5 when it is full(cuz we still add 1 in if condition) 
          d_partition_result_gpu[neighbor_id] = cur_partition; // no need to atomic here cuz only one thread will access this neighbor here
        }
        else {
          // before re-start a new partition, check neighbors of its parent cur_id to see if there is any neighbor i that has the d_parent[i] = cur_id

          int new_partition_id = atomicAdd(max_partition_id, 1) + 1; // now we have new partition when we find cur_partition is full
                                                                     // we need to store this new partition id locally to the thread
          d_partition_result_gpu[neighbor_id] = new_partition_id;
          d_partition_counter_gpu[new_partition_id]++;  
        }
      }
    }
  }
}
*/

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

    testing_kernel<<<num_block, BLOCK_SIZE>>>(
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

  // // check partition result
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
  partition_cpu(dep_size, _partition_result_cpu, _partition_counter_cpu, &max_partition_id_cpu);
  _total_num_partitions_cpu = max_partition_id_cpu + 1;
 
  
  // std::cout << "_partition_result_cpu = [";
  // for(auto partition : _partition_result_cpu) {
  //   std::cout << partition << " ";
  // }
  // std::cout << "]\n"; 
  // std::cout << "_partition_result_gpu = [";
  // for(auto partition : _partition_result_gpu) {
  //   std::cout << partition << " ";
  // }
  // std::cout << "]\n"; 
}

};  // end of namespace ot. -----------------------------------------------------------------------



















