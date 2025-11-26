import cupy as cp
import numpy as np
import logging
import time
from collections import deque
from ESD_Graph.structures.esd_graph import ESD_graph

# --- Algorithm 2 Kernel: Local Worklist ---
# Implements the logic from Algorithm 2 in the paper.
# - Processes nodes specifically listed in the current level's worklist.
# - Dynamically populates worklists for future levels using atomic operations.
LOCAL_WORKLIST_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void local_worklist_kernel(
    const int num_items,            // Number of active nodes in this worklist
    const int* current_worklist,    // Ptr to the active nodes for this level
    const int* nodes_v,             // Dest vertex (right(x))
    const int* nodes_a,             // Arrival time (arr(x))
    const int* adj_indptr,          // CSR pointers
    const int* adj_indices,         // CSR indices
    const int* node_levels,         // Look-up array for node levels
    int* start_times,               // Propagated Departure Times (startTime[x])
    int* status,                    // Active Status (0 or 1)
    int* wlist_storage,             // Global storage for all worklists
    const int* wlist_offsets,       // Start index for each level's worklist
    int* wlist_counters,            // Atomic counters for worklist size
    unsigned long long* packed_results // High 32: time, Low 32: predecessor
) {
    // 1. Calculate thread ID relative to the worklist
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Bounds check
    if (tid >= num_items) return;
    
    // 2. Retrieve the active node index from the worklist
    int idx = current_worklist[tid];
    
    int my_start_time = start_times[idx];
    int my_arrival = nodes_a[idx];
    int dest_vertex = nodes_v[idx];
    
    // 3. Compute Journey Time (Algorithm 2 Line 7)
    // journey = arr(x) - startTime[x]
    int journey = my_arrival - my_start_time;
    
    unsigned long long new_val = ((unsigned long long)journey << 32) | idx;
    atomicMin(&packed_results[dest_vertex], new_val);
    
    // 4. Propagate to Neighbors
    int start_edge = adj_indptr[idx];
    int end_edge = adj_indptr[idx + 1];
    
    for (int i = start_edge; i < end_edge; ++i) {
        int neighbor_idx = adj_indices[i];
        
        // atomicMax(startTime[y], startTime[x]) (Algorithm 2 Line 9)
        atomicMax(&start_times[neighbor_idx], my_start_time);
        
        // 5. Add to Future Worklist
        // if atomicCAS(status[y], false, true) == false then insert(y) (Algorithm 2 Line 10)
        // This ensures a node is added to its level's worklist exactly once.
        if (atomicCAS(&status[neighbor_idx], 0, 1) == 0) {
            
            // Determine which level the neighbor belongs to
            int neighbor_lvl = node_levels[neighbor_idx]; 
            
            // Calculate where to write in the global storage
            // wList[level(y)].insert(y)
            int level_offset = wlist_offsets[neighbor_lvl];
            int pos = atomicAdd(&wlist_counters[neighbor_lvl], 1);
            
            wlist_storage[level_offset + pos] = neighbor_idx;
        }
    }
}
''', 'local_worklist_kernel')

class ParallelESDG_LW:
    """
    GPU-accelerated Solver implementing Algorithm 2 (Local Worklist).
    Uses dynamic worklists on GPU to prune unreachable nodes and avoid 
    processing inactive nodes in sparse levels.
    """

    def __init__(self, esd_graph: ESD_graph):
        self.original_graph = esd_graph
        self.num_nodes = len(esd_graph.nodes)
        
        # Determine max vertex ID for result array sizing
        self.max_vertex_id = 0
        if self.num_nodes > 0:
            max_u = max(int(n.u) for n in esd_graph.nodes.values())
            max_v = max(int(n.v) for n in esd_graph.nodes.values())
            self.max_vertex_id = max(max_u, max_v)

        logging.info("Initializing Algorithm 2 (Local Worklist) Solver...")
        self._prepare_gpu_data()

    def _compute_levels(self):
        """
        Computes the topological level for every node in the ESDG.
        Same logic as ParallelESDG_LO.
        """
        in_degree = {i: 0 for i in self.original_graph.nodes}
        for u in self.original_graph.nodes:
            for v in self.original_graph.adj.get(u, []):
                in_degree[v] = in_degree.get(v, 0) + 1
        
        queue = deque([u for u in self.original_graph.nodes if in_degree[u] == 0])
        levels = {u: 1 for u in self.original_graph.nodes}
        
        visited_count = 0
        max_level = 0
        
        while queue:
            u = queue.popleft()
            current_lvl = levels[u]
            max_level = max(max_level, current_lvl)
            visited_count += 1
            
            for v in self.original_graph.adj.get(u, []):
                if levels[v] < current_lvl + 1:
                    levels[v] = current_lvl + 1
                
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        return levels, max_level

    def _prepare_gpu_data(self):
        """
        Prepares GPU data structures. 
        Specific to LW: Allocation of global worklist storage and level offsets.
        """
        t0 = time.perf_counter()
        
        # 1. Compute Levels
        node_levels_map, self.max_level = self._compute_levels()
        
        # 2. Sort/Relabel nodes for consistency (improves memory access patterns)
        sorted_node_ids = sorted(self.original_graph.nodes.keys(), 
                                 key=lambda k: (node_levels_map[k], k))
        
        self.id_map = {original: i for i, original in enumerate(sorted_node_ids)}
        
        # 3. Build Level Offsets & Max Level Sizes
        # We need to know how large each level is to allocate worklist space
        level_counts = cp.zeros(self.max_level + 2, dtype=cp.int32)
        h_level_offsets = np.zeros(self.max_level + 2, dtype=np.int32)
        
        # Create array of levels for the GPU kernel
        h_node_levels = np.zeros(self.num_nodes, dtype=np.int32)

        # Count nodes per level to build offsets
        temp_counts = {}
        current_level = 0
        
        for i, original_id in enumerate(sorted_node_ids):
            lvl = node_levels_map[original_id]
            h_node_levels[i] = lvl
            temp_counts[lvl] = temp_counts.get(lvl, 0) + 1

        # Calculate offsets (prefix sum of counts)
        cumulative = 0
        for l in range(1, self.max_level + 2):
            h_level_offsets[l] = cumulative
            count = temp_counts.get(l, 0)
            cumulative += count

        # 4. Build Node Attribute Arrays (SoA)
        nodes_v = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_a = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_u_cpu = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_t_cpu = np.zeros(self.num_nodes, dtype=np.int32)
        
        for i, original_id in enumerate(sorted_node_ids):
            node = self.original_graph.nodes[original_id]
            nodes_v[i] = int(node.v)
            nodes_a[i] = int(node.a)
            nodes_u_cpu[i] = int(node.u)
            nodes_t_cpu[i] = int(node.t)
            
        # 5. Build CSR Adjacency
        adj_indices_list = []
        adj_indptr = [0]
        
        for original_id in sorted_node_ids:
            neighbors = self.original_graph.adj.get(original_id, [])
            mapped_neighbors = [self.id_map[n] for n in neighbors if n in self.id_map]
            adj_indices_list.extend(mapped_neighbors)
            adj_indptr.append(len(adj_indices_list))
            
        # 6. Transfer to GPU
        self.d_nodes_v = cp.asarray(nodes_v)
        self.d_nodes_a = cp.asarray(nodes_a)
        self.d_adj_indices = cp.asarray(adj_indices_list, dtype=np.int32)
        self.d_adj_indptr = cp.asarray(adj_indptr, dtype=np.int32)
        
        # LW Specific: Transfer node levels so kernel knows where to push neighbors
        self.d_node_levels = cp.asarray(h_node_levels)
        
        # LW Specific: Worklist Structures
        # "Allocate worklists for each level... single long array"
        # Size = num_nodes (worst case, all nodes in worklists)
        self.d_wlist_storage = cp.zeros(self.num_nodes, dtype=cp.int32)
        self.d_wlist_offsets = cp.asarray(h_level_offsets)
        
        # Host-side source data
        self.h_nodes_u = nodes_u_cpu
        self.h_nodes_t = nodes_t_cpu
        self.h_node_levels = h_node_levels
        
        t1 = time.perf_counter()
        logging.info(f"LW Data Prep (L={self.max_level}) complete in {t1-t0:.4f}s")

    def find_fastest_paths(self, source_vertex_s: str, reconstruct_paths: bool = False):
        """
        Executes Algorithm 2 (Local Worklist).
        Iterates levels 1..L, launching kernels only for active nodes in worklists.
        """
        source_vertex_int = int(source_vertex_s)
        t_start = time.perf_counter()
        
        # --- 1. Initialization ---
        d_start_times = cp.full(self.num_nodes, -1, dtype=cp.int32)
        d_status = cp.zeros(self.num_nodes, dtype=cp.int32)
        
        # Reset worklist counters to 0 for this run
        d_wlist_counters = cp.zeros(self.max_level + 2, dtype=cp.int32)
        
        # Result arrays
        INF_TIME = 2147483647
        init_val = (INF_TIME << 32) | 0xFFFFFFFF
        d_packed_results = cp.full(self.max_vertex_id + 1, init_val, dtype=cp.uint64)
        
        source_init = (0 << 32) | 0xFFFFFFFF
        d_packed_results[source_vertex_int] = source_init

        # --- 2. Activate Source Nodes (Algorithm 2 Line 3-4) ---
        source_indices = np.where(self.h_nodes_u == source_vertex_int)[0]
        
        if len(source_indices) == 0:
            return {}, {}
            
        # Set start times and status on GPU
        d_start_times[source_indices] = cp.asarray(self.h_nodes_t[source_indices])
        d_status[source_indices] = 1
        
        # Insert source nodes into their respective level worklists
        # We do this on CPU for initialization simplicity, or via a simple kernel.
        # Given small number of source edges, CPU->GPU copy is fine.
        h_wlist_counters = np.zeros(self.max_level + 2, dtype=np.int32)
        
        # We need to write these source nodes into d_wlist_storage at correct offsets
        # This part requires a bit of care to batch writes or map them.
        # Simple approach: Group sources by level and copy.
        sources_by_level = {}
        for idx in source_indices:
            lvl = self.h_node_levels[idx]
            if lvl not in sources_by_level:
                sources_by_level[lvl] = []
            sources_by_level[lvl].append(idx)
            
        # Copy to GPU worklists
        for lvl, indices in sources_by_level.items():
            count = len(indices)
            offset = int(self.d_wlist_offsets[lvl])
            # Write indices to storage
            self.d_wlist_storage[offset:offset+count] = cp.asarray(indices, dtype=cp.int32)
            # Update counter
            h_wlist_counters[lvl] = count

        # Copy initial counters to GPU
        d_wlist_counters = cp.asarray(h_wlist_counters, dtype=cp.int32)

        # --- 3. Process Levels (Algorithm 2 Line 5) ---
        threads = 256
        
        # We must copy the counters back to host at each step to know launch dims?
        # Optimization: We can keep counters on GPU, but we need the count on CPU to launch grid.
        # Since number of levels is small relative to nodes, calculating it is okay.
        
        for l in range(1, self.max_level + 1):
            # Get number of active nodes in this level
            # We copy just one integer from GPU to CPU
            num_active = int(d_wlist_counters[l])
            
            if num_active == 0:
                continue
            
            blocks = (num_active + threads - 1) // threads
            
            # Pointer to the start of this level's worklist in global storage
            current_wlist_ptr = self.d_wlist_storage[int(self.d_wlist_offsets[l]):]
            
            LOCAL_WORKLIST_KERNEL(
                (blocks,), (threads,),
                (
                    num_active,
                    current_wlist_ptr,   # Start of current worklist
                    self.d_nodes_v,
                    self.d_nodes_a,
                    self.d_adj_indptr,
                    self.d_adj_indices,
                    self.d_node_levels,  # Needed to find level of neighbors
                    d_start_times,
                    d_status,
                    self.d_wlist_storage,
                    self.d_wlist_offsets,
                    d_wlist_counters,    # Atomic updates for NEXT levels
                    d_packed_results
                )
            )

        # --- 4. Unpack Results ---
        h_packed = cp.asnumpy(d_packed_results)
        times = h_packed >> 32
        
        result_times = {}
        for v in range(len(times)):
            t = int(times[v])
            if t < INF_TIME:
                result_times[str(v)] = t

        fastest_paths = {}

        t_end = time.perf_counter()
        logging.info(f"Algorithm 2 (LW) Complete: {t_end - t_start:.4f}s")
        
        return result_times, fastest_paths