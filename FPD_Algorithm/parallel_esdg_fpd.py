import cupy as cp
import numpy as np
import logging
import time
from ESD_Graph.structures.esd_graph import ESD_graph

# --- CUDA Kernel Definition ---
# This kernel implements Algorithm 3 (Level-Order Traversal) from Paper 1.
# It combines Distance Relaxation and Graph Traversal in one pass.
LEVEL_PROCESS_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void level_process_kernel(
    const int level_start_idx,      // Start index of nodes in the current level
    const int level_end_idx,        // End index (exclusive)
    const int* nodes_v,             // Destination vertex of the trip (node)
    const int* nodes_a,             // Arrival time of the trip
    const int* adj_indptr,          // CSR pointer for neighbors
    const int* adj_indices,         // CSR indices for neighbors
    int* start_time,                // Array storing max start time for each node
    int* journey_times,             // Array storing min journey time for each vertex
    bool* status                    // Array indicating if a node is active
) {
    // 1. Thread Mapping
    // Map thread to a specific node in the sorted node array
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int node_idx = level_start_idx + idx;

    // Boundary check
    if (node_idx >= level_end_idx) return;

    // 2. Status Check (Pruning)
    // If this node wasn't activated by a parent, we skip it.
    if (!status[node_idx]) return;

    // Load data into registers (Phase 2: Memory Optimization)
    int my_start_time = start_time[node_idx];
    int dest_vertex = nodes_v[node_idx];
    int arrival = nodes_a[node_idx];

    // 3. Distance Relaxation (Vertex Processing)
    // Journey Time = Arrival Time - Departure Time (at source)
    // We use atomicMin to safely update the global journey time for the destination vertex.
    int duration = arrival - my_start_time;
    atomicMin(&journey_times[dest_vertex], duration);

    // 4. Graph Traversal (Level-wise BFS)
    // Iterate over outgoing neighbors in the ESDG
    int start_edge = adj_indptr[node_idx];
    int end_edge = adj_indptr[node_idx + 1];

    for (int i = start_edge; i < end_edge; ++i) {
        int neighbor_idx = adj_indices[i];
        
        // Propagate the start time to the neighbor. 
        // We want the LATEST departure from source that can reach here.
        atomicMax(&start_time[neighbor_idx], my_start_time);
        
        // Activate the neighbor for the next level's processing
        // Note: Paper 1 [cite: 379] states race conditions on status flag are benign.
        status[neighbor_idx] = true; 
    }
}
''', 'level_process_kernel')

class ParallelESDG_FPD:
    """
    GPU-accelerated implementation of FPD using CuPy.
    Implements Phase 1 (Kernel Dev) and Phase 2 (Memory Layout) of the user plan.
    """

    def __init__(self, esd_graph: ESD_graph):
        self.original_graph = esd_graph
        self.num_nodes = len(esd_graph.nodes)
        
        # Determine max vertex ID for sizing the journey_times array
        self.max_vertex_id = 0
        if self.num_nodes > 0:
            max_u = max(n.u for n in esd_graph.nodes.values())
            max_v = max(n.v for n in esd_graph.nodes.values())
            self.max_vertex_id = max(max_u, max_v)

        logging.info("Initializing Parallel FPD Solver...")
        self._prepare_gpu_data()

    def _prepare_gpu_data(self):
        """
        Phase 1 & 2: Memory Management & Coalescing.
        Flattens the graph into SoA (Structure of Arrays) and sorts by level
        to ensure memory coalescing during kernel execution.
        """
        t0 = time.perf_counter()
        
        # 1. Sort nodes by Level (Crucial for Level-Order Algorithm)
        # We need a mapping from original_id -> gpu_index (0 to N-1)
        sorted_node_ids = sorted(
            self.original_graph.nodes.keys(),
            key=lambda nid: (self.original_graph.levels.get(nid, 0), nid)
        )
        
        self.id_map = {original: i for i, original in enumerate(sorted_node_ids)}
        
        # 2. Build Structure of Arrays (SoA) for Nodes
        # We only need 'v' and 'a' on GPU for the forward pass. 
        # 'u' and 't' are only needed for source initialization on CPU.
        nodes_v = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_a = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_u_cpu = np.zeros(self.num_nodes, dtype=np.int32) # Keep on CPU for init
        nodes_t_cpu = np.zeros(self.num_nodes, dtype=np.int32) # Keep on CPU for init
        
        # Track levels for batching
        self.level_offsets = []
        current_level = -1
        
        for i, original_id in enumerate(sorted_node_ids):
            node = self.original_graph.nodes[original_id]
            level = self.original_graph.levels.get(original_id, 1)
            
            nodes_v[i] = int(node.v)
            nodes_a[i] = int(node.a)
            nodes_u_cpu[i] = int(node.u)
            nodes_t_cpu[i] = int(node.t)
            
            # Record start index of new levels
            if level != current_level:
                self.level_offsets.append(i)
                current_level = level
        
        self.level_offsets.append(self.num_nodes) # Sentinel for the end
        
        # 3. Build CSR (Compressed Sparse Row) for Adjacency
        # This is optimized for the 'Graph Traversal' kernel phase
        adj_indices_list = []
        adj_indptr = [0]
        
        for original_id in sorted_node_ids:
            # Get neighbors (original IDs)
            neighbors = self.original_graph.adj.get(original_id, [])
            # Convert to GPU indices
            mapped_neighbors = [self.id_map[n] for n in neighbors if n in self.id_map]
            
            adj_indices_list.extend(mapped_neighbors)
            adj_indptr.append(len(adj_indices_list))
            
        # 4. Transfer to GPU (CuPy Arrays)
        self.d_nodes_v = cp.array(nodes_v)
        self.d_nodes_a = cp.array(nodes_a)
        self.d_adj_indices = cp.array(adj_indices_list, dtype=np.int32)
        self.d_adj_indptr = cp.array(adj_indptr, dtype=np.int32)
        
        # Keep these on CPU for source initialization logic
        self.h_nodes_u = nodes_u_cpu
        self.h_nodes_t = nodes_t_cpu
        
        t1 = time.perf_counter()
        logging.info(f"GPU Data Preparation complete in {t1-t0:.4f}s")

    def find_fastest_paths(self, source_vertex_s: str):
        """
        Executes the Level-Order Parallel FPD Algorithm.
        """
        source_vertex_int = int(source_vertex_s)
        t_start = time.perf_counter()
        
        # --- 1. Initialization (Phase 1: Memory Management) ---
        # Initialize Status and Start Time arrays
        # start_time array: initialized to -1 (inactive)
        d_start_time = cp.full(self.num_nodes, -1, dtype=cp.int32)
        
        # status array: initialized to False
        d_status = cp.full(self.num_nodes, False, dtype=bool)
        
        # journey_times: initialized to MAX_INT
        # Note: Size is based on Vertex IDs, not Node IDs
        d_journey_times = cp.full(self.max_vertex_id + 1, 2147483647, dtype=cp.int32)
        
        # Set journey time for source to 0
        d_journey_times[source_vertex_int] = 0

        # --- 2. Source Node Activation (CPU -> GPU) ---
        # Find all ESDG nodes corresponding to the source vertex
        # This corresponds to Line 3 of Algorithm 3 in Paper 1 
        source_indices = np.where(self.h_nodes_u == source_vertex_int)[0]
        
        if len(source_indices) == 0:
            return {}, {} # No paths

        # Update GPU arrays for source nodes
        # We perform a batched scatter update
        d_start_time.scatter_update(source_indices, self.h_nodes_t[source_indices])
        d_status.scatter_update(source_indices, True)

        # --- 3. Level-Wise Execution (Phase 1: Graph Traversal) ---
        # We iterate through levels defined by level_offsets
        num_levels = len(self.level_offsets) - 1
        
        threads_per_block = 256 # Occupancy Tuning (Phase 2)

        for l in range(num_levels):
            start_idx = self.level_offsets[l]
            end_idx = self.level_offsets[l+1]
            num_nodes_in_level = end_idx - start_idx
            
            if num_nodes_in_level == 0: 
                continue

            blocks_per_grid = (num_nodes_in_level + threads_per_block - 1) // threads_per_block

            # Launch Kernel
            LEVEL_PROCESS_KERNEL(
                (blocks_per_grid,), (threads_per_block,),
                (
                    start_idx,
                    end_idx,
                    self.d_nodes_v,
                    self.d_nodes_a,
                    self.d_adj_indptr,
                    self.d_adj_indices,
                    d_start_time,
                    d_journey_times,
                    d_status
                )
            )

        # --- 4. Retrieve Results (Device -> Host) ---
        h_journey_times = cp.asnumpy(d_journey_times)
        
        # Format results to match Serial API
        result_times = {}
        for v in range(len(h_journey_times)):
            t = h_journey_times[v]
            if t < 2147483647:
                result_times[str(v)] = int(t)
            else:
                result_times[str(v)] = float('inf')

        t_end = time.perf_counter()
        logging.info(f"GPU FPD finished in {t_end - t_start:.4f}s")
        
        # Note: Path reconstruction is complex on GPU and typically done on CPU 
        # via a predecessor map if strict path details are needed. 
        # For pure performance benchmarking (FPD), we return times.
        return result_times, {}