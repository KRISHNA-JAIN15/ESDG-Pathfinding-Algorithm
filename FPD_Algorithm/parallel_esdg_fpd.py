import cupy as cp
import numpy as np
import logging
import time
from ESD_Graph.structures.esd_graph import ESD_graph

# --- CUDA Kernel Definition ---
# This kernel implements Algorithm 1 (Multiple Breadth-First Search) from the paper.
# Processes one BFS frontier in parallel for each source node phase.
BFS_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void bfs_kernel(
    const int frontier_size,        // Number of nodes in current frontier
    const int* frontier_nodes,      // Current frontier node indices
    const int* nodes_v,             // Destination vertex of each node
    const int* nodes_a,             // Arrival time of each node
    const int source_departure,     // Departure time of source node for this phase
    const int* adj_indptr,          // CSR adjacency pointers
    const int* adj_indices,         // CSR adjacency indices
    int* journey_times,             // Global journey times array
    int* status,                    // Node visited status (0=false, 1=true)
    int* next_frontier,             // Next frontier markers (0=false, 1=true)
    int* predecessors               // For path reconstruction
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx >= frontier_size) return;
    
    int node_idx = frontier_nodes[idx];
    int dest_vertex = nodes_v[node_idx];
    int arrival_time = nodes_a[node_idx];
    
    // Compute journey time: arrival - source_departure
    int journey_time = arrival_time - source_departure;
    
    // Update minimum journey time atomically (Line 7 in Algorithm 1)
    int old_time = atomicMin(&journey_times[dest_vertex], journey_time);
    if (journey_time < old_time) {
        predecessors[dest_vertex] = node_idx;
    }
    
    // Explore neighbors (Lines 8-9 in Algorithm 1)
    int start_edge = adj_indptr[node_idx];
    int end_edge = adj_indptr[node_idx + 1];
    
    for (int i = start_edge; i < end_edge; ++i) {
        int neighbor_idx = adj_indices[i];
        
        // Atomic compare-and-swap to mark unvisited neighbors (Line 9)
        // Use atomicCAS with integers (0=false, 1=true)
        if (atomicCAS(&status[neighbor_idx], 0, 1) == 0) {
            next_frontier[neighbor_idx] = 1;  // Mark for next frontier
        }
    }
}
''', 'bfs_kernel')

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
            max_u = max(int(n.u) for n in esd_graph.nodes.values())
            max_v = max(int(n.v) for n in esd_graph.nodes.values())
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
        
        # 1. Create node mapping (no need to sort by levels for Algorithm 1)
        # We need a mapping from original_id -> gpu_index (0 to N-1)
        sorted_node_ids = list(self.original_graph.nodes.keys())
        
        self.id_map = {original: i for i, original in enumerate(sorted_node_ids)}
        
        # 2. Build Structure of Arrays (SoA) for Nodes
        # We only need 'v' and 'a' on GPU for the forward pass. 
        # 'u' and 't' are only needed for source initialization on CPU.
        nodes_v = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_a = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_u_cpu = np.zeros(self.num_nodes, dtype=np.int32) # Keep on CPU for init
        nodes_t_cpu = np.zeros(self.num_nodes, dtype=np.int32) # Keep on CPU for init
        
        # Build node arrays without level tracking
        for i, original_id in enumerate(sorted_node_ids):
            node = self.original_graph.nodes[original_id]
            
            nodes_v[i] = int(node.v)
            nodes_a[i] = int(node.a)
            nodes_u_cpu[i] = int(node.u)
            nodes_t_cpu[i] = int(node.t)
        
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
            
        # 4. Transfer to GPU (CuPy Arrays) - Use asarray for better performance
        self.d_nodes_v = cp.asarray(nodes_v)
        self.d_nodes_a = cp.asarray(nodes_a)
        self.d_adj_indices = cp.asarray(adj_indices_list, dtype=np.int32)
        self.d_adj_indptr = cp.asarray(adj_indptr, dtype=np.int32)
        
        # Keep these on CPU for source initialization logic
        self.h_nodes_u = nodes_u_cpu
        self.h_nodes_t = nodes_t_cpu
        
        t1 = time.perf_counter()
        logging.info(f"GPU Data Preparation complete in {t1-t0:.4f}s")
    
    def _reconstruct_paths_cpu_optimized(self, predecessors, source_vertex_int, result_times):
        """
        Optimized path reconstruction - only for top N destinations to avoid expensive computation.
        """
        fastest_paths = {}
        
        # Create reverse mapping once (more efficient than searching every time)
        idx_to_original = {mapped_idx: orig_id for orig_id, mapped_idx in self.id_map.items()}
        
        # Only reconstruct paths for top 20 destinations to save time
        sorted_dests = sorted(
            [(dest, time) for dest, time in result_times.items() if time != float('inf')],
            key=lambda x: x[1]
        )[:20]
        
        for dest_vertex_str, journey_time in sorted_dests:
            dest_vertex_int = int(dest_vertex_str)
            
            # Skip if destination is source
            if dest_vertex_int == source_vertex_int:
                continue
                
            # Quick path reconstruction with bounds checking
            path = []
            current_vertex = dest_vertex_int
            max_hops = 10  # Limit path length to avoid infinite loops
            
            while (max_hops > 0 and current_vertex < len(predecessors) and 
                   predecessors[current_vertex] != -1):
                pred_node_idx = predecessors[current_vertex]
                
                if pred_node_idx in idx_to_original:
                    original_node_id = idx_to_original[pred_node_idx]
                    path.append(self.original_graph.nodes[original_node_id])
                    current_vertex = int(self.original_graph.nodes[original_node_id].u)
                    max_hops -= 1
                else:
                    break
            
            if path:
                fastest_paths[dest_vertex_str] = path[::-1]  # Reverse to get source -> dest
        
        return fastest_paths

    def find_fastest_paths(self, source_vertex_s: str, reconstruct_paths: bool = False):
        """
        Executes Algorithm 1: Multiple Breadth-First Search from the paper.
        
        Args:
            source_vertex_s: Source vertex as string
            reconstruct_paths: If True, reconstruct full paths (expensive). If False, only compute times (fast).
        """
        source_vertex_int = int(source_vertex_s)
        t_start = time.perf_counter()
        
        # --- 1. Initialization ---
        # Initialize status array to 0 (false) for each node (Line 1 in Algorithm 1)
        d_status = cp.full(self.num_nodes, 0, dtype=cp.int32)
        
        # Initialize journey times to infinity
        d_journey_times = cp.full(self.max_vertex_id + 1, 2147483647, dtype=cp.int32)
        d_journey_times[source_vertex_int] = 0
        
        # Initialize predecessors for path reconstruction
        d_predecessors = cp.full(self.max_vertex_id + 1, -1, dtype=cp.int32)
        
        # --- 2. Find Source Nodes and Sort by Departure Time (Line 2) ---
        # Find all ESDG nodes corresponding to the source vertex
        source_node_indices = np.where(self.h_nodes_u == source_vertex_int)[0]
        
        if len(source_node_indices) == 0:
            logging.warning(f"No source nodes found for vertex {source_vertex_int}")
            return {}, {}
            
        # Sort source nodes by departure time in DECREASING order (critical for correctness)
        source_departures = self.h_nodes_t[source_node_indices]
        sorted_order = np.argsort(-source_departures)  # Negative for descending order
        sorted_source_indices = source_node_indices[sorted_order]
        sorted_departures = source_departures[sorted_order]
        
        logging.info(f"Found {len(sorted_source_indices)} source nodes, processing in decreasing departure order")
        if len(sorted_source_indices) > 0:
            logging.info(f"Source departure times: {sorted_departures[:5]}...")  # Show first 5
        else:
            logging.error(f"No source nodes found for vertex {source_vertex_int}. Available vertices: {np.unique(self.h_nodes_u)[:10]}...")
        
        # --- 3. Multiple BFS Phases (Algorithm 1 Main Loop) ---
        threads_per_block = 256
        
        for phase, source_idx in enumerate(sorted_source_indices):
            source_departure = sorted_departures[phase]
            
            # Skip if this source node was already visited in a previous phase
            if cp.asnumpy(d_status[source_idx]) != 0:
                continue
                
            # Initialize frontier with current source node
            current_frontier = cp.array([source_idx], dtype=np.int32)
            d_status[source_idx] = 1  # Mark as visited
            
            # BFS from this source node
            while len(current_frontier) > 0:
                frontier_size = len(current_frontier)
                blocks_per_grid = (frontier_size + threads_per_block - 1) // threads_per_block
                
                # Prepare next frontier markers
                d_next_frontier_markers = cp.full(self.num_nodes, 0, dtype=cp.int32)
                
                # Launch BFS kernel for current frontier
                BFS_KERNEL(
                    (blocks_per_grid,), (threads_per_block,),
                    (
                        frontier_size,
                        current_frontier,
                        self.d_nodes_v,
                        self.d_nodes_a,
                        source_departure,
                        self.d_adj_indptr,
                        self.d_adj_indices,
                        d_journey_times,
                        d_status,
                        d_next_frontier_markers,
                        d_predecessors
                    )
                )
                
                # Build next frontier from markers
                next_frontier_indices = cp.where(d_next_frontier_markers)[0]
                current_frontier = next_frontier_indices

        # --- 4. Retrieve Results (Device -> Host) ---
        h_journey_times = cp.asnumpy(d_journey_times)
        h_predecessors = cp.asnumpy(d_predecessors)
        
        # Format results to match Serial API
        result_times = {}
        for v in range(len(h_journey_times)):
            t = h_journey_times[v]
            if t < 2147483647:
                result_times[str(v)] = int(t)
            else:
                result_times[str(v)] = float('inf')

        # --- 5. Optionally Reconstruct Paths on CPU ---
        if reconstruct_paths:
            h_predecessors = cp.asnumpy(d_predecessors)
            fastest_paths = self._reconstruct_paths_cpu_optimized(h_predecessors, source_vertex_int, result_times)
        else:
            fastest_paths = {}  # Skip expensive path reconstruction for pure performance

        t_end = time.perf_counter()
        logging.info(f"GPU FPD finished in {t_end - t_start:.4f}s")
        
        return result_times, fastest_paths