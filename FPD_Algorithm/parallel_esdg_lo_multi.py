import cupy as cp
import numpy as np
import logging
import time
from collections import deque

# --- Improved Weighted Level Order Kernel with Conflict Tracking ---
WEIGHTED_LO_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void weighted_level_order_kernel(
    const int level_start_idx,
    const int level_end_idx,
    const int* nodes_v,
    const int* weights,
    const int* adj_indptr,
    const int* adj_indices,
    unsigned long long* node_scores,     // [IN/OUT] Packed(Cost, ParentESDGNode)
    unsigned long long* vertex_results,  // [OUT] Packed(Cost, FinalESDGNode)
    int* status,
    int* conflicts                       // [OUT] Conflict/Update Counter
) {
    int idx = level_start_idx + blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx >= level_end_idx) return;
    
    // Load score once
    unsigned long long my_packed = node_scores[idx];
    unsigned int my_cost = my_packed >> 32;
    
    // Optimization: Only proceed if this node has been reached (Cost < Infinity)
    if (my_cost != 0xFFFFFFFF) {
        
        int dest_vertex = nodes_v[idx];
        
        // 1. Update Best Path to Physical Vertex
        unsigned long long result_val = ((unsigned long long)my_cost << 32) | idx;
        atomicMin(&vertex_results[dest_vertex], result_val);
        
        // 2. Propagate to Neighbors
        int start_edge = adj_indptr[idx];
        int end_edge = adj_indptr[idx + 1];
        
        for (int i = start_edge; i < end_edge; ++i) {
            int neighbor_idx = adj_indices[i];
            int w = weights[neighbor_idx];
            
            // Calculate new cost
            unsigned int new_cost = my_cost + w;
            // Overflow protection
            if (new_cost < my_cost) new_cost = 0xFFFFFFFE;
            
            unsigned long long neighbor_new_val = ((unsigned long long)new_cost << 32) | idx;
            
            // Atomic Min returns the OLD value
            unsigned long long old_val = atomicMin(&node_scores[neighbor_idx], neighbor_new_val);
            
            // Check if we actually improved the path (Conflict/Update detection)
            unsigned int old_cost = old_val >> 32;
            if (new_cost < old_cost) {
                // We successfully updated the path -> This is a "conflict" won by this thread
                atomicAdd(&conflicts[neighbor_idx], 1);
                
                // Activate for next pass
                status[neighbor_idx] = 1; 
            }
        }
    }
}
''', 'weighted_level_order_kernel')

class ParallelWeightedLO:
    def __init__(self, esd_graph):
        self.original_graph = esd_graph
        self.num_nodes = len(esd_graph.nodes)
        
        self.max_vertex_id = 0
        if self.num_nodes > 0:
            max_u = max(int(n.u) for n in esd_graph.nodes.values())
            max_v = max(int(n.v) for n in esd_graph.nodes.values())
            self.max_vertex_id = max(max_u, max_v)

        self._prepare_gpu_data()

    def _compute_levels(self):
        in_degree = {i: 0 for i in self.original_graph.nodes}
        for u in self.original_graph.nodes:
            for v in self.original_graph.adj.get(u, []):
                in_degree[v] = in_degree.get(v, 0) + 1
        
        queue = deque([u for u in self.original_graph.nodes if in_degree[u] == 0])
        levels = {u: 1 for u in self.original_graph.nodes}
        max_level = 0
        
        while queue:
            u = queue.popleft()
            current_lvl = levels[u]
            max_level = max(max_level, current_lvl)
            for v in self.original_graph.adj.get(u, []):
                if levels[v] < current_lvl + 1:
                    levels[v] = current_lvl + 1
                in_degree[v] -= 1
                if in_degree[v] == 0: queue.append(v)
        return levels, max_level

    def _prepare_gpu_data(self):
        t0 = time.perf_counter()
        
        # 1. Sort nodes by Level
        node_levels, self.max_level = self._compute_levels()
        sorted_ids = sorted(self.original_graph.nodes.keys(), key=lambda k: (node_levels[k], k))
        
        self.id_map = {orig: i for i, orig in enumerate(sorted_ids)}
        self.rev_map = {i: self.original_graph.nodes[orig] for i, orig in enumerate(sorted_ids)}

        # 2. Build Arrays
        nodes_v = np.zeros(self.num_nodes, dtype=np.int32)
        weights = np.zeros(self.num_nodes, dtype=np.int32)
        self.node_u_list = np.zeros(self.num_nodes, dtype=np.int32)
        
        for i, orig_id in enumerate(sorted_ids):
            node = self.original_graph.nodes[orig_id]
            nodes_v[i] = int(node.v)
            # Use 'weight' attribute, default to 1
            weights[i] = getattr(node, 'weight', 1) 
            self.node_u_list[i] = int(node.u)

        # 3. Build CSR
        adj_indices = []
        adj_indptr = [0]
        for orig_id in sorted_ids:
            neighbors = self.original_graph.adj.get(orig_id, [])
            mapped = [self.id_map[n] for n in neighbors if n in self.id_map]
            adj_indices.extend(mapped)
            adj_indptr.append(len(adj_indices))

        # 4. Offsets
        self.level_offsets = []
        cur_lvl = 0
        for i, orig_id in enumerate(sorted_ids):
            lvl = node_levels[orig_id]
            while cur_lvl < lvl:
                self.level_offsets.append(i)
                cur_lvl += 1
        self.level_offsets.append(self.num_nodes)

        # 5. GPU Upload
        self.d_nodes_v = cp.asarray(nodes_v)
        self.d_weights = cp.asarray(weights)
        self.d_adj_indices = cp.asarray(adj_indices, dtype=np.int32)
        self.d_adj_indptr = cp.asarray(adj_indptr, dtype=np.int32)
        
        t1 = time.perf_counter()
        logging.info(f"Weighted LO Prep Complete: {t1-t0:.4f}s")

    def find_weighted_paths(self, source_vertices: list, dest_vertices: list):
        """
        Calculates weighted paths with conflict tracking.
        Returns: (results_list, global_conflict_stats)
        """
        INF = 0xFFFFFFFF
        init_packed = (INF << 32) | INF
        
        d_node_scores = cp.full(self.num_nodes, init_packed, dtype=cp.uint64)
        d_vertex_results = cp.full(self.max_vertex_id + 1, init_packed, dtype=cp.uint64)
        d_status = cp.zeros(self.num_nodes, dtype=cp.int32)
        d_conflicts = cp.zeros(self.num_nodes, dtype=cp.int32) # Track number of updates per node

        # Initialize Sources
        unique_sources = list(set(source_vertices))
        for src in unique_sources:
            indices = np.where(self.node_u_list == src)[0]
            if len(indices) > 0:
                init_weights = self.d_weights[indices]
                init_vals = (init_weights.astype(cp.uint64) << 32) | INF
                d_node_scores[indices] = init_vals
                d_status[indices] = 1

        # Kernel Loop
        threads = 256
        for l in range(1, len(self.level_offsets)):
            start = self.level_offsets[l-1]
            end = self.level_offsets[l]
            count = end - start
            if count <= 0: continue
            
            blocks = (count + threads - 1) // threads
            WEIGHTED_LO_KERNEL((blocks,), (threads,), (
                int(start), int(end),
                self.d_nodes_v, self.d_weights,
                self.d_adj_indptr, self.d_adj_indices,
                d_node_scores, d_vertex_results, d_status, d_conflicts
            ))

        # Reconstruct
        h_vertex_res = cp.asnumpy(d_vertex_results)
        h_node_scores = cp.asnumpy(d_node_scores)
        h_conflicts = cp.asnumpy(d_conflicts)
        
        results = []
        
        for i in range(len(source_vertices)):
            req_src = source_vertices[i]
            req_dest = dest_vertices[i]
            
            packed = h_vertex_res[req_dest] if req_dest <= self.max_vertex_id else init_packed
            cost = packed >> 32
            
            if cost == INF:
                results.append({
                    'source': req_src, 'dest': req_dest, 
                    'cost': float('inf'), 'path': [], 'status': 'Unreachable',
                    'conflicts': 0
                })
                continue
            
            final_node_idx = packed & 0xFFFFFFFF
            path = []
            curr_idx = final_node_idx
            actual_source = -1
            path_conflicts = 0
            
            safe = 0
            # Trace back
            while curr_idx != INF and safe < 20000:
                if curr_idx not in self.rev_map: break
                node_obj = self.rev_map[curr_idx]
                path.append(node_obj)
                actual_source = int(node_obj.u)
                
                # Sum conflicts along this path
                path_conflicts += int(h_conflicts[curr_idx])
                
                parent_packed = h_node_scores[curr_idx]
                parent_idx = parent_packed & 0xFFFFFFFF
                
                if parent_idx == INF: break # Reached a source
                curr_idx = parent_idx
                safe += 1
            
            status_msg = 'Found' if actual_source == req_src else f"Reached by {actual_source}"
            results.append({
                'source': req_src, 'dest': req_dest, 
                'cost': cost, 'path': path[::-1], 'status': status_msg,
                'conflicts': path_conflicts
            })

        # Global Stats: Top 5 most updated/conflicted nodes
        top_indices = np.argsort(-h_conflicts)[:5]
        global_stats = []
        for idx in top_indices:
            count = int(h_conflicts[idx])
            if count > 0:
                node = self.rev_map[idx]
                # Fix: Handle both 'id' and 'original_edge_id' dynamically
                node_id = getattr(node, 'original_edge_id', getattr(node, 'id', 'Unknown'))
                global_stats.append(f"Edge {node_id} ({node.u}->{node.v}): {count} updates")

        return results, global_stats