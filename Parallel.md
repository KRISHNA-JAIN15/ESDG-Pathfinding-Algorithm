## Phase 1 (Kernel Development):

Distance Relaxation: Handled by atomicMin inside the kernel.

Graph Traversal: Handled by the loop over adj_indices (CSR format) inside the kernel.

Memory Management: The _prepare_gpu_data method flattens the graph object into contiguous arrays sorted by level.

## Phase 2 (Performance Optimization):

Memory Coalescing: By sorting nodes by level, threads i and i+1 access nodes_v[i] and nodes_v[i+1] which are adjacent in memory, ensuring perfect coalescing.

Occupancy Tuning: We use a standard block size of 256.

Fine-Grained Parallelism: Each thread processes exactly one node/edge-set, matching the "fine-grained parallelism" requirement.

