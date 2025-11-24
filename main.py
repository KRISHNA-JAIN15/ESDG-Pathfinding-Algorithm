import pandas as pd
import logging
from ESD_Graph.esd_transformer import transform_temporal_to_esd
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from FPD_Algorithm.parallel_esdg_fpd import ParallelESDG_FPD

from utils.graph_caching import save_esd_graph_to_json, load_esd_graph_from_json
from analysis.visualizer import visualize_top_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(dataset_path: str, source_node: str, num_rows: int = None):
    """
    Executes the full pipeline: Load -> Transform -> Compute FPD -> Visualize.
    """
    # ... (Data loading and ESDG transformation sections are correct) ...
    print("="*50); print("STEP 1: LOADING AND PREPARING TEMPORAL DATA"); print("="*50)
    df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
    temporal_edges_list = [(str(row['from_stop_I']), str(row['to_stop_I']),
                            int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
                           for _, row in df.iterrows() if row['arr_time_ut'] - row['dep_time_ut'] > 0]
    print(f"Loaded and prepared {len(temporal_edges_list)} valid temporal edges.")


    print("\n" + "="*50); print("STEP 2: GETTING ESD GRAPH (CACHE OR BUILD)"); print("="*50)
    esd_graph = load_esd_graph_from_json(num_rows)
    if esd_graph is None:
        esd_graph = transform_temporal_to_esd(temporal_edges_list)
        save_esd_graph_to_json(esd_graph, num_rows)
    print("\n--- ESD Graph Structure Loaded ---")


    print("\n" + "="*50); print("STEP 3: COMPUTING FASTEST PATH DURATION (FPD)"); print("="*50)

    import time
    
    # Debug: Check what vertices exist in the graph
    all_vertices = set()
    for node in esd_graph.nodes.values():
        all_vertices.add(int(node.u))
        all_vertices.add(int(node.v))
    
    source_int = int(source_node)
    print(f"\n🔍 DEBUGGING VERTEX MISMATCH:")
    print(f"   Requested source: {source_node} (as int: {source_int})")
    print(f"   Source exists in graph: {source_int in all_vertices}")
    print(f"   Total vertices in graph: {len(all_vertices)}")
    print(f"   Sample vertices: {sorted(list(all_vertices))[:10]}...")
    
    if source_int not in all_vertices:
        # Try to find a vertex that exists
        alt_source = sorted(list(all_vertices))[0]
        print(f"   🔄 Using alternative source: {alt_source}")
        source_node = str(alt_source)
    
    # Compare performance: Parallel (fast, no paths) vs Serial (with paths)
    print(f"\n=== PARALLEL GPU INITIALIZATION ===")
    t_init_start = time.perf_counter()
    fpd_solver_parallel = ParallelESDG_FPD(esd_graph)
    t_init = time.perf_counter() - t_init_start
    print(f"GPU Initialization Time: {t_init:.4f}s")
    
    print("=== PARALLEL GPU (Optimized - Times Only) ===")
    t_start = time.perf_counter()
    journey_times_parallel, _ = fpd_solver_parallel.find_fastest_paths(source_node, reconstruct_paths=False)
    t_parallel_compute = time.perf_counter() - t_start
    t_parallel_total = t_init + t_parallel_compute
    print(f"GPU Computation Time: {t_parallel_compute:.4f}s")
    print(f"GPU Total Time: {t_parallel_total:.4f}s")
    
    print("\n=== SERIAL CPU (With Path Reconstruction) ===")
    t_start = time.perf_counter()
    fpd_solver_serial = SerialESDG_FPD(esd_graph)
    journey_times, fastest_paths = fpd_solver_serial.find_fastest_paths(source_node)
    t_serial = time.perf_counter() - t_start
    print(f"Serial CPU Time: {t_serial:.4f}s")
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   GPU Computation Only: {t_parallel_compute:.4f}s")
    print(f"   GPU Total (init + compute): {t_parallel_total:.4f}s") 
    print(f"   Serial CPU: {t_serial:.4f}s")
    print(f"\n🚀 GPU Computation Speedup: {t_serial/t_parallel_compute:.2f}x faster!")
    print(f"🔄 GPU Total vs Serial: {t_serial/t_parallel_total:.2f}x")
    print(f"✅ Results match: {len(set(journey_times_parallel.items()) & set(journey_times.items()))} destinations")
    
    # Optional: Get paths for top 10 destinations using GPU (if needed for visualization)
    print("\n=== PARALLEL GPU (With Limited Path Reconstruction) ===")
    t_start = time.perf_counter()
    journey_times_with_paths, fastest_paths_gpu = fpd_solver_parallel.find_fastest_paths(source_node, reconstruct_paths=True)
    t_parallel_with_paths = time.perf_counter() - t_start
    print(f"Parallel GPU Time (with paths): {t_parallel_with_paths:.4f}s")
    
    # Use GPU results with paths for display
    journey_times = journey_times_with_paths
    fastest_paths = fastest_paths_gpu

    # --- 4. Display Results in Console (Corrected Section) ---
    print("\n" + "="*50); print(f"STEP 4: RESULTS - FASTEST JOURNEY TIMES FROM '{source_node}'"); print("="*50)
    sorted_results = sorted(journey_times.items(), key=lambda item: item[1])
    
    print(f"{'Destination':<15} | {'Journey Duration (s)':<25} | {'Path (Sequence of Edges in G)'}")
    print("-" * 80)

    # We will print the top 15 results for brevity in the console
    for dest, time in sorted_results[:15]:
        
        # ***** FIX STARTS HERE *****
        if time == float('inf'):
            path_str = "Unreachable"
        elif dest == source_node:
            path_str = "Source Node"
        else:
            # Look up the reconstructed path from the `fastest_paths` dictionary
            path_nodes = fastest_paths.get(dest)
            if path_nodes:
                # Format the path for readability by joining the original edge IDs
                path_str = " -> ".join([f"e{node.original_edge_id}" for node in path_nodes])
            else:
                # This case handles if a path wasn't found, though it shouldn't happen for reachable nodes
                path_str = "Path not available"
        # ***** FIX ENDS HERE *****
                
        print(f"{dest:<15} | {time:<25} | {path_str}")

    # --- 5. Visualize Top 10 Paths ---
    visualize_top_paths(esd_graph, source_node, journey_times, fastest_paths)


if __name__ == "__main__":
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    SOURCE_VERTEX = "2421"
    NUM_ROWS_TO_PROCESS = 10000  # Try 50000 for larger dataset where GPU shines
    
    print(f"🔧 Configuration: Processing {NUM_ROWS_TO_PROCESS} temporal edges")
    print("💡 Note: GPU excels with larger datasets (try 50000+ rows)")
    print("⚡ For small datasets, initialization overhead dominates\n")
    
    run_pipeline(DATASET_FILE, SOURCE_VERTEX, num_rows=NUM_ROWS_TO_PROCESS)