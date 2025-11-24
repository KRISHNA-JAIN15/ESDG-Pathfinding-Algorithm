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


    # fpd_solver = SerialESDG_FPD(esd_graph)
    fpd_solver = ParallelESDG_FPD(esd_graph)

    journey_times, fastest_paths = fpd_solver.find_fastest_paths(source_node)

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
    NUM_ROWS_TO_PROCESS = 10000
    run_pipeline(DATASET_FILE, SOURCE_VERTEX, num_rows=NUM_ROWS_TO_PROCESS)