import pandas as pd
from ESD_Graph.esd_transformer import transform_temporal_to_esd
# Corrected the import to use the enhanced algorithm file
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD 
import logging

# It's good practice to configure logging in your main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(dataset_path: str, source_node: str, num_rows: int = None):
    """
    Executes the full pipeline: Load -> Transform -> Compute FPD.
    """
    # --- 1. Load and Prepare Data ---
    print("="*50)
    print("STEP 1: LOADING AND PREPARING TEMPORAL DATA")
    print("="*50)
    # ... (the rest of your loading code is perfect) ...
    try:
        df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loaded {len(df)} records from {dataset_path}.")

    temporal_edges_list = []
    for _, row in df.iterrows():
        duration = row['arr_time_ut'] - row['dep_time_ut']
        if duration > 0:
            temporal_edges_list.append((
                str(row['from_stop_I']),
                str(row['to_stop_I']),
                int(row['dep_time_ut']),
                int(duration)
            ))
    print(f"Prepared {len(temporal_edges_list)} valid temporal edges.")

    # --- 2. Transform Temporal Graph to ESD Graph ---
    print("\n" + "="*50)
    print("STEP 2: TRANSFORMING TO EDGE SCAN DEPENDENCY (ESD) GRAPH")
    print("="*50)
    esd_graph = transform_temporal_to_esd(temporal_edges_list)
    print("\n--- ESD Graph Structure ---")
    print(esd_graph)
    print("---------------------------\n")

    # --- 3. Run the FPD Algorithm ---
    print("\n" + "="*50)
    print("STEP 3: COMPUTING FASTEST PATH DURATION (FPD)")
    print("="*50)
    fpd_solver = SerialESDG_FPD(esd_graph)
    
    # ***** FIX 1: Unpack the tuple returned by the function *****
    # The function returns (journey_times, fastest_paths), so we need two variables
    journey_times, fastest_paths = fpd_solver.find_fastest_paths(source_node)

    # --- 4. Display Results ---
    print("\n" + "="*50)
    print(f"STEP 4: RESULTS - FASTEST JOURNEY TIMES FROM '{source_node}'")
    print("="*50)
    
    # ***** FIX 2: Sort the correct dictionary *****
    # We want to sort by the journey duration, which is in the `journey_times` dictionary
    sorted_results = sorted(journey_times.items(), key=lambda item: item[1])
    
    print(f"{'Destination':<15} | {'Journey Duration (s)':<25} | {'Path (ESDG Nodes)'}")
    print("-" * 70)
    
    for dest, time in sorted_results:
        if time == float('inf'):
            path_str = "Unreachable"
        elif dest == source_node:
            path_str = "Source Node"
        else:
            path_nodes = fastest_paths.get(dest)
            if path_nodes:
                path_str = "-".join([f"e{node.original_edge_id}" for node in path_nodes])
            else:
                path_str = "N/A"
                
        print(f"{dest:<15} | {time:<25} | {path_str}")


if __name__ == "__main__":
    # --- Configuration ---
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    SOURCE_VERTEX = "2421"
    NUM_ROWS_TO_PROCESS = 100  # Set to None to process the entire dataset 

    # --- Run the pipeline ---
    run_pipeline(DATASET_FILE, SOURCE_VERTEX, num_rows=NUM_ROWS_TO_PROCESS)