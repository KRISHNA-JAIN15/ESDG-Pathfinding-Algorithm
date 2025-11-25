import pandas as pd
import logging
import time
import random
import numpy as np
from ESD_Graph.esd_transformer import transform_temporal_to_esd

# Import Solvers
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from FPD_Algorithm.parallel_esdg_fpd import ParallelESDG_FPD
from FPD_Algorithm.parallel_esdg_lo import ParallelESDG_LO
from FPD_Algorithm.parallel_esdg_lo_multi import ParallelWeightedLO

from utils.graph_caching import save_esd_graph_to_json, load_esd_graph_from_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(dataset_path: str, source_node: str, num_rows: int = None, custom_pairs: list = None):
    """
    Executes the full pipeline.
    """
    # --- STEP 1: LOAD DATA ---
    print("="*60); print("STEP 1: LOADING DATA"); print("="*60)
    df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
    temporal_edges_list = [(str(row['from_stop_I']), str(row['to_stop_I']),
                            int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
                           for _, row in df.iterrows() if row['arr_time_ut'] - row['dep_time_ut'] > 0]
    print(f"Loaded {len(temporal_edges_list)} temporal edges.")

    # --- STEP 2: BUILD ESD GRAPH ---
    print("\n" + "="*60); print("STEP 2: ESD GRAPH GENERATION"); print("="*60)
    esd_graph = load_esd_graph_from_json(num_rows)
    if esd_graph is None:
        esd_graph = transform_temporal_to_esd(temporal_edges_list)
        save_esd_graph_to_json(esd_graph, num_rows)
    
    # Vertex Validation
    all_vertices = set()
    for node in esd_graph.nodes.values():
        all_vertices.add(int(node.u)); all_vertices.add(int(node.v))
    
    if int(source_node) not in all_vertices:
        if len(all_vertices) > 0:
            alt_source = sorted(list(all_vertices))[0]
            print(f"⚠️ Source {source_node} not found. Switching to {alt_source}.")
            source_node = str(alt_source)
        else:
            print("❌ Graph is empty."); return

    # ---------------------------------------------------------
    # STEP 3: STANDARD ALGORITHMS
    # ---------------------------------------------------------
    print("\n" + "="*60); print(f"STEP 3: RUNNING STANDARD ALGORITHMS (Source: {source_node})"); print("="*60)
    
    t_start = time.perf_counter()
    solver_serial = SerialESDG_FPD(esd_graph)
    res_serial, _ = solver_serial.find_fastest_paths(source_node)
    t_serial = time.perf_counter() - t_start
    print(f"Serial CPU Time: {t_serial:.4f}s")

    solver_lo = ParallelESDG_LO(esd_graph)
    t_start = time.perf_counter()
    res_lo, _ = solver_lo.find_fastest_paths(source_node, reconstruct_paths=False)
    t_lo = time.perf_counter() - t_start
    print(f"Parallel LO Time: {t_lo:.4f}s | Speedup: {t_serial/t_lo:.2f}x")

    # ---------------------------------------------------------
    # STEP 4: WEIGHTED LO SOLVER (Custom Pairs)
    # ---------------------------------------------------------
    print("\n" + "="*60); print(f"STEP 4: RUNNING CUSTOM PAIR WEIGHTED LO"); print("="*60)
    
    if not custom_pairs:
        custom_pairs = [(int(source_node), int(list(all_vertices)[-1]))]
    
    sources = [p[0] for p in custom_pairs]
    dests = [p[1] for p in custom_pairs]
    
    print(f"🔧 Configuration:")
    print(f"   - {len(custom_pairs)} Pairs Detected")
    print("   - Mode: Consistent Weighting (Initial Cost = 1 per Edge)")
    
    # --- SET INITIAL WEIGHTS TO 1 ---
    # This ensures the cost metric is consistent (Hop Count)
    for node in esd_graph.nodes.values(): 
        node.weight = 1

    print(f"\n🚀 Processing {len(custom_pairs)} path requests...")
    
    t_start_w = time.perf_counter()
    solver_weighted = ParallelWeightedLO(esd_graph)
    # Now returns tuple: (results, stats)
    results_list, global_stats = solver_weighted.find_weighted_paths(sources, dests)
    t_end_w = time.perf_counter()
    
    print(f"⏱️ Compute Time: {t_end_w - t_start_w:.4f}s")
    
    # Display Results
    print("\n--- Path Results & Conflict Stats ---")
    print(f"{'Pair':<20} | {'Cost':<5} | {'Conflicts':<10} | {'Path Details'}")
    print("-" * 100)
    
    for res in results_list:
        s, d, c, p, conf = res['source'], res['dest'], res['cost'], res['path'], res['conflicts']
        pair_str = f"{s} -> {d}"
        
        if c == float('inf'):
            print(f"{pair_str:<20} | {'INF':<5} | {conf:<10} | ({res['status']})")
        else:
            # Use getattr with default for robust ID display
            path_str = " -> ".join([f"e{getattr(n, 'original_edge_id', getattr(n, 'id', '?'))}" for n in p])
            print(f"{pair_str:<20} | {c:<5} | {conf:<10} | {path_str}")

    print("\n--- Top Global Conflicts (Most Updated Nodes) ---")
    if global_stats:
        for stat in global_stats:
            print(f"   • {stat}")
    else:
        print("   • No major conflicts detected (Paths were optimal on first visit)")


if __name__ == "__main__":
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    NUM_ROWS = 100000 
    
    # --- USER CONFIGURATION ---
    # Define your specific pairs here.
    MY_PAIRS = [
        (3391, 4325),
        (3391, 4054),
        (1163, 7010) 
    ]
    
    PRIMARY_SOURCE = str(MY_PAIRS[0][0]) 
    
    run_pipeline(DATASET_FILE, PRIMARY_SOURCE, NUM_ROWS, custom_pairs=MY_PAIRS)