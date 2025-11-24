import time
import pandas as pd
import logging
import sys
import os

# Ensure we can import from local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ESD_Graph.esd_transformer import transform_temporal_to_esd
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from FPD_Algorithm.parallel_esdg_fpd import ParallelESDG_FPD
from utils.graph_caching import get_or_build_esd_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_benchmark():
    # --- CONFIGURATION ---
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    SOURCE_VERTEX = "2421"
    NUM_ROWS = 30000  # Adjust this: Higher numbers = more visible GPU improvement
    
    print(f"\n{'='*60}")
    print(f"  RUNNING SERIAL vs PARALLEL BENCHMARK (Rows: {NUM_ROWS})")
    print(f"{'='*60}")

    # --- 1. DATA PREPARATION ---
    # Helper to build graph if not in cache
    def builder_fn(nrows):
        logging.info("Reading CSV and building ESD Graph...")
        df = pd.read_csv(DATASET_FILE, sep=',', nrows=nrows)
        temporal_edges = [
            (str(row['from_stop_I']), str(row['to_stop_I']), int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
            for _, row in df.iterrows() if row['arr_time_ut'] - row['dep_time_ut'] > 0
        ]
        return transform_temporal_to_esd(temporal_edges), nrows

    # Load Graph (using the fix we discussed earlier)
    esd_graph = get_or_build_esd_graph(NUM_ROWS, builder_fn)
    print(f"Graph Loaded: {len(esd_graph.nodes)} nodes | {len(esd_graph.levels)} levels")

    # --- 2. SERIAL EXECUTION ---
    print(f"\n--- Running SERIAL Algorithm ---")
    serial_solver = SerialESDG_FPD(esd_graph)
    
    # Warmup (optional, for JIT stability)
    # serial_solver.find_fastest_paths(SOURCE_VERTEX) 
    
    t0 = time.perf_counter()
    serial_times, _ = serial_solver.find_fastest_paths(SOURCE_VERTEX)
    t1 = time.perf_counter()
    serial_duration = t1 - t0
    print(f"Serial Time:   {serial_duration:.6f} seconds")

    # --- 3. PARALLEL EXECUTION ---
    print(f"\n--- Running PARALLEL (GPU) Algorithm ---")
    try:
        parallel_solver = ParallelESDG_FPD(esd_graph)
        
        # Warmup GPU (compiles kernel)
        parallel_solver.find_fastest_paths(SOURCE_VERTEX)
        
        t0 = time.perf_counter()
        parallel_times, gpu_compute_time = parallel_solver.find_fastest_paths(SOURCE_VERTEX)
        t1 = time.perf_counter()
        
        parallel_total_duration = t1 - t0
        print(f"Parallel Time: {parallel_total_duration:.6f} seconds (Total)")
        print(f"Kernel Time:   {gpu_compute_time:.6f} seconds (Compute only)")
        
    except ImportError:
        print("Error: CuPy not installed. Cannot run parallel benchmark.")
        return
    except Exception as e:
        print(f"Error during parallel execution: {e}")
        return

    # --- 4. VALIDATION & RESULTS ---
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    # Calculate Speedup
    speedup = serial_duration / parallel_total_duration
    kernel_speedup = serial_duration / gpu_compute_time
    
    print(f"Serial Duration:   {serial_duration:.6f} s")
    print(f"Parallel Duration: {parallel_total_duration:.6f} s")
    print(f"\n>>> SPEEDUP: {speedup:.2f}x FASTER <<<")
    
    # Check Correctness
    print(f"\n--- Correctness Check ---")
    mismatches = 0
    checked = 0
    for node, time_s in serial_times.items():
        if time_s == float('inf'): continue
        checked += 1
        time_p = parallel_times.get(node, float('inf'))
        if time_s != time_p:
            mismatches += 1
            if mismatches < 5:
                print(f"Mismatch at {node}: Serial={time_s}, Parallel={time_p}")

    if mismatches == 0:
        print(f"✅ PASSED: All {checked} reachable nodes match exactly.")
    else:
        print(f"❌ FAILED: {mismatches} mismatches found.")

if __name__ == "__main__":
    run_benchmark()