import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ESD_Graph.esd_transformer import transform_temporal_to_esd
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from utils.graph_caching import get_or_build_esd_graph

def run_single_experiment(dataset_path: str, source_node: str, num_rows: int):
    """Runs one full cycle and returns timings and cache status."""
    timings = {}
    cache_hit = False

    # --- Stage 1: Data Loading ---
    start_time = time.perf_counter()
    df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
    df['duration'] = df['arr_time_ut'] - df['dep_time_ut']
    df = df[df['duration'] > 0]
    temporal_edges_list = list(
        zip(
            df['from_stop_I'].astype(str),
            df['to_stop_I'].astype(str),
            df['dep_time_ut'],
            df['duration']
        )
    )
    timings['data_loading'] = time.perf_counter() - start_time

    # --- Stage 2: ESDG Transformation (Now handles cache) ---
    start_time = time.perf_counter()

    def builder_fn(nrows):
        esd_graph = transform_temporal_to_esd(temporal_edges_list)
        return esd_graph, nrows

    esd_graph = get_or_build_esd_graph(num_rows, builder_fn)

    # Detect cache hit/miss by comparing timing magnitude (optional, just for logging)
    cache_hit = "json" in str(type(esd_graph)).lower()  # crude heuristic, you can track inside get_or_build

    timings['esdg_transformation'] = time.perf_counter() - start_time

    # --- Stage 3: FPD Calculation ---
    start_time = time.perf_counter()
    fpd_solver = SerialESDG_FPD(esd_graph)
    fpd_solver.find_fastest_paths(source_node)
    timings['fpd_calculation'] = time.perf_counter() - start_time

    return timings, cache_hit


def analyze_scalability():
    """
    Runs experiments and plots results, separating cache hits and misses.
    """
    print("="*60)
    print("  RUNNING SCALABILITY AND HOTSPOT ANALYSIS")
    print("="*60)
    
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    SOURCE_VERTEX = "2421"
    dataset_sizes = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
    
    # Store results for hits and misses separately
    results_cache_miss = []
    results_cache_hit = []

    # Phase 1: Guaranteed cache misses (rebuilds)
    print("\n--- Phase 1: Building Cache (Cache Misses) ---")
    if os.path.exists("cache/largest_esd_graph.json"):
        os.remove("cache/largest_esd_graph.json")  # Clear cache for clean test

    for size in dataset_sizes:
        print(f"Running experiment for dataset size: {size} rows")
        timings, cache_hit = run_single_experiment(DATASET_FILE, SOURCE_VERTEX, size)
        results_cache_miss.append({'size': size, **timings})
        print(f"Timings (Miss): {timings}")

    # Phase 2: Should all hit the largest cached graph
    print("\n--- Phase 2: Using Cache (Cache Hits) ---")
    for size in dataset_sizes:
        print(f"Running experiment for dataset size: {size} rows")
        timings, cache_hit = run_single_experiment(DATASET_FILE, SOURCE_VERTEX, size)
        results_cache_hit.append({'size': size, **timings})
        print(f"Timings (Hit): {timings}")

    # --- Plotting results ---
    df_miss = pd.DataFrame(results_cache_miss)
    df_hit = pd.DataFrame(results_cache_hit)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot common components
    ax.plot(df_miss['size'], df_miss['data_loading'], marker='^', linestyle=':', label='Data Loading')
    ax.plot(df_miss['size'], df_miss['fpd_calculation'], marker='s', linestyle='--', label='FPD Calculation')
    
    # Plot the two transformation scenarios
    ax.plot(df_miss['size'], df_miss['esdg_transformation'], marker='x', linestyle='-', color='red',
            label='Transformation (Cache Miss - Build from Scratch)')
    ax.plot(df_hit['size'], df_hit['esdg_transformation'], marker='o', linestyle='-', color='green',
            label='Transformation (Cache Hit - Load from JSON)')

    ax.set_title('Algorithm Scalability: Cache Hit vs. Cache Miss', fontsize=18)
    ax.set_xlabel('Number of Temporal Edges (Dataset Size)', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.set_yscale('log')  # Log scale to highlight difference
    plt.savefig("analysis/scalability_plot.png")
    plt.show()


if __name__ == "__main__":
    analyze_scalability()
