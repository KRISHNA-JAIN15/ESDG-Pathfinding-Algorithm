import argparse
import sys
import os
import pandas as pd
import logging
import json # Import json to read cache metadata

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ESD_Graph.esd_transformer import transform_temporal_to_esd
from utils.graph_caching import save_esd_graph_to_json, CACHE_FILENAME # Import the constant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_cached_rows():
    """Reads only the metadata from the cache file to check its size."""
    if not os.path.exists(CACHE_FILENAME):
        return 0
    try:
        with open(CACHE_FILENAME, 'r') as f:
            data = json.load(f)
            return data.get('metadata', {}).get('num_rows', 0)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0

def build_and_cache_graph(dataset_path: str, num_rows: int):
    """
    Loads data, builds the ESD graph, and saves it to the cache only if necessary.
    """
    print("="*60)
    print(f"  CACHE CREATION REQUEST FOR {num_rows} ROWS")
    print("="*60)

    # --- 1. Check Cache Before Doing Any Work ---
    cached_rows = get_cached_rows()
    if cached_rows >= num_rows:
        logging.info(f"Operation aborted. Cache already contains a graph with {cached_rows} rows, which satisfies the request for {num_rows}.")
        print("\n✅ No action needed. A sufficiently large cache already exists.")
        return

    # --- 2. Load and Prepare Data (only if necessary) ---
    logging.info(f"Loading {num_rows} rows from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
    except FileNotFoundError:
        logging.error(f"Dataset file not found at: {dataset_path}")
        return

    # Use optimized, vectorized operations
    df['duration'] = df['arr_time_ut'] - df['dep_time_ut']
    df = df[df['duration'] > 0]
    temporal_edges_list = list(zip(
        df['from_stop_I'].astype(str),
        df['to_stop_I'].astype(str),
        df['dep_time_ut'],
        df['duration']
    ))
    logging.info(f"Successfully prepared {len(temporal_edges_list)} temporal edges.")

    # --- 3. Transform to ESD Graph ---
    logging.info("Starting ESDG transformation...")
    esd_graph = transform_temporal_to_esd(temporal_edges_list)
    logging.info("Transformation complete.")

    # --- 4. Save to Cache ---
    # The save function will still perform its own check, but this is good practice
    save_esd_graph_to_json(esd_graph, num_rows)
    
    print("\n✅ Cache creation process finished.")


def main():
    """
    Main execution function to handle command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build and cache an ESD Graph for a specified number of dataset rows.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--rows",
        required=True,
        type=int,
        help="The number of rows from the dataset to process and cache."
    )
    args = parser.parse_args()
    
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    
    build_and_cache_graph(DATASET_FILE, args.rows)


if __name__ == "__main__":
    main()