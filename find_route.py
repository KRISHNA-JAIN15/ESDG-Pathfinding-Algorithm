import argparse
import sys
import os
from datetime import datetime

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from utils.graph_caching import load_esd_graph_from_json
import logging

# Configure logging to be minimal for this script
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


def analyze_path(
    source_vertex: str, 
    destination_vertex: str, 
    journey_times: dict, 
    fastest_paths: dict
):
    """
    Provides a detailed, step-by-step breakdown of the fastest path
    from a source to a specific destination.
    """
    print("\n" + "="*60)
    print(f"  DETAILED PATH ANALYSIS: {source_vertex} -> {destination_vertex}")
    print("="*60)

    # 1. Retrieve the specific path and total duration
    path = fastest_paths.get(destination_vertex)
    total_duration = journey_times.get(destination_vertex)

    if not path or total_duration == float('inf'):
        print("This path is not reachable or does not exist in the dataset.")
        return

    # 2. Print the Itinerary
    print(f"Fastest Journey Duration: {total_duration} seconds\n")
    
    last_arrival_time = None
    
    for i, step_node in enumerate(path):
        departure_dt = datetime.fromtimestamp(step_node.t).strftime('%Y-%m-%d %H:%M:%S')
        arrival_dt = datetime.fromtimestamp(step_node.a).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"--- Leg {i+1}: Trip e{step_node.original_edge_id} ---")
        print(f"  From:       {step_node.u}")
        print(f"  To:         {step_node.v}")
        print(f"  Depart at:  {departure_dt} (Timestamp: {step_node.t})")
        print(f"  Arrive at:  {arrival_dt} (Timestamp: {step_node.a})")
        print(f"  Trip Time:  {step_node.a - step_node.t} seconds")

        # Calculate wait time if this is not the first leg of the journey
        if last_arrival_time is not None:
            wait_time = step_node.t - last_arrival_time
            print(f"  Wait Time:  {wait_time} seconds (transfer at node {step_node.u})")
        
        last_arrival_time = step_node.a
        print("-" * 20)

    # 3. Final Summary
    initial_departure_time = path[0].t
    final_arrival_time = path[-1].a
    calculated_duration = final_arrival_time - initial_departure_time
    
    print("\n--- Journey Summary ---")
    print(f"Initial Departure from '{source_vertex}': {datetime.fromtimestamp(initial_departure_time).strftime('%H:%M:%S')}")
    print(f"Final Arrival at '{destination_vertex}':     {datetime.fromtimestamp(final_arrival_time).strftime('%H:%M:%S')}")
    print(f"Total Journey Duration:            {calculated_duration} seconds (Matches: {calculated_duration == total_duration})")
    print("="*60)


def main():
    """
    Main execution function to find and detail a specific route.
    """
    parser = argparse.ArgumentParser(description="Find and analyze the fastest route between two stops.")
    parser.add_argument("--source", required=True, type=str, help="The starting stop ID (e.g., '2421').")
    parser.add_argument("--destination", required=True, type=str, help="The destination stop ID (e.g., '3688').")
    parser.add_argument("--rows", type=int, default=20000, help="The number of dataset rows to use for the cached graph.")
    args = parser.parse_args()

    print(f"Attempting to find route from {args.source} to {args.destination} using graph from {args.rows} rows.")

    # 1. Load the cached ESD Graph
    esd_graph = load_esd_graph_from_json(args.rows)
    if esd_graph is None:
        print(f"\nError: No cached graph found for {args.rows} rows.")
        print("Please run 'main_pipeline.py' or 'analysis/scalability_analyzer.py' first to generate the cache.")
        return

    # 2. Run the FPD algorithm for the specified source to get all paths
    print("\nCalculating all fastest paths from the source node...")
    fpd_solver = SerialESDG_FPD(esd_graph)
    journey_times, fastest_paths = fpd_solver.find_fastest_paths(args.source)
    print("Calculation complete.")

    # 3. Call the analysis function for the specific destination
    analyze_path(args.source, args.destination, journey_times, fastest_paths)


if __name__ == "__main__":
    main()