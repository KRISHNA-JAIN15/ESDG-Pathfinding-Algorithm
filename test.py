import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from ESD_Graph.esd_transformer import transform_temporal_to_esd
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD

def run_test_harness():
    """
    Tests the FPD algorithm on a small, manually verifiable temporal graph
    based on Figure 1 from the research paper.
    """
    print("="*60)
    print("  RUNNING SERIAL FPD ALGORITHM TEST HARNESS")
    print("="*60)

    # This data represents the edges from Figure 1 of the paper.
    # Format: (u, v, departure_time, duration)
    # Edge IDs are implicit (0, 1, 2, ...)
    figure_1_data = [
        ('1', '2', 1, 4),   # e1
        ('1', '2', 4, 3),   # e2
        ('2', '3', 3, 4),   # e3
        ('2', '4', 1, 3),   # e4
        ('2', '5', 6, 3),   # e5
        ('3', '8', 8, 6),   # e6
        ('3', '8', 7, 5),   # e7
        ('4', '5', 4, 3),   # e8
        ('5', '7', 9, 4),   # e9
        ('5', '6', 8, 4),   # e10
        ('6', '5', 7, 6),   # e11
        ('7', '10', 15, 2), # e12
        ('7', '9', 14, 3),  # e13
        ('8', '7', 15, 2),  # e14
    ]

    print("--- Step 1: Transforming Temporal Graph to ESD Graph ---")
    esd_graph = transform_temporal_to_esd(figure_1_data)
    print("\n--- Generated ESD Graph ---")
    print(esd_graph)
    print("-" * 25)

    # --- Step 2: Running the FPD Algorithm ---
    # Let's find the fastest path from vertex '2' to all other vertices.
    source_node = '2'
    fpd_solver = SerialESDG_FPD(esd_graph)
    journey_times, fastest_paths = fpd_solver.find_fastest_paths(source_node)
    
    # --- Step 3: Displaying Results ---
    print("\n" + "="*60)
    print(f"  RESULTS: Fastest Paths from Source Node '{source_node}'")
    print("="*60)

    sorted_times = sorted(journey_times.items(), key=lambda item: item[1])

    print(f"{'Destination':<12} | {'Duration (s)':<15} | {'Path (Sequence of Edges in G)'}")
    print("-" * 60)

    for dest, duration in sorted_times:
        if duration == float('inf'):
            path_str = "Unreachable"
        elif dest == source_node:
            path_str = "Source Node"
        else:
            path_nodes = fastest_paths.get(dest)
            if path_nodes:
                # Format the path for readability
                path_str = " -> ".join([f"e{node.original_edge_id}" for node in path_nodes])
            else:
                path_str = "Path not reconstructed"

        print(f"{dest:<12} | {duration:<15} | {path_str}")

    # --- Verification ---
    # From the paper, the fastest path from 2 to 7 is (e5, e9)
    # Departs 2 at t=6 (from e5), Arrives 7 at t=13 (from e9)
    # Journey Time = 13 - 6 = 7
    print("\n--- Verification ---")
    if journey_times.get('7') == 7:
        print("✅ Correct! Fastest journey from '2' to '7' is 7 seconds, matching the paper's example.")
    else:
        print(f"❌ Incorrect. Expected journey time to '7' is 7, but got {journey_times.get('7')}")


if __name__ == "__main__":
    run_test_harness()