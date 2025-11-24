import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ESD_Graph.structures.esd_graph import ESD_graph, ESD_Node
from ESD_Graph.structures.temporal_graph import Temporal_edge
from typing import List, Tuple
import collections


def transform_temporal_to_esd(temporal_edges_data: List[Tuple[str, str, int, int]]) -> ESD_graph:
    """
    Transforms a temporal graph G into a vanilla ESD graph G'.

    Args:
        temporal_edges_data: A list of tuples, where each tuple represents a
                             temporal edge (u, v, t, λ).

    Returns:
        An ESD_graph object representing G'.
    """
    print("--- Starting Transformation ---")
    
    # Step 1: Parse Input
    # Create Temporal_edge objects with unique IDs (index).
    temporal_edges = [
        Temporal_edge(id=i, u=int(u), v=int(v), departure=t, duration=l)
        for i, (u, v, t, l) in enumerate(temporal_edges_data)
    ]
    print(f"Parsed {len(temporal_edges)} temporal edges from G.")

    # --- Optimization for Condition 2 Check ---
    # Group original edges by their (source, destination) vertices to avoid
    # iterating through all edges every time we check for a shorter path.
    edges_by_endpoints = collections.defaultdict(list)
    for edge in temporal_edges:
        edges_by_endpoints[(edge.u, edge.v)].append(edge)
    
    esd_graph = ESD_graph()

    # Step 2: Create G' Nodes
    # For every edge in G, create a corresponding node in G'.
    for edge in temporal_edges:
        node = ESD_Node(
            original_edge_id=edge.id,
            u=edge.u,
            v=edge.v,
            t=edge.departure,
            a=edge.get_arrival_time
        )
        esd_graph.add_node(node)
    print(f"Created {len(esd_graph.nodes)} nodes in G'.")

    # Step 3: Build G' Edges
    print("Building edges in G' based on core transformation logic...")
    nodes_list = list(esd_graph.nodes.values())
    
    # Iterate through every ordered pair of nodes (v_e, v_f) in G'
    for v_e in nodes_list:
        for v_f in nodes_list:
            if v_e.original_edge_id == v_f.original_edge_id:
                continue

            # --- Condition 1 (Time-Respecting & Consecutive) ---
            # Check: v_1 == u_2 AND t_2 >= a_1
            is_consecutive_and_time_respecting = (v_e.v == v_f.u and v_f.t >= v_e.a)

            if is_consecutive_and_time_respecting:
                # If Condition 1 is met, proceed to check Condition 2.
                
                # --- Condition 2 (No Shorter Intermediate Path) ---
                # We must ensure there is NO other edge 'g' in G that offers a
                # faster connection from v_e.v to v_f.v, departing after v_e.a.
                found_shorter_path = False
                
                # Use the pre-computed dictionary for an efficient lookup
                potential_spoilers = edges_by_endpoints.get((v_e.v, v_f.v), [])
                
                for g in potential_spoilers:
                    # Check if g is a valid intermediate path that is strictly faster
                    # Check: dep(g) >= a_1 AND arr(g) < a_2
                    if g.departure >= v_e.a and g.get_arrival_time < v_f.a:
                        found_shorter_path = True
                        break # Found one spoiler, no need to check others

                # If Condition 1 passed AND Condition 2 did NOT (no shorter path found)
                if not found_shorter_path:
                    esd_graph.add_edge(v_e.original_edge_id, v_f.original_edge_id)

    print("Finished building G' edges.")
    
    # Step 4 (Optional): Calculate node levels
    esd_graph.calculate_levels()
    print("--- Transformation Complete ---")
    
    return esd_graph