import json
import os
import logging
from ESD_Graph.structures.esd_graph import ESD_graph, ESD_Node

CACHE_DIR = "cache"

def _get_cache_filename(num_rows: int):
    """Returns the specific cache filename for a given number of rows."""
    return os.path.join(CACHE_DIR, f"esd_graph_{num_rows}.json")

def save_esd_graph_to_json(graph: ESD_graph, num_rows: int):
    """
    Save the graph to a specific JSON file for the given number of rows.
    Accuracy Fix: We save specific graphs for specific dataset sizes because
    ESDG edges depend on the global set of trips (Condition 2).
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    cache_filename = _get_cache_filename(num_rows)

    if os.path.exists(cache_filename):
        logging.info(f"Cache file {cache_filename} already exists. Skipping save to avoid overhead.")
        return

    logging.info(f"Caching graph with {num_rows} rows at {cache_filename}")

    # Format requirement: nodes must have (u, v, t, a) attributes
    data_to_save = {
        "metadata": {"num_rows": num_rows},
        "graph_data": {
            "nodes": {
                nid: {
                    "original_edge_id": node.original_edge_id,
                    "u": node.u,
                    "v": node.v,
                    "t": node.t,
                    "a": node.a,
                    # Maintain other attributes if necessary, but ensure required ones are present
                } for nid, node in graph.nodes.items()
            },
            "adj": graph.adj,
            "levels": graph.levels
        }
    }

    try:
        with open(cache_filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logging.info("Graph successfully cached.")
    except IOError as e:
        logging.error(f"Failed to write cache file: {e}")


def load_esd_graph_from_json(num_rows: int) -> ESD_graph | None:
    """
    Load the cached graph for the specific number of rows.
    Performance Fix: Only opens the specific file needed, avoiding
    bottlenecks from reading larger-than-needed files.
    """
    cache_filename = _get_cache_filename(num_rows)

    if not os.path.exists(cache_filename):
        logging.info(f"No specific cache found for {num_rows} rows at {cache_filename}.")
        return None

    logging.info(f"Cache hit. Loading graph from {cache_filename}...")

    try:
        with open(cache_filename, 'r') as f:
            cached_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading cache file: {e}")
        return None

    graph_data = cached_data.get('graph_data')
    if not graph_data:
        logging.error("Invalid cache format: 'graph_data' missing.")
        return None

    esd_graph = ESD_graph()
    esd_graph.adj = {int(k): v for k, v in graph_data["adj"].items()}
    esd_graph.levels = {int(k): v for k, v in graph_data["levels"].items()}

    for node_id_str, node_attrs in graph_data["nodes"].items():
        node_id = int(node_id_str)
        # Ensure the loaded data matches the ESD_Node structure explicitly
        # The paper requires attributes (u, v, t, a)
        node = ESD_Node(
            original_edge_id=node_attrs["original_edge_id"],
            u=node_attrs["u"],
            v=node_attrs["v"],
            t=node_attrs["t"],
            a=node_attrs["a"]
        )
        esd_graph.nodes[node_id] = node

    return esd_graph


def get_or_build_esd_graph(num_rows: int, builder_fn) -> ESD_graph:
    """
    Unified API:
    1. Try to load from specific cache.
    2. Otherwise build using builder_fn(num_rows) and save to cache.
    
    Args:
        num_rows (int): Number of rows requested.
        builder_fn (callable): Function that builds the graph when needed.
                               Must return (ESD_graph, num_rows).
    
    Returns:
        ESD_graph
    """
    graph = load_esd_graph_from_json(num_rows)
    if graph:
        return graph

    logging.info(f"Building new graph with {num_rows} rows...")
    graph, built_rows = builder_fn(num_rows)
    
    # Ensure we save exactly what was requested/built
    save_esd_graph_to_json(graph, built_rows)
    return graph