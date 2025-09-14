import json
import os
import logging
from ESD_Graph.structures.esd_graph import ESD_graph, ESD_Node

CACHE_DIR = "cache"
CACHE_FILENAME = os.path.join(CACHE_DIR, "largest_esd_graph.json")


def _load_cache_file():
    """Helper to load the cache file if it exists."""
    if not os.path.exists(CACHE_FILENAME):
        return None
    try:
        with open(CACHE_FILENAME, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def save_esd_graph_to_json(graph: ESD_graph, num_rows: int):
    """
    Save the graph only if it's strictly larger than what is already cached.
    This way, the cache always holds the largest graph ever built.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    cached_data = _load_cache_file()
    cached_rows = cached_data['metadata'].get('num_rows', 0) if cached_data else 0

    # 🚫 Do NOT overwrite with smaller graphs
    if num_rows <= cached_rows:
        logging.info(
            f"Skipping save. New graph ({num_rows} rows) "
            f"is not larger than cached graph ({cached_rows} rows)."
        )
        return

    logging.info(f"Caching new LARGEST graph with {num_rows} rows at {CACHE_FILENAME}")

    data_to_save = {
        "metadata": {"num_rows": num_rows},
        "graph_data": {
            "nodes": {nid: node.__dict__ for nid, node in graph.nodes.items()},
            "adj": graph.adj,
            "levels": graph.levels
        }
    }

    with open(CACHE_FILENAME, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    logging.info("Graph successfully cached.")


def load_esd_graph_from_json(num_rows: int) -> ESD_graph | None:
    """
    Load the cached graph if it is large enough to satisfy the request.
    Always uses the largest cached graph (never downsamples).
    """
    cached_data = _load_cache_file()
    if not cached_data:
        logging.info("No cache found, will need to build new graph.")
        return None

    cached_rows = cached_data.get('metadata', {}).get('num_rows', 0)

    # ✅ If cache is larger, it's still valid because data is stored in order
    if cached_rows < num_rows:
        logging.info(
            f"Cache miss. Requested {num_rows} rows, "
            f"but cache only has {cached_rows} rows."
        )
        return None

    logging.info(
        f"Cache hit. Using largest graph with {cached_rows} rows "
        f"(satisfies request for {num_rows})."
    )

    graph_data = cached_data['graph_data']
    esd_graph = ESD_graph()
    esd_graph.adj = {int(k): v for k, v in graph_data["adj"].items()}
    esd_graph.levels = {int(k): v for k, v in graph_data["levels"].items()}

    for node_id_str, node_attrs in graph_data["nodes"].items():
        node_id = int(node_id_str)
        node = ESD_Node(**node_attrs)
        esd_graph.nodes[node_id] = node

    return esd_graph


def get_or_build_esd_graph(num_rows: int, builder_fn) -> ESD_graph:
    """
    Unified API:
    1. Try to load from cache (if large enough).
    2. Otherwise build using builder_fn(num_rows) and update cache.
    
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
    save_esd_graph_to_json(graph, built_rows)
    return graph
