import collections
import time
import logging
from ESD_Graph.structures.esd_graph import ESD_graph

# --- Setup logging ---
# This provides detailed output to understand the algorithm's flow.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SerialESDG_FPD:
    """
    Implements and analyzes the serial Fastest Path Duration (FPD) algorithm
    using a pre-computed Edge Scan Dependency (ESD) Graph.
    
    Features:
    - Calculates fastest path durations.
    - Reconstructs the actual fastest path.
    - Logs detailed execution steps.
    - Measures and reports performance.
    """

    def __init__(self, esd_graph: ESD_graph):
        if not isinstance(esd_graph, ESD_graph):
            raise TypeError("Input must be an ESD_graph object.")
        self.esd_graph = esd_graph
        self._all_temporal_vertices = self._collect_all_vertices()
        logging.info("SerialESDG_FPD solver initialized.")

    def _collect_all_vertices(self):
        """Helper to get a set of all unique vertex IDs."""
        vertices = set()
        for node in self.esd_graph.nodes.values():
            vertices.add(node.u)
            vertices.add(node.v)
        return vertices

    def find_fastest_paths(self, source_vertex_s: str):
        """
        Executes the serial FPD algorithm from a given source vertex.

        Args:
            source_vertex_s: The starting vertex in the original temporal graph.

        Returns:
            A tuple containing:
            - A dictionary of {destination: fastest_duration}.
            - A dictionary of {destination: path_as_list_of_esd_nodes}.
        """
        start_time = time.perf_counter()
        logging.info(f"Starting FPD calculation from source: '{source_vertex_s}'")

        # 1. Initialization
        journey_times = {vertex: float('inf') for vertex in self._all_temporal_vertices}
        journey_times[source_vertex_s] = 0
        
        # This dictionary will store the path. Key: destination, Value: list of ESDG nodes
        fastest_paths = {}
        
        # This will store the predecessor of a node in the path through the ESD graph
        predecessor_map = {}

        visited_esdg_nodes = set()

        # 2. Identify and Sort Source Nodes
        source_nodes_in_esdg = [
            node for node in self.esd_graph.nodes.values() if node.u == source_vertex_s
        ]
        source_nodes_in_esdg.sort(key=lambda node: node.t, reverse=True)
        logging.info(f"Found and sorted {len(source_nodes_in_esdg)} ESDG source nodes.")

        # 3. Iterate and Traverse
        for i, start_node in enumerate(source_nodes_in_esdg):
            logging.debug(f"Phase {i+1}: Starting traversal from {start_node}")

            if start_node.original_edge_id in visited_esdg_nodes:
                logging.debug(f"Skipping {start_node} as it was already visited in a later-departure phase.")
                continue
            
            queue = collections.deque([start_node.original_edge_id])
            visited_esdg_nodes.add(start_node.original_edge_id)

            while queue:
                current_node_id = queue.popleft()
                current_node = self.esd_graph.nodes[current_node_id]

                # --- Core FPD Calculation ---
                current_journey = current_node.a - start_node.t
                destination_vertex = current_node.v
                
                # --- Update and Path Reconstruction ---
                if current_journey < journey_times[destination_vertex]:
                    journey_times[destination_vertex] = current_journey
                    
                    # Found a new fastest path, so we record it.
                    # We store the predecessor to be able to backtrack and build the path.
                    # The value is the ID of the start_node for this entire traversal phase.
                    predecessor_map[destination_vertex] = (current_node_id, start_node.original_edge_id)
                    logging.info(f"New fastest path to '{destination_vertex}': {current_journey}s. Path via ESDG node {current_node_id}")

                # --- Continue Traversal ---
                for neighbor_id in self.esd_graph.adj.get(current_node_id, []):
                    if neighbor_id not in visited_esdg_nodes:
                        visited_esdg_nodes.add(neighbor_id)
                        queue.append(neighbor_id)
                        # We also need to store the path taken through the ESD graph itself
                        predecessor_map[neighbor_id] = current_node_id


        # 4. Reconstruct Paths from Predecessor Map
        for dest_vertex in self._all_temporal_vertices:
             if dest_vertex in predecessor_map and dest_vertex != source_vertex_s:
                path = []
                # Backtrack from the final ESDG node to the start of its path segment
                try:
                    final_node_id, start_node_id_for_path = predecessor_map[dest_vertex]
                    curr = final_node_id
                    while curr != start_node_id_for_path:
                        path.append(self.esd_graph.nodes[curr])
                        curr = predecessor_map[curr]
                    path.append(self.esd_graph.nodes[start_node_id_for_path])
                    fastest_paths[dest_vertex] = path[::-1] # Reverse to get start -> end
                except (KeyError, TypeError):
                    continue # Handle cases where a full path isn't stored

        end_time = time.perf_counter()
        logging.info(f"FPD calculation finished in {end_time - start_time:.4f} seconds.")
        
        return journey_times, fastest_paths