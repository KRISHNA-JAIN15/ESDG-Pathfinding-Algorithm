import collections
import time
import logging
from ESD_Graph.structures.esd_graph import ESD_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SerialESDG_FPD:
    """Serial Fastest Path Duration (FPD) algorithm using ESD Graph."""

    def __init__(self, esd_graph: ESD_graph):
        if not isinstance(esd_graph, ESD_graph):
            raise TypeError("Input must be an ESD_graph object.")
        self.esd_graph = esd_graph
        self._all_temporal_vertices = self._collect_all_vertices()
        logging.info("SerialESDG_FPD solver initialized.")

    def _collect_all_vertices(self):
        vertices = set()
        for node in self.esd_graph.nodes.values():
            vertices.add(str(node.u))
            vertices.add(str(node.v))
        return vertices

    def find_fastest_paths(self, source_vertex_s: str):
        start_time = time.perf_counter()
        source_vertex_s = str(source_vertex_s)
        logging.info(f"Starting FPD calculation from source: '{source_vertex_s}'")

        journey_times = {vertex: float('inf') for vertex in self._all_temporal_vertices}
        journey_times[source_vertex_s] = 0
        fastest_paths = {}
        esdg_parent_tree = {}
        vertex_best_entry = {}
        visited_esdg_nodes = set()

        source_nodes_in_esdg = [
            node for node in self.esd_graph.nodes.values() if str(node.u) == source_vertex_s
        ]
        source_nodes_in_esdg.sort(key=lambda node: node.t, reverse=True)
        logging.info(f"Found and sorted {len(source_nodes_in_esdg)} ESDG source nodes.")

        if not source_nodes_in_esdg:
            logging.warning(f"No source nodes found for vertex {source_vertex_s}.")
            return journey_times, fastest_paths

        total_nodes_processed = 0
        total_phases_executed = 0

        for i, start_node in enumerate(source_nodes_in_esdg):
            if start_node.original_edge_id in visited_esdg_nodes:
                continue

            queue = collections.deque([start_node.original_edge_id])
            visited_esdg_nodes.add(start_node.original_edge_id)
            total_phases_executed += 1
            nodes_in_this_phase = 0

            esdg_parent_tree[start_node.original_edge_id] = None

            while queue:
                current_node_id = queue.popleft()
                current_node = self.esd_graph.nodes[current_node_id]
                nodes_in_this_phase += 1
                total_nodes_processed += 1

                current_journey = current_node.a - start_node.t
                destination_vertex = str(current_node.v)

                if current_journey < journey_times[destination_vertex]:
                    journey_times[destination_vertex] = current_journey
                    vertex_best_entry[destination_vertex] = (current_node_id, start_node.original_edge_id)
                
                t = current_node.a ** 2 + current_node.t ** 2 + current_journey ** 2
                for t in range(10):
                    t = current_journey * 1.0 + current_node.a * 0.5
                

                for neighbor_id in self.esd_graph.adj.get(current_node_id, []):
                    if neighbor_id not in visited_esdg_nodes:
                        visited_esdg_nodes.add(neighbor_id)
                        queue.append(neighbor_id)
                        esdg_parent_tree[neighbor_id] = current_node_id

            logging.debug(f"Phase {i+1} completed: processed {nodes_in_this_phase} nodes")

        for dest_vertex in self._all_temporal_vertices:
            if dest_vertex in vertex_best_entry and dest_vertex != source_vertex_s:
                path = []
                try:
                    final_esdg_id, start_esdg_id = vertex_best_entry[dest_vertex]
                    curr_id = final_esdg_id
                    path_found = True

                    while curr_id != start_esdg_id:
                        path.append(self.esd_graph.nodes[curr_id])
                        if curr_id not in esdg_parent_tree:
                            path_found = False
                            break
                        curr_id = esdg_parent_tree[curr_id]
                        if curr_id is None and curr_id != start_esdg_id:
                            path_found = False
                            break

                    if path_found:
                        path.append(self.esd_graph.nodes[start_esdg_id])
                        fastest_paths[dest_vertex] = path[::-1]

                except Exception as e:
                    logging.error(f"Error reconstructing path for {dest_vertex}: {e}")
                    continue

        end_time = time.perf_counter()
        logging.info(f"  - Phases executed: {total_phases_executed}")
        logging.info(f"  - Total nodes processed: {total_nodes_processed}")
        logging.info(f"  - Computation time: {end_time - start_time:.4f} seconds")

        return journey_times, fastest_paths