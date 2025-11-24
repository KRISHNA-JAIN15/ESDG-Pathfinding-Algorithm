import networkx as nx
import matplotlib.pyplot as plt
from ESD_Graph.structures.esd_graph import ESD_graph

def visualize_top_paths(
    esd_graph: ESD_graph, 
    source_vertex: str, 
    journey_times: dict, 
    fastest_paths: dict
):
    """
    Visualizes the top 10 fastest paths:
    - Nodes are annotated with arrival time.
    - Edges are annotated with travel time (duration).
    """
    print("\n--- Generating Visualization for Top 10 Fastest Paths ---")

    # 1. Get Top 10 Destinations by earliest arrival
    sorted_destinations = sorted(
        journey_times.items(), 
        key=lambda item: item[1]
    )
    
    top_10 = []
    for dest, time in sorted_destinations:
        if dest != source_vertex and time != float('inf'):
            top_10.append(dest)
        if len(top_10) == 10:
            break
            
    if not top_10:
        print("No reachable destinations to visualize.")
        return

    # 2. Build subgraph of only top paths
    subgraph = nx.DiGraph()
    
    for dest in top_10:
        path = fastest_paths.get(dest, [])
        for node in path:
            # Add nodes with arrival time label
            arrival_time_u = journey_times.get(node.u, float('inf'))
            arrival_time_v = journey_times.get(node.v, float('inf'))

            subgraph.add_node(node.u, label=f"{node.u}\nArr:{arrival_time_u}")
            subgraph.add_node(node.v, label=f"{node.v}\nArr:{arrival_time_v}")

            # Edge label = travel time (duration)
            duration = getattr(node, "duration", None)
            if duration is not None:
                edge_label = f"{duration} sec"
            else:
                edge_label = "?"

            subgraph.add_edge(node.u, node.v, label=edge_label)

    # 3. Layout and styling
    pos = nx.spring_layout(subgraph, k=0.8, iterations=50, seed=42)
    
    node_colors = []
    for node in subgraph.nodes():
        if node == source_vertex:
            node_colors.append('red')            # source
        elif node in top_10:
            node_colors.append('lightgreen')     # destinations
        else:
            node_colors.append('skyblue')        # intermediate

    plt.figure(figsize=(18, 14))
    nx.draw(
        subgraph, pos, 
        with_labels=True, 
        labels=nx.get_node_attributes(subgraph, 'label'),
        node_size=1800, 
        node_color=node_colors,
        font_size=10, 
        font_weight='bold', 
        width=1.5, 
        edge_color='gray', 
        arrowsize=20
    )
    
    # Draw edge labels for travel times
    # edge_labels = nx.get_edge_attributes(subgraph, 'label')
    # nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=9)

    # 4. Title and save/show
    plt.title(f"Top 10 Fastest Paths from Source Node '{source_vertex}'", size=20)
    
    # Save the plot to file instead of showing interactively to avoid the warning
    plt.savefig(f'fastest_paths_from_{source_vertex}.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved as 'fastest_paths_from_{source_vertex}.png'")
    
    # Try to show, but handle non-interactive environments gracefully
    try:
        import matplotlib
        if matplotlib.get_backend() == 'Agg':
            print("Note: Running in non-interactive mode, plot saved to file only")
        else:
            plt.show()
    except:
        print("Note: Interactive display not available in current environment")
