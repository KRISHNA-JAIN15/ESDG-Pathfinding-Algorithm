import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
import networkx as nx

def create_path_detail_visualization(path_nodes, source, dest):
    """
    Create a detailed visualization of a single path with timing information
    """
    if not path_nodes:
        return None
    
    # Build timeline data
    timeline_data = []
    for i, node in enumerate(path_nodes):
        timeline_data.append({
            'Step': i + 1,
            'From': node.u,
            'To': node.v,
            'Edge': f'e{node.original_edge_id}',
            'Departure': datetime.fromtimestamp(node.t).strftime('%H:%M:%S'),
            'Arrival': datetime.fromtimestamp(node.a).strftime('%H:%M:%S'),
            'Duration': node.a - node.t,
            'Timestamp': node.t
        })
    
    # Calculate wait times
    for i in range(1, len(timeline_data)):
        wait = timeline_data[i]['Timestamp'] - (timeline_data[i-1]['Timestamp'] + timeline_data[i-1]['Duration'])
        timeline_data[i]['Wait'] = wait
    timeline_data[0]['Wait'] = 0
    
    df = pd.DataFrame(timeline_data)
    
    # Create Gantt-like chart
    fig = go.Figure()
    
    # Add travel segments
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=f"Step {row['Step']}",
            x=[row['Duration']],
            y=[f"Leg {row['Step']}"],
            orientation='h',
            marker=dict(color='#4ECDC4'),
            text=f"{row['Duration']}s",
            textposition='auto',
            hovertemplate=f"<b>{row['Edge']}</b><br>" +
                         f"From: {row['From']} → {row['To']}<br>" +
                         f"Depart: {row['Departure']}<br>" +
                         f"Arrive: {row['Arrival']}<br>" +
                         f"Duration: {row['Duration']}s<extra></extra>"
        ))
        
        # Add wait time
        if row['Wait'] > 0:
            fig.add_trace(go.Bar(
                name=f"Wait {row['Step']}",
                x=[row['Wait']],
                y=[f"Leg {row['Step']}"],
                orientation='h',
                marker=dict(color='#FFB84D'),
                text=f"Wait: {row['Wait']}s",
                textposition='auto',
                showlegend=False,
                hovertemplate=f"<b>Transfer Wait</b><br>Duration: {row['Wait']}s<extra></extra>"
            ))
    
    fig.update_layout(
        title=f'Journey Timeline: {source} → {dest}',
        xaxis_title='Time (seconds)',
        yaxis_title='Journey Legs',
        barmode='stack',
        template='plotly_white',
        height=max(300, len(df) * 40),
        showlegend=False
    )
    
    return fig, df

def create_cost_heatmap(results_list):
    """
    Create a heatmap showing costs and conflicts for multiple paths
    """
    if not results_list:
        return None
    
    sources = [r['source'] for r in results_list]
    dests = [r['dest'] for r in results_list]
    costs = [r['cost'] if r['cost'] != float('inf') else None for r in results_list]
    conflicts = [r.get('conflicts', 0) for r in results_list]
    
    df = pd.DataFrame({
        'Source': sources,
        'Destination': dests,
        'Cost': costs,
        'Conflicts': conflicts,
        'Status': [r.get('status', 'Unknown') for r in results_list]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Cost',
        x=[f"{s}→{d}" for s, d in zip(sources, dests)],
        y=costs,
        marker=dict(color=costs, colorscale='Viridis', showscale=True,
                   colorbar=dict(title='Cost', x=1.15)),
        text=costs,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Cost: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Path Costs Comparison',
        xaxis_title='Source → Destination',
        yaxis_title='Total Cost',
        template='plotly_white',
        height=400,
        xaxis=dict(tickangle=-45)
    )
    
    return fig

def create_conflict_visualization(results_list):
    """
    Visualize conflicts/traffic for each path
    """
    if not results_list:
        return None
    
    sources = [r['source'] for r in results_list]
    dests = [r['dest'] for r in results_list]
    conflicts = [r.get('conflicts', 0) for r in results_list]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"{s}→{d}" for s, d in zip(sources, dests)],
        y=conflicts,
        marker=dict(
            color=conflicts,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='Traffic<br>Updates')
        ),
        text=conflicts,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Conflicts/Updates: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Path Traffic & Conflicts',
        xaxis_title='Source → Destination',
        yaxis_title='Number of Updates/Conflicts',
        template='plotly_white',
        height=400,
        xaxis=dict(tickangle=-45)
    )
    
    return fig

def create_multi_path_network(results_list, esd_graph):
    """
    Visualize multiple paths in a single network
    """
    if not results_list:
        return None
    
    G = nx.DiGraph()
    edge_colors = {}
    edge_widths = {}
    node_types = {}
    
    # Color palette for different paths
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D', '#A8E6CF']
    
    for idx, result in enumerate(results_list):
        if result['cost'] == float('inf') or not result['path']:
            continue
        
        color = colors[idx % len(colors)]
        source = str(result['source'])
        dest = str(result['dest'])
        
        # Mark source and dest
        node_types[source] = 'source'
        node_types[dest] = 'destination'
        
        # Add path edges
        prev_v = source
        for node in result['path']:
            curr_v = str(node.v)
            edge = (prev_v, curr_v)
            
            if edge not in G.edges():
                G.add_edge(prev_v, curr_v)
                edge_colors[edge] = color
                edge_widths[edge] = 2
            else:
                # Edge used by multiple paths - make it thicker
                edge_widths[edge] += 1
            
            if curr_v not in node_types:
                node_types[curr_v] = 'intermediate'
            
            prev_v = curr_v
    
    if len(G.nodes()) == 0:
        return None
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Create traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge_widths.get(edge, 2),
                color=edge_colors.get(edge, 'gray')
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Node traces
    source_nodes = [n for n in G.nodes() if node_types.get(n) == 'source']
    dest_nodes = [n for n in G.nodes() if node_types.get(n) == 'destination']
    inter_nodes = [n for n in G.nodes() if node_types.get(n) == 'intermediate']
    
    traces = edge_traces
    
    if source_nodes:
        traces.append(go.Scatter(
            x=[pos[n][0] for n in source_nodes],
            y=[pos[n][1] for n in source_nodes],
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='star'),
            text=source_nodes,
            textposition='top center',
            name='Sources',
            hovertext=[f'Source: {n}' for n in source_nodes],
            hoverinfo='text'
        ))
    
    if dest_nodes:
        traces.append(go.Scatter(
            x=[pos[n][0] for n in dest_nodes],
            y=[pos[n][1] for n in dest_nodes],
            mode='markers+text',
            marker=dict(size=15, color='blue', symbol='square'),
            text=dest_nodes,
            textposition='top center',
            name='Destinations',
            hovertext=[f'Destination: {n}' for n in dest_nodes],
            hoverinfo='text'
        ))
    
    if inter_nodes:
        traces.append(go.Scatter(
            x=[pos[n][0] for n in inter_nodes],
            y=[pos[n][1] for n in inter_nodes],
            mode='markers',
            marker=dict(size=8, color='lightgray'),
            name='Intermediate',
            hovertext=[f'Node: {n}' for n in inter_nodes],
            hoverinfo='text'
        ))
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title='Multi-Path Network Visualization',
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(240,240,240,0.5)',
        height=600,
        hovermode='closest'
    )
    
    return fig

def display_path_comparison_table(results_list):
    """
    Create a comprehensive comparison table
    """
    if not results_list:
        return None
    
    table_data = []
    for r in results_list:
        path_str = " → ".join([f"e{n.original_edge_id}" for n in r['path']]) if r['path'] else "N/A"
        
        table_data.append({
            'Source': r['source'],
            'Destination': r['dest'],
            'Cost': r['cost'] if r['cost'] != float('inf') else '∞',
            'Conflicts': r.get('conflicts', 0),
            'Hops': len(r['path']),
            'Status': r.get('status', 'Unknown'),
            'Path': path_str[:100] + '...' if len(path_str) > 100 else path_str
        })
    
    return pd.DataFrame(table_data)

def create_edge_usage_chart(results_list):
    """
    Show which edges are used most frequently across all paths
    """
    edge_usage = {}
    
    for result in results_list:
        if result['path']:
            for node in result['path']:
                edge_id = f"e{node.original_edge_id}"
                edge_usage[edge_id] = edge_usage.get(edge_id, 0) + 1
    
    if not edge_usage:
        return None
    
    # Get top 20 most used edges
    sorted_edges = sorted(edge_usage.items(), key=lambda x: x[1], reverse=True)[:20]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[e[0] for e in sorted_edges],
        y=[e[1] for e in sorted_edges],
        marker=dict(
            color=[e[1] for e in sorted_edges],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Usage Count')
        ),
        text=[e[1] for e in sorted_edges],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Used in %{y} path(s)<extra></extra>'
    ))
    
    fig.update_layout(
        title='Top 20 Most Used Edges Across All Paths',
        xaxis_title='Edge ID',
        yaxis_title='Usage Count',
        template='plotly_white',
        height=400,
        xaxis=dict(tickangle=-45)
    )
    
    return fig
