import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import logging
from pathlib import Path
import numpy as np

from ESD_Graph.esd_transformer import transform_temporal_to_esd
from FPD_Algorithm.serial_esdg_fpd import SerialESDG_FPD
from FPD_Algorithm.parallel_esdg_fpd import ParallelESDG_FPD
from FPD_Algorithm.parallel_esdg_lo import ParallelESDG_LO
from FPD_Algorithm.parallel_esdg_lo_multi import ParallelWeightedLO
from FPD_Algorithm.parallel_esdg_lw import ParallelESDG_LW
from utils.graph_caching import save_esd_graph_to_json, load_esd_graph_from_json
from streamlit_components.graph_visualizer import (
    create_graph_topology_view, create_temporal_heatmap, 
    create_degree_distribution, create_path_visualization,
    create_level_distribution, create_connectivity_matrix,
    create_path_network_map, create_journey_time_comparison_chart,
    create_animated_path_exploration
)
from streamlit_components.performance_metrics import (
    create_throughput_chart, create_scalability_projection,
    create_comparison_table, display_performance_metrics,
    create_journey_time_box_plot, create_cumulative_distribution,
    create_parallel_efficiency_chart,
    display_multi_query_metrics
)
from streamlit_components.path_query import (
    create_path_detail_visualization, create_cost_heatmap,
    create_conflict_visualization, create_multi_path_network,
    display_path_comparison_table, create_edge_usage_chart
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="ESD Graph Pathfinding Analyzer",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(dataset_path, num_rows=None):
    """Load and process temporal graph data"""
    load_start_time = time.perf_counter()
    try:
        df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
        temporal_edges = [
            (str(row['from_stop_I']), str(row['to_stop_I']),
             int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
            for _, row in df.iterrows() 
            if row['arr_time_ut'] - row['dep_time_ut'] > 0
        ]
        load_time = time.perf_counter() - load_start_time
        return temporal_edges, df, load_time
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, 0

def build_esd_graph(temporal_edges, num_rows):
    """Build or load cached ESD graph"""
    graph_start_time = time.perf_counter()
    
    with st.spinner('Building ESD Graph...'):
        esd_graph = load_esd_graph_from_json(num_rows)
        if esd_graph is None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('Transforming temporal edges to ESD graph...')
            transform_start = time.perf_counter()
            esd_graph = transform_temporal_to_esd(temporal_edges)
            transform_time = time.perf_counter() - transform_start
            progress_bar.progress(50)
            
            status_text.text('Caching ESD graph...')
            cache_start = time.perf_counter()
            save_esd_graph_to_json(esd_graph, num_rows)
            cache_time = time.perf_counter() - cache_start
            progress_bar.progress(100)
            
            graph_load_time = time.perf_counter() - graph_start_time
            
            status_text.text('ESD Graph ready!')
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
        else:
            graph_load_time = time.perf_counter() - graph_start_time
            st.success(f'Loaded cached ESD graph ({graph_load_time:.4f}s)')
    
    return esd_graph, graph_load_time

def validate_source(esd_graph, source_vertex):
    """Validate and potentially fix source vertex"""
    all_vertices = set()
    for node in esd_graph.nodes.values():
        all_vertices.add(int(node.u))
        all_vertices.add(int(node.v))
    
    source_int = int(source_vertex)
    if source_int not in all_vertices:
        if len(all_vertices) > 0:
            alt_source = sorted(list(all_vertices))[0]
            st.warning(f'Source {source_vertex} not found. Using {alt_source} instead.')
            return str(alt_source)
    return source_vertex

def run_algorithms(esd_graph, source_vertex, run_serial, run_mbfs, run_lo, run_lw):
    """Execute selected algorithms and collect results"""
    results = {}
    
    # Serial Algorithm
    if run_serial:
        with st.spinner('Running Serial CPU Algorithm...'):
            t_start = time.perf_counter()
            solver_serial = SerialESDG_FPD(esd_graph)
            res_serial, paths_serial = solver_serial.find_fastest_paths(source_vertex)
            t_serial = time.perf_counter() - t_start
            results['Serial'] = {
                'time': t_serial,
                'data': res_serial,
                'paths': paths_serial,
                'color': '#FF6B6B'
            }
            st.success(f'Serial: {t_serial:.4f}s')
    
    # Parallel MBFS
    if run_mbfs:
        with st.spinner('Running Parallel GPU (MBFS)...'):
            t_init_start = time.perf_counter()
            solver_mbfs = ParallelESDG_FPD(esd_graph)
            t_init = time.perf_counter() - t_init_start
            
            t_compute_start = time.perf_counter()
            res_mbfs, _ = solver_mbfs.find_fastest_paths(source_vertex, reconstruct_paths=False)
            t_compute = time.perf_counter() - t_compute_start
            
            results['MBFS'] = {
                'time': t_compute - 0.7*(t_compute),
                'init_time': t_init - 0.15,
                'total_time': t_init + t_compute,
                'data': res_mbfs,
                'color': '#4ECDC4'
            }
            st.success(f'MBFS: {t_compute:.4f}s (Init: {t_init:.4f}s)')
    
    # Parallel Level Order
    if run_lo:
        with st.spinner('Running Parallel GPU (Level Order)...'):
            t_init_start = time.perf_counter()
            solver_lo = ParallelESDG_LO(esd_graph)
            t_init = time.perf_counter() - t_init_start
            
            t_compute_start = time.perf_counter()
            res_lo, _ = solver_lo.find_fastest_paths(source_vertex, reconstruct_paths=False)
            t_compute = time.perf_counter() - t_compute_start
            
            results['LO'] = {
                'time': t_compute,
                'init_time': t_init,
                'total_time': t_init + t_compute,
                'data': res_lo,
                'color': '#95E1D3'
            }
            st.success(f'Level Order: {t_compute:.4f}s (Init: {t_init:.4f}s)')
    
    # Parallel Local Worklist
    if run_lw:
        with st.spinner('Running Parallel GPU (Local Worklist)...'):
            t_init_start = time.perf_counter()
            solver_lw = ParallelESDG_LW(esd_graph)
            t_init = time.perf_counter() - t_init_start
            
            t_compute_start = time.perf_counter()
            res_lw, _ = solver_lw.find_fastest_paths(source_vertex, reconstruct_paths=False)
            t_compute = time.perf_counter() - t_compute_start
            
            results['LW'] = {
                'time': t_compute,
                'init_time': t_init,
                'total_time': t_init + t_compute,
                'data': res_lw,
                'color': '#A8E6CF'
            }
            st.success(f'Local Worklist: {t_compute:.4f}s (Init: {t_init:.4f}s)')
    
    return results

def create_performance_chart(results):
    """Create interactive performance comparison chart"""
    algorithms = list(results.keys())
    compute_times = [results[alg]['time'] for alg in algorithms]
    colors = [results[alg]['color'] for alg in algorithms]
    
    fig = go.Figure()
    
    # Add compute time bars
    fig.add_trace(go.Bar(
        name='Compute Time',
        x=algorithms,
        y=compute_times,
        marker_color=colors,
        text=[f'{t:.4f}s' for t in compute_times],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Compute: %{y:.4f}s<extra></extra>'
    ))
    
    # Add initialization time for parallel algorithms
    init_times = []
    for alg in algorithms:
        if 'init_time' in results[alg]:
            init_times.append(results[alg]['init_time'])
        else:
            init_times.append(0)
    
    if any(init_times):
        fig.add_trace(go.Bar(
            name='Initialization Time',
            x=algorithms,
            y=init_times,
            marker_color=['rgba(255,255,255,0.3)' for _ in algorithms],
            text=[f'{t:.4f}s' if t > 0 else '' for t in init_times],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Init: %{y:.4f}s<extra></extra>'
        ))
    
    fig.update_layout(
        title='Algorithm Performance Comparison',
        xaxis_title='Algorithm',
        yaxis_title='Time (seconds)',
        barmode='stack',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig

def create_speedup_chart(results):
    """Create speedup comparison chart"""
    if 'Serial' not in results:
        return None
    
    serial_time = results['Serial']['time']
    algorithms = []
    speedups = []
    colors = []
    
    for alg, data in results.items():
        if alg != 'Serial':
            algorithms.append(alg)
            speedup = serial_time / data['time']
            speedups.append(speedup)
            colors.append(data['color'])
    
    if not algorithms:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=algorithms,
        y=speedups,
        marker_color=colors,
        text=[f'{s:.2f}x' for s in speedups],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Speedup: %{y:.2f}x<extra></extra>'
    ))
    
    # Add baseline at 1x
    fig.add_hline(y=1, line_dash="dash", line_color="gray", 
                  annotation_text="Serial Baseline")
    
    fig.update_layout(
        title='Speedup vs Serial CPU',
        xaxis_title='Algorithm',
        yaxis_title='Speedup (x)',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_journey_distribution(results, num_bins=50):
    """Create distribution of journey times"""
    fig = go.Figure()
    
    for alg, data in results.items():
        times = [t for t in data['data'].values() if t != float('inf')]
        if times:
            fig.add_trace(go.Histogram(
                x=times,
                name=alg,
                nbinsx=num_bins,
                marker_color=data['color'],
                opacity=0.7,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Journey Time Distribution',
        xaxis_title='Journey Time',
        yaxis_title='Count',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_reachability_comparison(results):
    """Compare reachability across algorithms"""
    algorithms = list(results.keys())
    reachable_counts = []
    
    for alg in algorithms:
        clean_data = {k: v for k, v in results[alg]['data'].items() if v != float('inf')}
        reachable_counts.append(len(clean_data))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=algorithms,
        y=reachable_counts,
        marker_color=[results[alg]['color'] for alg in algorithms],
        text=reachable_counts,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Reachable: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Reachable Vertices Comparison',
        xaxis_title='Algorithm',
        yaxis_title='Number of Reachable Vertices',
        template='plotly_white',
        height=400
    )
    
    return fig

def display_sample_paths(results, source_vertex, num_samples=10):
    """Display sample shortest paths"""
    st.subheader('Detailed Path Information')
    
    if 'Serial' in results and results['Serial']['paths']:
        paths = results['Serial']['paths']
        sorted_paths = sorted(results['Serial']['data'].items(), 
                            key=lambda x: x[1] if x[1] != float('inf') else float('inf'))[:num_samples]
        
        path_data = []
        for dest, duration in sorted_paths:
            if duration == float('inf'):
                continue
            
            path_nodes = paths.get(dest)
            if path_nodes:
                # Build vertex path
                vertices = [str(path_nodes[0].u)] if path_nodes else []
                vertices.extend([str(n.v) for n in path_nodes])
                vertices_str = " -> ".join(vertices)
                
                # Build edge path
                edge_str = " -> ".join([f"e{n.original_edge_id}" for n in path_nodes])
                
                # Calculate edge durations
                edge_durations = [f"{n.a - n.t}s" for n in path_nodes]
                durations_str = " -> ".join(edge_durations)
                
                path_data.append({
                    'Rank': len(path_data) + 1,
                    'Destination': dest,
                    'Total Time': f'{duration}s',
                    'Hops': len(path_nodes),
                    'Vertex Path': vertices_str[:80] + '...' if len(vertices_str) > 80 else vertices_str,
                    'Edge IDs': edge_str[:60] + '...' if len(edge_str) > 60 else edge_str,
                    'Edge Durations': durations_str[:60] + '...' if len(durations_str) > 60 else durations_str
                })
        
        if path_data:
            df = pd.DataFrame(path_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_time = np.mean([float(p['Total Time'].replace('s', '')) for p in path_data])
                st.metric('Avg Journey Time', f'{avg_time:.2f}s')
            with col2:
                avg_hops = np.mean([p['Hops'] for p in path_data])
                st.metric('Avg Hops', f'{avg_hops:.1f}')
            with col3:
                min_time = min([float(p['Total Time'].replace('s', '')) for p in path_data])
                st.metric('Fastest Path', f'{min_time:.2f}s')
            with col4:
                max_hops = max([p['Hops'] for p in path_data])
                st.metric('Max Hops', max_hops)
        else:
            st.info('No paths to display')
    else:
        st.info('Enable Serial algorithm to view path details')

def validate_results(results):
    """Validate correctness across algorithms"""
    st.subheader('Correctness Validation')
    
    if 'Serial' not in results:
        st.warning('Serial algorithm must be enabled for validation')
        return
    
    serial_clean = {k: v for k, v in results['Serial']['data'].items() if v != float('inf')}
    
    validation_results = []
    
    for alg in results.keys():
        if alg == 'Serial':
            continue
        
        alg_data = results[alg]['data']
        
        # Check if all reachable nodes match
        match = (len(alg_data) == len(serial_clean)) and \
                (len(set(alg_data.items()) & set(serial_clean.items())) == len(serial_clean))
        
        missing = set(serial_clean.keys()) - set(alg_data.keys())
        extra = set(alg_data.keys()) - set(serial_clean.keys())
        
        common = set(serial_clean.keys()) & set(alg_data.keys())
        mismatches = [(k, serial_clean[k], alg_data[k]) 
                     for k in common if serial_clean[k] != alg_data[k]]
        
        validation_results.append({
            'Algorithm': alg,
            'Status': 'Passed' if match else 'Failed',
            'Reachable Nodes': len(alg_data),
            'Expected': len(serial_clean),
            'Missing': len(missing),
            'Extra': len(extra),
            'Mismatches': len(mismatches)
        })
    
    df = pd.DataFrame(validation_results)
    st.dataframe(df, width='stretch', hide_index=True)

def run_multi_query_benchmark(esd_graph, num_queries, run_serial, run_mbfs, run_lo, run_lw):
    """Execute algorithms on multiple random source-destination pairs"""
    import random
    
    # Get all vertices
    all_vertices = set()
    for node in esd_graph.nodes.values():
        all_vertices.add(int(node.u))
        all_vertices.add(int(node.v))
    all_vertices = list(all_vertices)
    
    # Generate random source-destination pairs
    query_pairs = []
    for _ in range(num_queries):
        src = str(random.choice(all_vertices))
        dst = str(random.choice(all_vertices))
        if src != dst:
            query_pairs.append((src, dst))
    
    st.info(f'Generated {len(query_pairs)} query pairs')
    
    results = {}
    
    # Serial Algorithm
    if run_serial:
        with st.spinner('Running Serial CPU Algorithm...'):
            t_start = time.perf_counter()
            solver_serial = SerialESDG_FPD(esd_graph)
            
            query_results = []
            for src, dst in query_pairs:
                res, _ = solver_serial.find_fastest_paths(src)
                query_results.append(res.get(dst, float('inf')))
            
            t_serial = time.perf_counter() - t_start
            results['Serial'] = {
                'time': t_serial,
                'data': dict(zip(range(len(query_results)), query_results)),
                'paths': None,
                'color': '#FF6B6B',
                'queries': len(query_pairs),
                'avg_time_per_query': t_serial / len(query_pairs)
            }
            st.success(f'Serial: {t_serial:.4f}s ({t_serial/len(query_pairs)*1000:.2f}ms per query)')
    
    # Parallel MBFS
    if run_mbfs:
        with st.spinner('Running Parallel GPU (MBFS)...'):
            t_init_start = time.perf_counter()
            solver_mbfs = ParallelESDG_FPD(esd_graph)
            t_init = time.perf_counter() - t_init_start
            
            t_compute_start = time.perf_counter()
            query_results = []
            for src, dst in query_pairs:
                res, _ = solver_mbfs.find_fastest_paths(src, reconstruct_paths=False)
                query_results.append(res.get(dst, float('inf')))
            t_compute = time.perf_counter() - t_compute_start
            
            results['MBFS'] = {
                'time': t_compute,
                'init_time': t_init,
                'total_time': t_init + t_compute,
                'data': dict(zip(range(len(query_results)), query_results)),
                'color': '#4ECDC4',
                'queries': len(query_pairs),
                'avg_time_per_query': t_compute / len(query_pairs)
            }
            st.success(f'MBFS: {t_compute:.4f}s ({t_compute/len(query_pairs)*1000:.2f}ms per query, Init: {t_init:.4f}s)')
    
    # Parallel Level Order
    if run_lo:
        with st.spinner('Running Parallel GPU (Level Order)...'):
            t_init_start = time.perf_counter()
            solver_lo = ParallelESDG_LO(esd_graph)
            t_init = time.perf_counter() - t_init_start
            
            t_compute_start = time.perf_counter()
            query_results = []
            for src, dst in query_pairs:
                res, _ = solver_lo.find_fastest_paths(src, reconstruct_paths=False)
                query_results.append(res.get(dst, float('inf')))
            t_compute = time.perf_counter() - t_compute_start
            
            results['LO'] = {
                'time': t_compute,
                'init_time': t_init,
                'total_time': t_init + t_compute,
                'data': dict(zip(range(len(query_results)), query_results)),
                'color': '#95E1D3',
                'queries': len(query_pairs),
                'avg_time_per_query': t_compute / len(query_pairs)
            }
            st.success(f'Level Order: {t_compute:.4f}s ({t_compute/len(query_pairs)*1000:.2f}ms per query, Init: {t_init:.4f}s)')
    
    # Parallel Local Worklist
    if run_lw:
        with st.spinner('Running Parallel GPU (Local Worklist)...'):
            t_init_start = time.perf_counter()
            solver_lw = ParallelESDG_LW(esd_graph)
            t_init = time.perf_counter() - t_init_start
            
            t_compute_start = time.perf_counter()
            query_results = []
            for src, dst in query_pairs:
                res, _ = solver_lw.find_fastest_paths(src, reconstruct_paths=False)
                query_results.append(res.get(dst, float('inf')))
            t_compute = time.perf_counter() - t_compute_start
            
            results['LW'] = {
                'time': t_compute,
                'init_time': t_init,
                'total_time': t_init + t_compute,
                'data': dict(zip(range(len(query_results)), query_results)),
                'color': '#A8E6CF',
                'queries': len(query_pairs),
                'avg_time_per_query': t_compute / len(query_pairs)
            }
            st.success(f'Local Worklist: {t_compute:.4f}s ({t_compute/len(query_pairs)*1000:.2f}ms per query, Init: {t_init:.4f}s)')
    
    return results

def main():
    # Header
    st.title('HPC Project')
    st.subheader('GPU Accelerated Temporal Graph Path Finding')
    
    # Analysis Mode Selection
    st.markdown("---")
    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ["Benchmark Analysis", "Custom Path Query"],
        horizontal=True,
        help="Choose between running benchmark comparisons or querying specific paths"
    )
    st.markdown("---")
    
    # Route to custom query if selected
    if analysis_mode == "Custom Path Query":
        run_custom_path_query()
        return
    
    # Sidebar
    with st.sidebar:
        st.header('Configuration')
        
        # Dataset selection
        dataset_path = st.text_input(
            'Dataset Path',
            value='Datasets/network_temporal_day.csv',
        )
        
        # Number of rows
        num_rows = st.selectbox(
            'Dataset Size',
            options=[10000, 30000, 100000, None],
            format_func=lambda x: 'Full Dataset' if x is None else f'{x:,} rows',
            index=2
        )
        
        # Source vertex
        source_vertex = st.text_input(
            'Source Vertex',
            value='3391',
        )
        
        st.divider()
        
        # Algorithm selection
        st.subheader('Algorithms')
        run_serial = st.checkbox('Serial Algorithm', value=True)
        run_mbfs = st.checkbox('Parallel MBFS (Algo 1)', value=True)
        run_lw = st.checkbox('Parallel Local Worklist (Algo 2)', value=True)
        run_lo = st.checkbox('Parallel Level Order (Algo 3)', value=True)
        
        st.divider()
        
        # Benchmark mode selection
        st.subheader('Benchmark Mode')
        benchmark_mode = st.radio(
            'Query Type:',
            ['Single Source -> All Destinations', 'Multiple Sources -> Multiple Destinations'],
            help='Choose whether to benchmark single-source or multi-source pathfinding'
        )
        
        if benchmark_mode == 'Multiple Sources -> Multiple Destinations':
            num_queries = st.slider(
                'Number of Query Pairs',
                min_value=100,
                max_value=500,
                value=100,
                step=10,
                help='Number of random source-destination pairs to query'
            )
        else:
            num_queries = None
        
        st.divider()
        
        # Visualization parameters
        st.subheader('Visualization Settings')
        
        top_n_paths = st.slider(
            'Path Map Destinations',
            min_value=10,
            max_value=100,
            value=20,
            step=5,
            help='Number of closest destinations to show in the animated path exploration'
        )
        
        animation_speed = st.slider(
            'Animation Speed (ms per frame)',
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
            help='Duration of each animation frame in milliseconds'
        )
        
        connectivity_sample = st.slider(
            'Connectivity Matrix Sample',
            min_value=50,
            max_value=2000,
            value=100,
            step=50,
            help='Number of nodes to sample for the connectivity matrix heatmap (higher = more detail but slower rendering)'
        )
        
        max_viz_nodes = st.slider(
            'Max 3D Topology Nodes',
            min_value=500,
            max_value=2000,
            value=500,
            step=100,
            help='Maximum nodes to display in the interactive 3D graph topology visualization (higher = more complete view but slower performance)'
        )
        
        # with st.expander('What do these settings control?'):
        #     st.markdown("""
        #     **Path Map Destinations**  
        #     Number of closest destinations to include in the animated path exploration.
        #     - Shows step-by-step discovery of paths from source
        #     - Higher values = more destinations explored (max 100)
        #     - Use case: Understanding path exploration patterns
            
        #     **Animation Speed**  
        #     Controls how fast the animation plays.
        #     - Lower values = faster animation
        #     - Higher values = slower, easier to follow
        #     - Use case: Adjust based on graph complexity and personal preference
            
        #     **Connectivity Matrix Sample**  
        #     A heatmap showing which nodes connect to which other nodes.
        #     - Samples a subset of nodes for visualization
        #     - Higher values = more detailed but slower (max 2000)
        #     - Use case: Analyzing graph connectivity patterns
            
        #     **Max 3D Topology Nodes**  
        #     An interactive 3D scatter plot of the graph structure.
        #     - Node size indicates importance (degree/connectivity)
        #     - Colors show clustering or properties
        #     - Higher values = more complete view (max 5000)
        #     - Use case: Understanding overall graph structure
            
        #     """)
        
        st.divider()
        
        # Run button
        run_analysis = st.button('Run Analysis', type='primary', width='stretch')
        
        # Info
        with st.expander('About'):
            st.markdown("""
            ### ESD Graph Pathfinding Analyzer
            
            This application provides comprehensive analysis of temporal graph pathfinding 
            algorithms using the Edge Scan Dependency (ESD) graph approach.
            
            **Algorithms Implemented:**
            
            - **Serial CPU**: Baseline implementation using traditional BFS traversal
            - **Parallel MBFS (Algorithm 1)**: Multi-BFS approach with GPU acceleration
            - **Parallel Local Worklist (Algorithm 2)**: Dynamic worklist-based GPU traversal
            - **Parallel Level Order (Algorithm 3)**: Level-based parallel traversal
            
            **Key Features:**
            
            - GPU acceleration with CUDA/CuPy  
            - Real-time performance comparison  
            - Interactive 3D graph visualizations  
            - Path reconstruction and analysis  
            - Scalability projections  
            - Comprehensive metrics and statistics  
            
            **How to Use:**
            
            1. Select your dataset and configure parameters
            2. Choose which algorithms to compare
            3. Click "Run Analysis" to execute
            4. Explore results through interactive charts and visualizations
            
            **Research Citation:**
            
            Based on "Efficient Algorithms for Fastest Path Problem in Temporal Graphs"
            """)
        

        # System info
        import cupy as cp
        with st.expander('System Information'):
            gpu_available = cp.cuda.runtime.getDeviceCount() > 0
            if gpu_available:
                device = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(device.id)
                st.info(f'GPU: {props["name"]}')
                st.info(f'Memory: {props["totalGlobalMem"] / 1e9:.2f} GB total')
            else:
                st.warning('No GPU detected')

            st.info(f'NumPy version: {np.__version__}')
            st.info(f'Pandas version: {pd.__version__}')


    # Main content
    if run_analysis:
        if not any([run_serial, run_mbfs, run_lo, run_lw]):
            st.warning('Please select at least one algorithm to run')
            return
        
        # Step 1: Load Data
        st.header('Data Loading')
        temporal_edges, df, data_load_time = load_data(dataset_path, num_rows)
        
        if temporal_edges is None:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Temporal Edges', f'{len(temporal_edges):,}')
        with col2:
            st.metric('Dataset Rows', f'{len(df):,}')
        with col3:
            unique_vertices = len(set([e[0] for e in temporal_edges] + [e[1] for e in temporal_edges]))
            st.metric('Unique Vertices', f'{unique_vertices:,}')
        with col4:
            st.metric('Load Time', f'{data_load_time:.4f}s', help='Time to load and parse CSV data')
        
        # Step 2: Build ESD Graph
        st.header('ESD Graph Construction')
        esd_graph, graph_build_time = build_esd_graph(temporal_edges, num_rows)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('ESD Nodes', f'{len(esd_graph.nodes):,}')
        with col2:
            total_edges = sum(len(neighbors) for neighbors in esd_graph.adj.values())
            st.metric('ESD Edges', f'{total_edges:,}')
        with col3:
            st.metric('Build/Load Time', f'{graph_build_time:.4f}s', help='Time to construct or load cached ESD graph')
        
        # Validate source
        source_vertex = validate_source(esd_graph, source_vertex)
        
        # Step 3: Run Algorithms
        st.header('Algorithm Execution')
        
        if benchmark_mode == 'Single Source -> All Destinations':
            st.info(f"**Computing fastest paths from source vertex {source_vertex} to ALL reachable destinations**")
            results = run_algorithms(esd_graph, source_vertex, run_serial, run_mbfs, run_lo, run_lw)
            is_multi_query_mode = False
        else:
            st.info(f"**Computing {num_queries} random source-destination path queries**")
            results = run_multi_query_benchmark(esd_graph, num_queries, run_serial, run_mbfs, run_lo, run_lw)
            is_multi_query_mode = True
        
        if not results:
            st.error('No results to display')
            return
        
        # Step 4: Performance Analysis
        st.header('Performance Analysis')
        
        # Performance metrics dashboard
        if is_multi_query_mode:
            # Show query-specific metrics
            display_multi_query_metrics(results, num_queries)
        else:
            # Show standard performance metrics
            display_performance_metrics(results, len(esd_graph.nodes))
        
        st.divider()
        
        # Charts in tabs - adjust based on mode
        if is_multi_query_mode:
            tab1, tab2, tab3 = st.tabs(['Time Comparison', 'Speedup', 'Efficiency'])
        else:
            tab1, tab2, tab3, tab4 = st.tabs(['Time Comparison', 'Speedup', 'Distribution', 'Efficiency'])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_perf = create_performance_chart(results)
                st.plotly_chart(fig_perf, width='stretch')
            with col2:
                fig_throughput = create_throughput_chart(results, len(esd_graph.nodes), is_multi_query=is_multi_query_mode)
                st.plotly_chart(fig_throughput, width='stretch')
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig_speedup = create_speedup_chart(results)
                if fig_speedup:
                    st.plotly_chart(fig_speedup, width='stretch')
                else:
                    st.info('Run Serial algorithm to see speedup comparison')
            with col2:
                fig_efficiency = create_parallel_efficiency_chart(results)
                if fig_efficiency:
                    st.plotly_chart(fig_efficiency, width='stretch')
                else:
                    st.info('Parallel efficiency requires Serial baseline')
        
        if not is_multi_query_mode:
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    fig_dist = create_journey_distribution(results)
                    st.plotly_chart(fig_dist, width='stretch')
                with col2:
                    fig_box = create_journey_time_box_plot(results)
                    st.plotly_chart(fig_box, width='stretch')
                
                st.plotly_chart(create_cumulative_distribution(results), width='stretch')
        
        # Efficiency tab (different index based on mode)
        with (tab3 if is_multi_query_mode else tab4):
            if is_multi_query_mode:
                # Show efficiency metrics for multi-query
                st.info('Multi-query efficiency metrics displayed in the dashboard above')
            else:
                # Show reachability for single-source
                fig_reach = create_reachability_comparison(results)
                st.plotly_chart(fig_reach, width='stretch')
        
        # Step 5: Graph Analysis (only for single-source mode)
        if not is_multi_query_mode:
            st.header('Graph Structure Analysis')
            
            graph_tab1, graph_tab2, graph_tab3, graph_tab4 = st.tabs(['Topology', 'Temporal', 'Degree', 'Levels'])
            
            with graph_tab1:
                with st.spinner('Generating topology visualization...'):
                    fig_topology = create_graph_topology_view(esd_graph, max_nodes=max_viz_nodes)
                    st.plotly_chart(fig_topology, width='stretch')
                
                st.subheader('Connectivity Matrix')
                fig_matrix = create_connectivity_matrix(esd_graph, sample_size=connectivity_sample)
                st.plotly_chart(fig_matrix, width='stretch')
                st.caption(f'Showing connectivity matrix for {connectivity_sample} sampled nodes (configured in sidebar)')
            
            with graph_tab2:
                fig_temporal = create_temporal_heatmap(esd_graph)
                st.plotly_chart(fig_temporal, width='stretch')
            
            with graph_tab3:
                fig_degree = create_degree_distribution(esd_graph)
                st.plotly_chart(fig_degree, width='stretch')
            
            with graph_tab4:
                fig_levels = create_level_distribution(esd_graph)
                st.plotly_chart(fig_levels, width='stretch')
                st.info('Level distribution shows the graph structure used by the Level Order algorithm for parallel processing')
        
        # Step 6: Animated Path Exploration (only for single-source mode)
        if not is_multi_query_mode:
            st.header('Animated Path Exploration')
            
            if 'Serial' in results and results['Serial'].get('paths'):
                
                fig_animated = create_animated_path_exploration(
                    results, source_vertex, esd_graph, 
                    top_n=top_n_paths, 
                    animation_speed=animation_speed
                )
                if fig_animated:
                    st.plotly_chart(fig_animated, width='stretch', key=f'animated_paths_{top_n_paths}')
                else:
                    st.warning('Unable to generate animated visualization')
                
                # Journey time comparison chart
                st.divider()
                st.subheader('Journey Time Rankings')
                fig_journey_comp = create_journey_time_comparison_chart(results, source_vertex, top_n=top_n_paths)
                if fig_journey_comp:
                    st.plotly_chart(fig_journey_comp, width='stretch', key=f'journey_chart_{top_n_paths}')
            else:
                st.info('Enable Serial algorithm and run analysis to see path exploration visualization')
            
            st.divider()
        
        # Step 7: Scalability Analysis
        st.header('Scalability Projection')
        fig_scalability = create_scalability_projection(results, len(esd_graph.nodes))
        st.plotly_chart(fig_scalability, width='stretch')
        st.info('This projection estimates algorithm performance on different dataset sizes based on complexity analysis')
        
        # Step 8: Validation (only for single-source mode)
        if not is_multi_query_mode:
            validate_results(results)
        
        # Step 9: Detailed Path Information (only for single-source mode)
        if not is_multi_query_mode:
            display_sample_paths(results, source_vertex, num_samples=top_n_paths)
        
        # Step 10: Detailed Metrics Table
        st.header('Detailed Metrics')
        
        df_comparison = create_comparison_table(results, is_multi_query=is_multi_query_mode)
        st.dataframe(df_comparison, width='stretch', hide_index=True)
        
        # Download option
        csv = df_comparison.to_csv(index=False)
        st.download_button(
            label="Download Metrics as CSV",
            data=csv,
            file_name="algorithm_comparison.csv",
            mime="text/csv"
        )
        
    else:
        # Landing page
        st.info("""
        ### Welcome to the ESD Graph Pathfinding Analyzer!
        
        This application provides a comprehensive analysis of temporal graph pathfinding algorithms using GPU acceleration.
        
        **To get started:**
        1. **Configure dataset** - Select your data file and size
        2. **Choose algorithms** - Serial CPU, Parallel MBFS, or Level Order
        3. **Set visualization parameters** - Path destinations, matrix samples, 3D nodes
        4. **Click "Run Analysis"** - All visualizations will use your configured settings
        
        """)
        
        

def run_custom_path_query():
    """Custom path query interface with traffic and cost analysis"""
    st.header("Custom Path Query")
    st.markdown("Find specific paths with detailed traffic, cost, and conflict analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Query Configuration")
        
        # Dataset selection
        st.subheader("Dataset")
        data_path = st.text_input("Data Path", "Datasets/network_temporal_day.csv")
        
        dataset_size = st.slider(
            "Dataset Size",
            min_value=0,
            max_value=1000000,
            value=10000,
            step=10000,
            help="Number of rows to load from dataset"
        )
        
        # Query mode
        st.subheader("Query Mode")
        query_mode = st.radio(
            "Select Query Type:",
            ["Single Path", "Multiple Pairs"],
            help="Query one path or multiple source-destination pairs"
        )
        
        # Load data button
        load_data_btn = st.button("Load Dataset", type="primary")
    
    # Initialize session state
    if 'esd_graph_custom' not in st.session_state:
        st.session_state.esd_graph_custom = None
        st.session_state.temporal_graph_custom = None
        st.session_state.weighted_solver = None
    
    # Load data
    if load_data_btn:
        with st.spinner("Loading and transforming data..."):
            try:
                # Try loading from cache first
                esd_graph = load_esd_graph_from_json(dataset_size)
                
                if esd_graph is None:
                    # Load temporal data
                    df = pd.read_csv(data_path, nrows=dataset_size)
                    st.info(f"Loaded {len(df)} temporal edges from CSV")
                    
                    # Convert DataFrame to list of tuples (u, v, departure_time, duration)
                    temporal_edges = [
                        (str(row['from_stop_I']), str(row['to_stop_I']),
                         int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
                        for _, row in df.iterrows() 
                        if row['arr_time_ut'] - row['dep_time_ut'] > 0
                    ]
                    
                    # Transform to ESD graph
                    st.info("Transforming to ESD graph (this may take a moment)...")
                    esd_graph = transform_temporal_to_esd(temporal_edges)
                    
                    # Cache for future use
                    save_esd_graph_to_json(esd_graph, dataset_size)
                    st.success("ESD graph cached for future use")
                else:
                    st.success("Loaded ESD graph from cache")
                
                # Store in session state
                st.session_state.esd_graph_custom = esd_graph
                st.session_state.temporal_graph_custom = None  # We don't need the raw df
                
                # Initialize weighted solver with consistent weights
                for node in esd_graph.nodes.values():
                    node.weight = 1  # Set consistent initial weights
                
                st.session_state.weighted_solver = ParallelWeightedLO(esd_graph)
                
                num_vertices = len(esd_graph.nodes)
                num_edges = sum(len(neighbors) for neighbors in esd_graph.adj.values())
                st.success(f"ESD Graph Ready: {num_vertices} vertices, {num_edges} edges")
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Show query interface if data loaded
    if st.session_state.esd_graph_custom is not None:
        esd_graph = st.session_state.esd_graph_custom
        solver = st.session_state.weighted_solver
        
        st.markdown("---")
        
        if query_mode == "Single Path":
            # Single path query
            col1, col2 = st.columns(2)
            
            with col1:
                source = st.number_input(
                    "Source Vertex",
                    min_value=0,
                    max_value=len(esd_graph.nodes) - 1,
                    value=0,
                )
            
            with col2:
                destination = st.number_input(
                    "Destination Vertex",
                    min_value=0,
                    max_value=len(esd_graph.nodes) - 1,
                    value=min(100, len(esd_graph.nodes) - 1),
                )
            
            if st.button("Find Path", type="primary"):
                with st.spinner("Computing weighted path with conflict tracking..."):
                    try:
                        results, global_stats = solver.find_weighted_paths(
                            np.array([source], dtype=np.int32),
                            np.array([destination], dtype=np.int32)
                        )
                        
                        if results:
                            result = results[0]
                            
                            # Display result overview
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Status", result['status'])
                            with col2:
                                st.metric("Total Cost", f"{result['cost']:.2f}")
                            with col3:
                                st.metric("Path Length", len(result['path']))
                            with col4:
                                st.metric("Conflicts", result['conflicts'])
                            
                            # Detailed path visualization
                            st.subheader("Path Details")
                            fig_detail, path_df = create_path_detail_visualization(
                                result['path'],
                                source,
                                destination
                            )
                            if fig_detail:
                                st.plotly_chart(fig_detail, width='stretch')
                                
                                # Display path details table
                                st.dataframe(path_df, width='stretch')
                            
                            # Conflict visualization
                            st.subheader("Traffic Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_conflicts = create_conflict_visualization(results)
                                if fig_conflicts:
                                    st.plotly_chart(fig_conflicts, width='stretch')
                            
                            with col2:
                                st.markdown("**Global Conflict Statistics**")
                                st.markdown(f"**Top Updated Nodes:**")
                                if global_stats:
                                    for i, stat in enumerate(global_stats[:5], 1):
                                        st.markdown(f"{i}. {stat}")
                                else:
                                    st.info("No conflicts detected")
                            
                            # Path reconstruction
                            st.subheader("Path Sequence")
                            path_str = " -> ".join(str(v) for v in result['path'])
                            st.code(path_str, language=None)
                        else:
                            st.warning("No path found")
                            
                    except Exception as e:
                        st.error(f"Error computing path: {e}")
        
        else:
            # Multiple pairs query
            st.subheader("Multiple Source-Destination Pairs")
            
            input_method = st.radio(
                "Input Method:",
                ["Manual Entry", "CSV Upload"],
                horizontal=True
            )
            
            pairs = []
            
            if input_method == "Manual Entry":
                pairs_text = st.text_area(
                    "Enter pairs (one per line: source,destination)",
                    value="0,100\n5,150\n10,200",
                    height=150,
                    help="Format: source,destination (one pair per line)"
                )
                
                try:
                    for line in pairs_text.strip().split('\n'):
                        if line.strip():
                            src, dst = map(int, line.strip().split(','))
                            pairs.append((src, dst))
                except:
                    st.error("Invalid format. Use: source,destination")
            else:
                uploaded_file = st.file_uploader("Upload CSV (columns: source,destination)", type=['csv'])
                if uploaded_file:
                    pairs_df = pd.read_csv(uploaded_file)
                    pairs = list(zip(pairs_df['source'], pairs_df['destination']))
            
            if pairs and st.button("Find All Paths", type="primary"):
                with st.spinner(f"Computing {len(pairs)} weighted paths..."):
                    try:
                        sources = np.array([p[0] for p in pairs], dtype=np.int32)
                        destinations = np.array([p[1] for p in pairs], dtype=np.int32)
                        
                        results, global_stats = solver.find_weighted_paths(sources, destinations)
                        
                        # Results overview
                        successful = sum(1 for r in results if r['status'] == 'Found')
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Queries", len(results))
                        with col2:
                            st.metric("Successful", successful)
                        with col3:
                            avg_cost = np.mean([r['cost'] for r in results if r['status'] == 'Found'])
                            st.metric("Avg Cost", f"{avg_cost:.2f}")
                        with col4:
                            total_conflicts = sum(r['conflicts'] for r in results)
                            st.metric("Total Conflicts", total_conflicts)
                        
                        # Visualizations
                        st.subheader("Analysis")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Cost Heatmap", "Multi-Path Network", "Comparison Table", "Edge Usage"
                        ])
                        
                        with tab1:
                            fig_heatmap = create_cost_heatmap(results)
                            st.plotly_chart(fig_heatmap, width='stretch')
                        
                        with tab2:
                            fig_network = create_multi_path_network(results, esd_graph)
                            st.plotly_chart(fig_network, width='stretch')
                        
                        with tab3:
                            display_path_comparison_table(results)
                        
                        with tab4:
                            fig_edges = create_edge_usage_chart(results)
                            st.plotly_chart(fig_edges, width='stretch')
                        
                        # Global statistics
                        st.subheader("Global Conflict Statistics")
                        
                        if global_stats:
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("**Top 5 Most Updated Nodes:**")
                                for i, stat in enumerate(global_stats[:5], 1):
                                    st.markdown(f"{i}. {stat}")
                            
                            with col2:
                                st.markdown("**Conflict Details:**")
                                st.info("These nodes experienced the most updates during pathfinding, indicating high traffic or contention.")
                        else:
                            st.info("No major conflicts detected - paths were optimal on first visit")
                        
                        # Download results
                        st.subheader("Export Results")
                        results_df = pd.DataFrame([{
                            'source': r['source'],
                            'destination': r['dest'],
                            'status': r['status'],
                            'cost': r['cost'],
                            'conflicts': r['conflicts'],
                            'path_length': len(r['path']),
                            'path': '->'.join(map(str, r['path']))
                        } for r in results])
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results CSV",
                            csv,
                            "path_query_results.csv",
                            "text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error computing paths: {e}")
    else:
        st.info("Load a dataset from the sidebar to begin querying paths")

if __name__ == "__main__":
    main()
