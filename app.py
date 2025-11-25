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
from utils.graph_caching import save_esd_graph_to_json, load_esd_graph_from_json
from streamlit_components.graph_visualizer import (
    create_graph_topology_view, create_temporal_heatmap, 
    create_degree_distribution, create_path_visualization,
    create_level_distribution, create_connectivity_matrix,
    create_path_network_map, create_journey_time_comparison_chart
)
from streamlit_components.performance_metrics import (
    create_throughput_chart, create_scalability_projection,
    create_comparison_table, display_performance_metrics,
    create_journey_time_box_plot, create_cumulative_distribution,
    create_parallel_efficiency_chart, create_efficiency_radar
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
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data(dataset_path, num_rows=None):
    """Load and process temporal graph data"""
    try:
        df = pd.read_csv(dataset_path, sep=',', nrows=num_rows)
        temporal_edges = [
            (str(row['from_stop_I']), str(row['to_stop_I']),
             int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
            for _, row in df.iterrows() 
            if row['arr_time_ut'] - row['dep_time_ut'] > 0
        ]
        return temporal_edges, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def build_esd_graph(temporal_edges, num_rows):
    """Build or load cached ESD graph"""
    with st.spinner('Building ESD Graph...'):
        esd_graph = load_esd_graph_from_json(num_rows)
        if esd_graph is None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('Transforming temporal edges to ESD graph...')
            esd_graph = transform_temporal_to_esd(temporal_edges)
            progress_bar.progress(50)
            
            status_text.text('Caching ESD graph...')
            save_esd_graph_to_json(esd_graph, num_rows)
            progress_bar.progress(100)
            
            status_text.text('✅ ESD Graph ready!')
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
        else:
            st.success('✅ Loaded cached ESD graph')
    
    return esd_graph

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
            st.warning(f'⚠️ Source {source_vertex} not found. Using {alt_source} instead.')
            return str(alt_source)
    return source_vertex

def run_algorithms(esd_graph, source_vertex, run_serial, run_mbfs, run_lo):
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
            st.success(f'✅ Serial: {t_serial:.4f}s')
    
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
                'time': t_compute,
                'init_time': t_init,
                'total_time': t_init + t_compute,
                'data': res_mbfs,
                'color': '#4ECDC4'
            }
            st.success(f'✅ MBFS: {t_compute:.4f}s (Init: {t_init:.4f}s)')
    
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
            st.success(f'✅ Level Order: {t_compute:.4f}s (Init: {t_init:.4f}s)')
    
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
    st.subheader('📍 Detailed Path Information')
    
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
                vertices_str = " → ".join(vertices)
                
                # Build edge path
                edge_str = " → ".join([f"e{n.original_edge_id}" for n in path_nodes])
                
                # Calculate edge durations
                edge_durations = [f"{n.a - n.t}s" for n in path_nodes]
                durations_str = " → ".join(edge_durations)
                
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
            st.dataframe(df, use_container_width=True, hide_index=True)
            
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
    st.subheader('✅ Correctness Validation')
    
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
            'Status': '✅ Passed' if match else '❌ Failed',
            'Reachable Nodes': len(alg_data),
            'Expected': len(serial_clean),
            'Missing': len(missing),
            'Extra': len(extra),
            'Mismatches': len(mismatches)
        })
    
    df = pd.DataFrame(validation_results)
    st.dataframe(df, use_container_width=True, hide_index=True)

def main():
    # Header
    st.markdown('<div class="main-header">🚀 ESD Graph Pathfinding Analyzer</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">GPU-Accelerated Temporal Graph Analysis</div>', 
                unsafe_allow_html=True)
    
    # Analysis Mode Selection
    st.markdown("---")
    analysis_mode = st.radio(
        "📊 Select Analysis Mode:",
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
        st.header('⚙️ Configuration')
        
        # Dataset selection
        dataset_path = st.text_input(
            'Dataset Path',
            value='Datasets/network_temporal_day.csv',
            help='Path to the temporal graph dataset'
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
            help='Starting vertex for pathfinding'
        )
        
        st.divider()
        
        # Algorithm selection
        st.subheader('🔬 Algorithms')
        run_serial = st.checkbox('Serial CPU', value=True)
        run_mbfs = st.checkbox('Parallel MBFS (Algo 1)', value=True)
        run_lo = st.checkbox('Parallel Level Order (Algo 3)', value=True)
        
        st.divider()
        
        # Visualization parameters (set before analysis)
        st.subheader('📊 Visualization Settings')
        st.caption('Configure these before running analysis to avoid reloads')
        
        top_n_paths = st.slider(
            'Path Map Destinations',
            min_value=5,
            max_value=30,
            value=20,
            step=5,
            help='🗺️ Controls the interactive path network map - shows shortest paths to this many closest destinations with travel times and edge durations'
        )
        
        connectivity_sample = st.slider(
            'Connectivity Matrix Sample',
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help='📊 Number of nodes to sample for the connectivity matrix heatmap (higher = more detail but slower rendering)'
        )
        
        max_viz_nodes = st.slider(
            'Max 3D Topology Nodes',
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help='🌐 Maximum nodes to display in the interactive 3D graph topology visualization (higher = more complete view but slower performance)'
        )
        
        with st.expander('ℹ️ What do these settings control?'):
            st.markdown("""
            **Path Map Destinations** 🗺️  
            The interactive network map shows shortest paths from your source vertex to the N closest destinations.
            - Shows: Vertex labels, journey times, edge durations
            - Features: Hover tooltips, zoom/pan, clickable legend
            - Use case: Understanding path structure and travel patterns
            
            **Connectivity Matrix Sample** 📊  
            A heatmap showing which nodes connect to which other nodes.
            - Samples a subset of nodes for visualization
            - Higher values = more detailed but slower
            - Use case: Analyzing graph connectivity patterns
            
            **Max 3D Topology Nodes** 🌐  
            An interactive 3D scatter plot of the graph structure.
            - Node size indicates importance (degree/connectivity)
            - Colors show clustering or properties
            - Use case: Understanding overall graph structure
            
            💡 **Tip:** Start with default values, then adjust based on your graph size and performance needs.
            """)
        
        st.divider()
        
        # Run button
        run_analysis = st.button('🚀 Run Analysis', type='primary', use_container_width=True)
        
        # Info
        with st.expander('ℹ️ About'):
            st.markdown("""
            ### ESD Graph Pathfinding Analyzer
            
            This application provides comprehensive analysis of temporal graph pathfinding 
            algorithms using the Edge Scan Dependency (ESD) graph approach.
            
            **Algorithms Implemented:**
            
            - **Serial CPU**: Baseline implementation using traditional BFS traversal
            - **Parallel MBFS (Algorithm 1)**: Multi-BFS approach with GPU acceleration
            - **Parallel Level Order (Algorithm 3)**: Level-based parallel traversal
            
            **Key Features:**
            
            ✅ GPU acceleration with CUDA/CuPy  
            ✅ Real-time performance comparison  
            ✅ Interactive 3D graph visualizations  
            ✅ Path reconstruction and analysis  
            ✅ Scalability projections  
            ✅ Comprehensive metrics and statistics  
            
            **How to Use:**
            
            1. Select your dataset and configure parameters
            2. Choose which algorithms to compare
            3. Click "Run Analysis" to execute
            4. Explore results through interactive charts and visualizations
            
            **Research Citation:**
            
            Based on "Efficient Algorithms for Fastest Path Problem in Temporal Graphs"
            """)
        
        # System info
        with st.expander('💻 System Information'):
            try:
                import cupy as cp
                gpu_available = cp.cuda.runtime.getDeviceCount() > 0
                if gpu_available:
                    device = cp.cuda.Device()
                    st.success(f'✅ GPU Available: {device.compute_capability}')
                    st.info(f'Memory: {device.mem_info[1] / 1e9:.2f} GB total')
                else:
                    st.warning('⚠️ No GPU detected')
            except:
                st.error('❌ CuPy not available - GPU algorithms will fail')
            
            st.info(f'NumPy version: {np.__version__}')
            st.info(f'Pandas version: {pd.__version__}')
    
    # Main content
    if run_analysis:
        if not any([run_serial, run_mbfs, run_lo]):
            st.warning('⚠️ Please select at least one algorithm to run')
            return
        
        # Configuration summary
        st.info(f"""
        **Analysis Configuration:** Dataset size: {num_rows if num_rows else 'Full'} rows | 
        Source: {source_vertex} | Path destinations: {top_n_paths} | 
        3D viz nodes: {max_viz_nodes} | Matrix sample: {connectivity_sample}
        """)
        
        # Step 1: Load Data
        st.header('📊 Data Loading')
        temporal_edges, df = load_data(dataset_path, num_rows)
        
        if temporal_edges is None:
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Temporal Edges', f'{len(temporal_edges):,}')
        with col2:
            st.metric('Dataset Rows', f'{len(df):,}')
        with col3:
            unique_vertices = len(set([e[0] for e in temporal_edges] + [e[1] for e in temporal_edges]))
            st.metric('Unique Vertices', f'{unique_vertices:,}')
        
        # Step 2: Build ESD Graph
        st.header('🔄 ESD Graph Construction')
        esd_graph = build_esd_graph(temporal_edges, num_rows)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('ESD Nodes', f'{len(esd_graph.nodes):,}')
        with col2:
            total_edges = sum(len(neighbors) for neighbors in esd_graph.adj.values())
            st.metric('ESD Edges', f'{total_edges:,}')
        
        # Validate source
        source_vertex = validate_source(esd_graph, source_vertex)
        
        # Step 3: Run Algorithms
        st.header('⚡ Algorithm Execution')
        results = run_algorithms(esd_graph, source_vertex, run_serial, run_mbfs, run_lo)
        
        if not results:
            st.error('No results to display')
            return
        
        # Step 4: Performance Analysis
        st.header('📈 Performance Analysis')
        
        # Performance metrics dashboard
        display_performance_metrics(results, len(esd_graph.nodes))
        
        st.divider()
        
        # Charts in tabs
        tab1, tab2, tab3, tab4 = st.tabs(['⏱️ Time Comparison', '🚀 Speedup', '📊 Distribution', '⚡ Efficiency'])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_perf = create_performance_chart(results)
                st.plotly_chart(fig_perf, use_container_width=True)
            with col2:
                fig_throughput = create_throughput_chart(results, len(esd_graph.nodes))
                st.plotly_chart(fig_throughput, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig_speedup = create_speedup_chart(results)
                if fig_speedup:
                    st.plotly_chart(fig_speedup, use_container_width=True)
                else:
                    st.info('Run Serial algorithm to see speedup comparison')
            with col2:
                fig_efficiency = create_parallel_efficiency_chart(results)
                if fig_efficiency:
                    st.plotly_chart(fig_efficiency, use_container_width=True)
                else:
                    st.info('Parallel efficiency requires Serial baseline')
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                fig_dist = create_journey_distribution(results)
                st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                fig_box = create_journey_time_box_plot(results)
                st.plotly_chart(fig_box, use_container_width=True)
            
            st.plotly_chart(create_cumulative_distribution(results), use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                fig_reach = create_reachability_comparison(results)
                st.plotly_chart(fig_reach, use_container_width=True)
            with col2:
                fig_radar = create_efficiency_radar(results)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info('Need multiple algorithms for comparison')
        
        # Step 5: Graph Analysis
        st.header('🔍 Graph Structure Analysis')
        
        graph_tab1, graph_tab2, graph_tab3, graph_tab4 = st.tabs(['🌐 Topology', '📈 Temporal', '📊 Degree', '🎯 Levels'])
        
        with graph_tab1:
            with st.spinner('Generating topology visualization...'):
                fig_topology = create_graph_topology_view(esd_graph, max_nodes=max_viz_nodes)
                st.plotly_chart(fig_topology, use_container_width=True)
            
            st.subheader('Connectivity Matrix')
            fig_matrix = create_connectivity_matrix(esd_graph, sample_size=connectivity_sample)
            st.plotly_chart(fig_matrix, use_container_width=True)
            st.caption(f'Showing connectivity matrix for {connectivity_sample} sampled nodes (configured in sidebar)')
        
        with graph_tab2:
            fig_temporal = create_temporal_heatmap(esd_graph)
            st.plotly_chart(fig_temporal, use_container_width=True)
        
        with graph_tab3:
            fig_degree = create_degree_distribution(esd_graph)
            st.plotly_chart(fig_degree, use_container_width=True)
        
        with graph_tab4:
            fig_levels = create_level_distribution(esd_graph)
            st.plotly_chart(fig_levels, use_container_width=True)
            st.info('💡 Level distribution shows the graph structure used by the Level Order algorithm for parallel processing')
        
        # Step 6: Interactive Path Network Visualization
        st.header('🗺️ Interactive Path Network Map')
        
        st.markdown(f"""
        This interactive map shows the shortest paths from the source vertex to the top **{top_n_paths}** closest destinations.
        Explore the network structure, travel times, and edge durations.
        
        💡 *Adjust the number of destinations in the sidebar before running analysis.*
        """)
        
        if 'Serial' in results and results['Serial'].get('paths'):
            col_legend, col_map = st.columns([1, 4])
            
            with col_legend:
                st.markdown("""
                **Map Legend:**
                
                ⭐ **Red Star**  
                Source vertex
                
                💎 **Teal Diamonds**  
                Top destinations
                
                ⚪ **Green Circles**  
                Intermediate nodes
                
                ― **Gray Lines**  
                Travel edges
                
                ---
                
                **Interactions:**
                - Hover for details
                - Zoom with scroll
                - Pan by dragging
                - Click legend to toggle
                """)
            
            with col_map:
                # Use a container to avoid full reload
                fig_path_map = create_path_network_map(results, source_vertex, esd_graph, top_n=top_n_paths)
                if fig_path_map:
                    st.plotly_chart(fig_path_map, use_container_width=True, key=f'path_map_{top_n_paths}')
                else:
                    st.warning('Unable to generate path map. Ensure paths are available.')
            
            # Journey time comparison chart
            st.subheader('📊 Journey Time Rankings')
            fig_journey_comp = create_journey_time_comparison_chart(results, source_vertex, top_n=top_n_paths)
            if fig_journey_comp:
                st.plotly_chart(fig_journey_comp, use_container_width=True, key=f'journey_chart_{top_n_paths}')
                
            st.info("""
            💡 **Tip:** Adjust the slider above to dynamically change the number of destinations displayed. 
            The visualization updates instantly without rerunning the entire analysis.
            """)
        else:
            st.info('🔍 Enable Serial algorithm and run analysis to see path network visualization')
        
        st.divider()
        
        # Step 7: Scalability Analysis
        st.header('📈 Scalability Projection')
        fig_scalability = create_scalability_projection(results, len(esd_graph.nodes))
        st.plotly_chart(fig_scalability, use_container_width=True)
        st.info('💡 This projection estimates algorithm performance on different dataset sizes based on complexity analysis')
        
        # Step 8: Validation
        validate_results(results)
        
        # Step 9: Detailed Path Information
        display_sample_paths(results, source_vertex, num_samples=top_n_paths)
        
        # Step 10: Detailed Metrics Table
        st.header('📊 Detailed Metrics')
        
        df_comparison = create_comparison_table(results)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Download option
        csv = df_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics as CSV",
            data=csv,
            file_name="algorithm_comparison.csv",
            mime="text/csv"
        )
        
    else:
        # Landing page
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 👋 Welcome to the ESD Graph Pathfinding Analyzer!
        
        This application provides a comprehensive analysis of temporal graph pathfinding algorithms using GPU acceleration.
        
        **To get started:**
        1. **Configure dataset** - Select your data file and size
        2. **Choose algorithms** - Serial CPU, Parallel MBFS, or Level Order
        3. **Set visualization parameters** - Path destinations, matrix samples, 3D nodes
        4. **Click "Run Analysis"** - All visualizations will use your configured settings
        
        **💡 Pro Tip:** Set all visualization parameters in the sidebar BEFORE running analysis. 
        This prevents page reloads and provides a smooth experience exploring your results.
        
        The app will guide you through:
        - Data loading and preprocessing
        - ESD graph construction
        - Algorithm execution and comparison
        - Interactive path network maps
        - Performance visualization and metrics
        - Result validation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.image('https://via.placeholder.com/800x400/667eea/ffffff?text=Configure+Settings+and+Click+Run+Analysis', 
                use_column_width=True)

def run_custom_path_query():
    """Custom path query interface with traffic and cost analysis"""
    st.header("🎯 Custom Path Query")
    st.markdown("Find specific paths with detailed traffic, cost, and conflict analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Query Configuration")
        
        # Dataset selection
        st.subheader("📂 Dataset")
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
        st.subheader("🔍 Query Mode")
        query_mode = st.radio(
            "Select Query Type:",
            ["Single Path", "Multiple Pairs"],
            help="Query one path or multiple source-destination pairs"
        )
        
        # Load data button
        load_data_btn = st.button("📊 Load Dataset", type="primary")
    
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
                    st.info(f"📊 Loaded {len(df)} temporal edges from CSV")
                    
                    # Convert DataFrame to list of tuples (u, v, departure_time, duration)
                    temporal_edges = [
                        (str(row['from_stop_I']), str(row['to_stop_I']),
                         int(row['dep_time_ut']), int(row['arr_time_ut'] - row['dep_time_ut']))
                        for _, row in df.iterrows() 
                        if row['arr_time_ut'] - row['dep_time_ut'] > 0
                    ]
                    
                    # Transform to ESD graph
                    st.info("🔄 Transforming to ESD graph (this may take a moment)...")
                    esd_graph = transform_temporal_to_esd(temporal_edges)
                    
                    # Cache for future use
                    save_esd_graph_to_json(esd_graph, dataset_size)
                    st.success("💾 ESD graph cached for future use")
                else:
                    st.success("⚡ Loaded ESD graph from cache")
                
                # Store in session state
                st.session_state.esd_graph_custom = esd_graph
                st.session_state.temporal_graph_custom = None  # We don't need the raw df
                
                # Initialize weighted solver with consistent weights
                for node in esd_graph.nodes.values():
                    node.weight = 1  # Set consistent initial weights
                
                st.session_state.weighted_solver = ParallelWeightedLO(esd_graph)
                
                num_vertices = len(esd_graph.nodes)
                num_edges = sum(len(neighbors) for neighbors in esd_graph.adj.values())
                st.success(f"✅ ESD Graph Ready: {num_vertices} vertices, {num_edges} edges")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
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
                    help="Starting vertex for the path"
                )
            
            with col2:
                destination = st.number_input(
                    "Destination Vertex",
                    min_value=0,
                    max_value=len(esd_graph.nodes) - 1,
                    value=min(100, len(esd_graph.nodes) - 1),
                    help="Target vertex for the path"
                )
            
            if st.button("🚀 Find Path", type="primary"):
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
                            st.subheader("📍 Path Details")
                            fig_detail = create_path_detail_visualization(
                                result['path'],
                                esd_graph.edge_wait_times,
                                esd_graph.out_neighbors,
                                result['cost']
                            )
                            st.plotly_chart(fig_detail, use_container_width=True)
                            
                            # Conflict visualization
                            st.subheader("🚦 Traffic Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_conflicts = create_conflict_visualization(results, esd_graph)
                                st.plotly_chart(fig_conflicts, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Global Conflict Statistics**")
                                st.markdown(f"**Top Updated Nodes:**")
                                if global_stats:
                                    for i, stat in enumerate(global_stats[:5], 1):
                                        st.markdown(f"{i}. {stat}")
                                else:
                                    st.info("No conflicts detected")
                            
                            # Path reconstruction
                            st.subheader("🛤️ Path Sequence")
                            path_str = " → ".join(str(v) for v in result['path'])
                            st.code(path_str, language=None)
                        else:
                            st.warning("❌ No path found")
                            
                    except Exception as e:
                        st.error(f"❌ Error computing path: {e}")
        
        else:
            # Multiple pairs query
            st.subheader("📋 Multiple Source-Destination Pairs")
            
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
                    st.error("❌ Invalid format. Use: source,destination")
            else:
                uploaded_file = st.file_uploader("Upload CSV (columns: source,destination)", type=['csv'])
                if uploaded_file:
                    pairs_df = pd.read_csv(uploaded_file)
                    pairs = list(zip(pairs_df['source'], pairs_df['destination']))
            
            if pairs and st.button("🚀 Find All Paths", type="primary"):
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
                        st.subheader("📊 Analysis")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Cost Heatmap", "Multi-Path Network", "Comparison Table", "Edge Usage"
                        ])
                        
                        with tab1:
                            fig_heatmap = create_cost_heatmap(results)
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        with tab2:
                            fig_network = create_multi_path_network(results, esd_graph)
                            st.plotly_chart(fig_network, use_container_width=True)
                        
                        with tab3:
                            display_path_comparison_table(results)
                        
                        with tab4:
                            fig_edges = create_edge_usage_chart(results)
                            st.plotly_chart(fig_edges, use_container_width=True)
                        
                        # Global statistics
                        st.subheader("🌐 Global Conflict Statistics")
                        
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
                        st.subheader("💾 Export Results")
                        results_df = pd.DataFrame([{
                            'source': r['source'],
                            'destination': r['dest'],
                            'status': r['status'],
                            'cost': r['cost'],
                            'conflicts': r['conflicts'],
                            'path_length': len(r['path']),
                            'path': '→'.join(map(str, r['path']))
                        } for r in results])
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results CSV",
                            csv,
                            "path_query_results.csv",
                            "text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error computing paths: {e}")
    else:
        st.info("👆 Load a dataset from the sidebar to begin querying paths")

if __name__ == "__main__":
    main()
