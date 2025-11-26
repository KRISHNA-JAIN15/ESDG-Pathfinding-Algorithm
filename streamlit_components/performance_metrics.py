import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from collections import defaultdict

def display_multi_query_metrics(results, num_queries):
    """
    Display performance metrics specifically for multi-query benchmark mode
    """
    st.subheader('Multi-Query Performance Dashboard')
    
    # Create metrics grid
    num_algorithms = len(results)
    cols = st.columns(num_algorithms)
    
    for col, (alg, data) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {alg}")
            
            # Total compute time
            st.metric('Total Time', f"{data['time']:.4f}s")
            
            # Average time per query
            avg_per_query = data['avg_time_per_query'] * 1000  # Convert to ms
            st.metric('Avg per Query', f"{avg_per_query:.2f}ms")
            
            # Throughput (queries per second)
            throughput = num_queries / data['time']
            st.metric('Throughput', f"{throughput:.1f} q/s")
            
            # Speedup (if applicable)
            if 'Serial' in results and alg != 'Serial':
                speedup = results['Serial']['time'] / data['time']
                st.metric('Speedup', f"{speedup:.2f}x", 
                         delta=f"{(speedup-1)*100:.1f}%",
                         delta_color="normal")

def create_algorithm_timeline(execution_log):
    """
    Create a Gantt-chart style timeline of algorithm execution phases
    """
    fig = go.Figure()
    
    for i, entry in enumerate(execution_log):
        fig.add_trace(go.Bar(
            y=[entry['algorithm']],
            x=[entry['duration']],
            name=entry['phase'],
            orientation='h',
            marker=dict(color=entry['color']),
            text=f"{entry['duration']:.4f}s",
            textposition='auto',
            hovertemplate=f"<b>{entry['algorithm']}</b><br>" +
                         f"Phase: {entry['phase']}<br>" +
                         f"Duration: {entry['duration']:.4f}s<extra></extra>"
        ))
    
    fig.update_layout(
        title='Algorithm Execution Timeline',
        xaxis_title='Time (seconds)',
        yaxis_title='Algorithm',
        barmode='stack',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig

def create_memory_usage_chart(memory_stats):
    """
    Visualize memory usage across different components
    """
    categories = list(memory_stats.keys())
    values = list(memory_stats.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=values,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='MB')
        ),
        text=[f'{v:.2f} MB' for v in values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Memory: %{y:.2f} MB<extra></extra>'
    ))
    
    fig.update_layout(
        title='Memory Usage by Component',
        xaxis_title='Component',
        yaxis_title='Memory (MB)',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_throughput_chart(results, num_nodes=None, is_multi_query=False):
    """
    Calculate and visualize throughput (nodes/queries processed per second)
    """
    algorithms = []
    throughputs = []
    colors = []
    
    for alg, data in results.items():
        algorithms.append(alg)
        
        if is_multi_query and 'queries' in data:
            # Throughput in queries per second
            throughput = data['queries'] / data['time']
        elif num_nodes:
            # Throughput in nodes processed per second
            throughput = num_nodes / data['time']
        else:
            # Fallback: use inverse time
            throughput = 1.0 / data['time']
        
        throughputs.append(throughput)
        colors.append(data['color'])
    
    fig = go.Figure()
    
    if is_multi_query:
        ylabel = 'Queries per Second'
        hover_label = 'queries/s'
    else:
        ylabel = 'Nodes Processed per Second'
        hover_label = 'nodes/s'
    
    fig.add_trace(go.Bar(
        x=algorithms,
        y=throughputs,
        marker_color=colors,
        text=[f'{t:,.0f}' if t >= 1 else f'{t:.2f}' for t in throughputs],
        textposition='auto',
        hovertemplate=f'<b>%{{x}}</b><br>Throughput: %{{y:,.2f}} {hover_label}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Processing Throughput',
        xaxis_title='Algorithm',
        yaxis_title=ylabel,
        template='plotly_white',
        height=400
    )
    
    return fig

def create_scalability_projection(results, current_size):
    """
    Project performance on different dataset sizes
    """
    sizes = [1000, 10000, 30000, 100000, 500000, 1000000]
    
    fig = go.Figure()
    
    for alg, data in results.items():
        # Simple linear projection (real-world would be more complex)
        base_time = data['time']
        
        if 'Serial' in alg:
            # O(n) scaling
            projected_times = [base_time * (s / current_size) for s in sizes]
        else:
            # Better than linear for parallel
            projected_times = [base_time * ((s / current_size) ** 0.8) for s in sizes]
        
        fig.add_trace(go.Scatter(
            x=sizes,
            y=projected_times,
            mode='lines+markers',
            name=alg,
            line=dict(color=data['color'], width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Size: %{x:,} nodes<br>' +
                         'Est. Time: %{y:.4f}s<extra></extra>'
        ))
    
    # Mark current size
    fig.add_vline(x=current_size, line_dash="dash", line_color="gray",
                  annotation_text=f"Current: {current_size:,}")
    
    fig.update_layout(
        title='Scalability Projection',
        xaxis_title='Dataset Size (nodes)',
        yaxis_title='Estimated Time (seconds)',
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_comparison_table(results, is_multi_query=False):
    """
    Create a detailed comparison table
    """
    data = []
    
    for alg, res in results.items():
        row = {
            'Algorithm': alg,
            'Compute Time': f"{res['time']:.6f}s",
        }
        
        if is_multi_query:
            # Multi-query mode: show query-specific metrics
            row['Queries'] = res.get('queries', 'N/A')
            row['Avg per Query'] = f"{res.get('avg_time_per_query', 0) * 1000:.2f}ms" if 'avg_time_per_query' in res else 'N/A'
            row['Throughput'] = f"{res.get('queries', 0) / res['time']:.1f} q/s" if res['time'] > 0 else 'N/A'
            
            # Get query result times
            times = [t for t in res['data'].values() if t != float('inf')]
            if times:
                row['Min Result'] = f"{min(times):.2f}s"
                row['Max Result'] = f"{max(times):.2f}s"
                row['Median Result'] = f"{np.median(times):.2f}s"
            else:
                row['Min Result'] = 'N/A'
                row['Max Result'] = 'N/A'
                row['Median Result'] = 'N/A'
        else:
            # Single-source mode: show reachability metrics
            clean_data = {k: v for k, v in res['data'].items() if v != float('inf')}
            times = list(clean_data.values()) if clean_data else []
            
            row['Reachable Nodes'] = len(clean_data)
            row['Min Journey'] = f"{min(times):.2f}s" if times else 'N/A'
            row['Max Journey'] = f"{max(times):.2f}s" if times else 'N/A'
            row['Median Journey'] = f"{np.median(times):.2f}s" if times else 'N/A'
            row['Std Dev'] = f"{np.std(times):.2f}s" if times else 'N/A'
        
        if 'init_time' in res:
            row['Init Time'] = f"{res['init_time']:.6f}s"
            row['Total Time'] = f"{res['total_time']:.6f}s"
        else:
            row['Init Time'] = 'N/A'
            row['Total Time'] = row['Compute Time']
        
        data.append(row)
    
    return pd.DataFrame(data)

def display_performance_metrics(results, num_nodes):
    """
    Display comprehensive performance metrics in a grid layout
    """
    st.subheader('Performance Metrics Dashboard')
    
    # Create metrics grid
    num_algorithms = len(results)
    cols = st.columns(num_algorithms)
    
    for col, (alg, data) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {alg}")
            
            # Compute time
            st.metric('Compute Time', f"{data['time']:.4f}s")
            
            # Throughput
            throughput = num_nodes / data['time']
            st.metric('Throughput', f"{throughput:,.0f} nodes/s")
            
            # Reachable nodes
            clean_data = {k: v for k, v in data['data'].items() if v != float('inf')}
            st.metric('Reachable', f"{len(clean_data):,}")
            
            # Speedup (if applicable)
            if 'Serial' in results and alg != 'Serial':
                speedup = results['Serial']['time'] / data['time']
                st.metric('Speedup', f"{speedup:.2f}x", 
                         delta=f"{(speedup-1)*100:.1f}%",
                         delta_color="normal")

def create_journey_time_box_plot(results):
    """
    Create box plot comparing journey time distributions
    """
    fig = go.Figure()
    
    for alg, data in results.items():
        times = [t for t in data['data'].values() if t != float('inf')]
        
        if times:
            fig.add_trace(go.Box(
                y=times,
                name=alg,
                marker_color=data['color'],
                boxmean='sd',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Time: %{y}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Journey Time Distribution (Box Plot)',
        yaxis_title='Journey Time (seconds)',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_cumulative_distribution(results):
    """
    Create cumulative distribution function of journey times
    """
    fig = go.Figure()
    
    for alg, data in results.items():
        times = sorted([t for t in data['data'].values() if t != float('inf')])
        
        if times:
            cumulative = np.arange(1, len(times) + 1) / len(times) * 100
            
            fig.add_trace(go.Scatter(
                x=times,
                y=cumulative,
                mode='lines',
                name=alg,
                line=dict(color=data['color'], width=3),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Journey Time: %{x:.2f}s<br>' +
                             'Percentile: %{y:.1f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title='Cumulative Journey Time Distribution',
        xaxis_title='Journey Time (seconds)',
        yaxis_title='Cumulative Percentage (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_algorithm_summary_cards(results):
    """
    Create summary cards for each algorithm with key insights
    """
    for alg, data in results.items():
        with st.container():
            st.markdown(f"### {alg}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Get statistics
            clean_data = {k: v for k, v in data['data'].items() if v != float('inf')}
            times = list(clean_data.values()) if clean_data else []
            
            with col1:
                st.metric('⏱️ Compute', f"{data['time']:.4f}s")
            
            with col2:
                if 'init_time' in data:
                    st.metric('🔧 Init', f"{data['init_time']:.4f}s")
                else:
                    st.metric('🔧 Init', 'N/A')
            
            with col3:
                st.metric('🎯 Reachable', f"{len(clean_data):,}")
            
            with col4:
                if 'Serial' in results and alg != 'Serial':
                    speedup = results['Serial']['time'] / data['time']
                    st.metric('🚀 Speedup', f"{speedup:.2f}x")
                else:
                    st.metric('🚀 Speedup', 'Baseline')
            
            # Additional insights
            if times:
                st.markdown(f"""
                **Journey Time Statistics:**
                - Min: {min(times):.2f}s | Max: {max(times):.2f}s | Median: {np.median(times):.2f}s
                """)
            
            st.divider()

def create_parallel_efficiency_chart(results):
    """
    Calculate and visualize parallel efficiency
    (assumes ideal speedup = number of cores)
    """
    if 'Serial' not in results:
        return None
    
    algorithms = []
    efficiencies = []
    colors = []
    
    # Assume 8 cores (adjust based on actual hardware)
    ideal_cores = 8
    
    serial_time = results['Serial']['time']
    
    for alg, data in results.items():
        if alg != 'Serial':
            algorithms.append(alg)
            speedup = serial_time / data['time']
            efficiency = (speedup / ideal_cores) * 100  # Percentage
            efficiencies.append(efficiency)
            colors.append(data['color'])
    
    if not algorithms:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=algorithms,
        y=efficiencies,
        marker_color=colors,
        text=[f'{e:.1f}%' for e in efficiencies],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Efficiency: %{y:.1f}%<extra></extra>'
    ))
    
    # Add ideal efficiency line
    fig.add_hline(y=100, line_dash="dash", line_color="green",
                  annotation_text="Ideal (100%)")
    
    fig.update_layout(
        title=f'Parallel Efficiency (assuming {ideal_cores} cores)',
        xaxis_title='Algorithm',
        yaxis_title='Efficiency (%)',
        template='plotly_white',
        height=400
    )
    
    return fig
