import cProfile
import pstats
import io
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function we want to profile from the scalability analyzer
from analysis.scalability_analyzer import run_single_experiment

def run_profiling_analysis():
    """
    Runs a detailed cProfile analysis on the main experiment function to
    identify hotspots and generates a human-readable report.
    """
    print("="*60)
    print("  RUNNING DETAILED PROFILING ANALYSIS")
    print("="*60)
    
    # --- Configuration ---
    # We profile the largest dataset size to find the most significant bottlenecks
    DATASET_FILE = "Datasets/network_temporal_day.csv"
    SOURCE_VERTEX = "2421"
    PROFILE_DATASET_SIZE = 20000 
    
    # --- Profiling ---
    print(f"Profiling the experiment with {PROFILE_DATASET_SIZE} rows...")
    
    # Create a profiler object
    profiler = cProfile.Profile()
    
    # Run the target function under the profiler's control
    profiler.enable()
    run_single_experiment(DATASET_FILE, SOURCE_VERTEX, PROFILE_DATASET_SIZE)
    profiler.disable()
    
    print("Profiling complete. Generating report...")

    # --- Report Generation ---
    # Use io.StringIO to capture the pstats output as a string
    string_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=string_stream)
    
    # Sort the stats for clarity. 'cumulative' is the most useful for finding hotspots.
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    
    # --- Create the Report Header ---
    report_header = (
        f"Profiling Analysis Report\n"
        f"{'='*60}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Dataset: {DATASET_FILE}\n"
        f"Rows Processed: {PROFILE_DATASET_SIZE}\n"
        f"{'='*60}\n\n"
    )

    # --- Add Stats to the Report ---
    string_stream.write(report_header)
    string_stream.write("--- Top 25 Functions by Cumulative Time (incl. sub-functions) ---\n")
    stats.print_stats(25) # Print the top 25 offenders
    
    string_stream.write("\n\n--- Top 25 Functions by Total Time (excl. sub-functions) ---\n")
    stats.sort_stats(pstats.SortKey.TIME) # Sort by internal time
    stats.print_stats(25)

    report_content = string_stream.getvalue()

    # --- Save the Report to a File ---
    report_filename = "analysis/profiling_report.txt"
    with open(report_filename, 'w') as f:
        f.write(report_content)
        
    print(f"\n✅ Success! Detailed analysis saved to: {report_filename}")
    
    # Also print to console for immediate viewing
    print("\n" + report_content)


if __name__ == "__main__":
    run_profiling_analysis()