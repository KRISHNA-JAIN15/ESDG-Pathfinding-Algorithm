from datetime import datetime

def analyze_path(
    source_vertex: str, 
    destination_vertex: str, 
    journey_times: dict, 
    fastest_paths: dict
):
    """
    Provides a detailed, step-by-step breakdown of the fastest path
    from a source to a specific destination.
    """
    print("\n" + "="*60)
    print(f"  DETAILED PATH ANALYSIS: {source_vertex} -> {destination_vertex}")
    print("="*60)

    # 1. Retrieve the specific path and total duration
    path = fastest_paths.get(destination_vertex)
    total_duration = journey_times.get(destination_vertex)

    if not path or total_duration == float('inf'):
        print("This path is not reachable or does not exist.")
        return

    # 2. Print the Itinerary
    print(f"Fastest Journey Duration: {total_duration} seconds\n")
    
    last_arrival_time = None
    
    for i, step_node in enumerate(path):
        departure_dt = datetime.fromtimestamp(step_node.t).strftime('%Y-%m-%d %H:%M:%S')
        arrival_dt = datetime.fromtimestamp(step_node.a).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"--- Leg {i+1}: Trip e{step_node.original_edge_id} ---")
        print(f"  From:       {step_node.u}")
        print(f"  To:         {step_node.v}")
        print(f"  Depart at:  {departure_dt} (Timestamp: {step_node.t})")
        print(f"  Arrive at:  {arrival_dt} (Timestamp: {step_node.a})")
        print(f"  Trip Time:  {step_node.a - step_node.t} seconds")

        # Calculate wait time if this is not the first leg of the journey
        if last_arrival_time is not None:
            wait_time = step_node.t - last_arrival_time
            print(f"  Wait Time:  {wait_time} seconds (transfer at node {step_node.u})")
        
        last_arrival_time = step_node.a
        print("-" * 20)

    # 3. Final Summary
    initial_departure_time = path[0].t
    final_arrival_time = path[-1].a
    calculated_duration = final_arrival_time - initial_departure_time
    
    print("\n--- Journey Summary ---")
    print(f"Initial Departure from '{source_vertex}': {datetime.fromtimestamp(initial_departure_time).strftime('%H:%M:%S')}")
    print(f"Final Arrival at '{destination_vertex}':     {datetime.fromtimestamp(final_arrival_time).strftime('%H:%M:%S')}")
    print(f"Total Journey Duration:            {calculated_duration} seconds (Matches: {calculated_duration == total_duration})")
    print("="*60)