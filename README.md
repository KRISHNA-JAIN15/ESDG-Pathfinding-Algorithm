# ESDG Pathfinding Framework

This project implements a two-step framework for efficient schedule-based pathfinding using the **Event-Station Directed Graph (ESDG)** approach. It separates the messy temporal constraints of real-world trip schedules into a clean, static graph structure that enables fast pathfinding.

## 🎯 Streamlit Web Application

A comprehensive web interface is available for interactive analysis and visualization:

```bash
streamlit run app.py
```

### Features:
- **Two Analysis Modes**:
  1. **Benchmark Analysis**: Compare Serial CPU, Parallel MBFS, and Level Order algorithms
  2. **Custom Path Query**: Find specific paths with traffic/cost analysis
  
- **15+ Interactive Visualizations**: 3D graphs, heatmaps, timelines, network maps
- **GPU Acceleration**: CUDA/CuPy for high-performance pathfinding
- **Conflict Tracking**: Identify bottlenecks and high-traffic nodes
- **Export Results**: Download metrics and paths as CSV

See [PATH_QUERY_GUIDE.md](PATH_QUERY_GUIDE.md) for the custom query feature documentation.

---

## 📌 Part 1: Transformation — Building the Connection Map (ESDG)

Handled by: **`esd_transformer.py`**

The goal of this step is to transform the **temporal graph G** (where edges are trips) into a **static directed acyclic graph (ESDG, denoted G̃)** where nodes represent trips and edges represent feasible connections.

### 🔹 Step 1: Trips → Nodes

* In the temporal graph, each **edge e** is a trip (e.g., `A → B, dep 09:00, arr 09:30`).
* In the ESDG, every trip becomes a **node vₑ** (e.g., node `v₁` representing the trip `A → B (09:00–09:30)`).

### 🔹 Step 2: Create Directed Edges

Directed edges between trips are added if two conditions are satisfied:

#### ✅ Condition 1: Time-Respecting & Consecutive

* The **arrival location** of trip `e` matches the **departure location** of trip `f`.
* The **departure time** of `f` is **≥ arrival time** of `e`.

Example:

```
e = (A → B, arr 09:30)
f = (B → C, dep 09:45)
```

This connection is valid because 09:45 ≥ 09:30.

#### ✅ Condition 2: No Better Intermediate Option

* An edge `(vₑ → v𝒻)` is added **only if no other trip g from the same origin to the same destination is strictly better**.
* "Better" means: departs after `e` arrives **but** reaches the destination **earlier** than `f`.

Example:

```
e = (A → B, arr 09:30)
f = (B → C, dep 09:45, arr 10:15)
g = (B → C, dep 09:50, arr 10:05)
```

Here, `g` is strictly better than `f` → ❌ Do not add `(vₑ → v𝒻)`.

After applying these rules, the ESDG becomes a **directed acyclic graph (DAG)**. All time logic is embedded into the graph structure.

---

## 📌 Part 2: Search — Fastest Path on the ESDG

Handled by: **`serial_esdg_fpd.py`**

Once the ESDG is built, finding the **fastest journey** is much simpler.

### 🔹 Step 1: Identify & Sort Starting Trips

* Given a **source station s** (e.g., station `2421`), find all trips that **depart from s**.
* Sort these trips in **descending order of departure time** → later departures are processed first.

### 🔹 Step 2: Traverse from Each Source Node

* For each starting trip, perform a **graph traversal** (e.g., BFS).
* Compute **journey time** relative to the starting node:

  ```
  Journey Time = arrival_time(current_node) − departure_time(source_node)
  ```
* Maintain a dictionary `journey_times[destination] = best_duration`.
* Update only if a shorter duration is found.

### 🔹 Step 3: Pruning & Optimization

* A set `visited_esdg_nodes` avoids redundant searches.
* If a node was already reached from a **later departure**, paths through earlier departures are pruned — they cannot yield a better result.

---

## 🚀 Why This Works

* **Separation of Concerns:**

  * Transformation handles all complex **temporal constraints** once.
  * Search runs on a simple static graph → efficient.

* **Efficiency:**

  * Pruning ensures no wasted exploration.
  * Reverse-time processing guarantees shortest journeys are found quickly.

* **Correctness:**

  * The ESDG structure ensures only valid, optimal trip connections exist.

---

## ✨ Project Features

This framework goes beyond a basic implementation and includes a full analysis suite:

* **Fast Data Loading:** Optimized data preparation using `pandas` vectorization, avoiding slow `iterrows` operations.
* **ESDG Caching:** The largest generated ESDG is automatically cached to `cache/largest_esd_graph.json` to speed up subsequent runs and analysis.
* **Detailed Path Analysis:** An "itinerary" view for any source-destination pair, showing each leg of the journey, transfer times, and a full summary.
* **Rich Visualization:** Generates clear, dedicated plots for the top 3 fastest paths, including departure and arrival times.
* **Performance Profiling:** A `gprof`-style profiler (`analysis/detailed_profiler.py`) that identifies function-level hotspots.
* **Scalability Analysis:** A script (`analysis/scalability_analyzer.py`) that measures and plots the performance of each project stage against increasing dataset sizes.

---
## 🚀 Performance Analysis & Hotspots

After extensive optimization and profiling, the following conclusions were reached:

1.  **Initial Bottleneck (SOLVED):** The initial profiler run revealed that **`pandas.iterrows()`** was consuming over 45% of the runtime. This was fixed by switching to vectorized operations, resulting in a **577% speedup** in the data loading stage.

2.  **Secondary Bottleneck (Managed):** Excessive `logging` calls, particularly for I/O operations, were identified as a secondary bottleneck. These have been reduced, but can be disabled entirely for pure performance runs.

3.  **True Computational Hotspot:** With the artificial bottlenecks removed, profiling confirms that the **`find_fastest_paths` function** in `serial_esdg_fpd.py` is the main computational core, consuming **~66% of the execution time**.

run_single_experiment (runtime: 0.280s): This is the total time for the entire experiment. It's at the top because it's the main function we are profiling.

find_fastest_paths (runtime: 0.171s): This is the first function called by run_single_experiment in the list. This tells you that out of the total 0.280 seconds, a massive 0.171 seconds (about 61%) was spent inside the actual FPD algorithm.

load_esd_graph_from_json (runtime: 0.075s): This is the next major function called by run_single_experiment. It shows that loading and parsing the cached JSON file also takes a noticeable amount of time.

**Conclusion:** The analysis successfully isolated the true hotspot. The next logical step, parallelization, should be focused on the `find_fastest_paths` algorithm to gain the most significant performance improvements.

---

## 📂 Project Structure

```
Project/
├── Datasets/
│   └── network_temporal_day.csv
├── ESD_Graph/
│   ├── structures/
│   └── esd_transformer.py
├── FPD_Algorithm/
│   └── serial_esdg_fpd.py
├── analysis/
│   ├── scalability_analyzer.py
│   ├── detailed_profiler.py
│   ├── path_analyzer.py
│   └── visualizer.py
├── cache/
│   └── largest_esd_graph.json  (auto-generated)
├── utils/
│   └── graph_caching.py
├── main.py
├── find_route.py
├── .gitignore
└── README.md
```

---

## 📝 Example Workflow

1. **Build the ESDG**

   ```bash
   python esd_transformer.py input_schedule.csv esdg_output.json
   ```

2. **Find Fastest Path**

   ```bash
   python serial_esdg_fpd.py esdg_output.json --source 2421 --target 3150
   ```

3. **Run the Detailed Profiler**
    ```bash
    python analysis/detailed_profiler.py
    ```

4. **Find Specific Route**

    ```bash
    python find_route.py --source "2421" --destination "3688" --rows 20000
    ```

5. **Run Scalibility Analyzer**

    ```bash
    python analysis/scalability_analyzer.py
    ```

6. **Create Cache** 

    ```bash
    python create_cache.py --rows 20000
    ```
---

💡 With the ESDG approach, **messy real-world schedules** are transformed into **elegant graph problems**, making optimal route planning both fast and reliable.
