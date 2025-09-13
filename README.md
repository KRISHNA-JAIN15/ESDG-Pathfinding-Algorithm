# ESDG Pathfinding Framework

This project implements a two-step framework for efficient schedule-based pathfinding using the **Event-Station Directed Graph (ESDG)** approach. It separates the messy temporal constraints of real-world trip schedules into a clean, static graph structure that enables fast pathfinding.

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

## 📂 Project Structure

```
├── esd_transformer.py   # Builds the ESDG from raw schedule data
├── serial_esdg_fpd.py   # Fastest path discovery on the ESDG
├── README.md            # Documentation
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

---

## 📖 References

* Original ESDG Algorithm Research Paper (refer to your course/project documentation)
* Graph Theory & Pathfinding Algorithms (BFS, DAG search)

---

💡 With the ESDG approach, **messy real-world schedules** are transformed into **elegant graph problems**, making optimal route planning both fast and reliable.
