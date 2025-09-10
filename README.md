# Assessment Intelic – Drone Path Planning

## Task Description  

Avalor AI is developing an advanced drone swarm system for autonomous area monitoring. To achieve effective coverage, we model the world as an **N×N grid**, where each grid cell (or *plane*) has an associated numerical value.  

- A drone collects the value of a cell when it visits it.  
- After being visited, a cell’s value drops to `0`, but it gradually **recharges** back toward its original value over time.  
- Moving to an adjacent cell (horizontal, vertical, or diagonal) costs **1 time step**.  
- The drone’s goal is to maximize the **total collected score** within a given number of time steps `t` and while respecting a maximum time budget `T` (ms) for planning each move.  

### Assignment Requirements  
- Input:  
  - `N`: grid size (N×N)  
  - `t`: number of time steps available  
  - `T`: maximum duration (ms) the planning algorithm may take at each step  
  - `(x, y)`: starting position of the drone  
- Output:  
  - The path chosen by the drone  
  - The total score collected  
- Algorithm should work efficiently for large grids and long time horizons.  
- Bonus challenge: extend to multiple drones (swarm planning).  

---

## Implemented Solutions  

Two different approaches to the assignment are included in this repository. Both use the same core components (`Grid`, `Drone`, `Simulation`) but differ in how the **Planner** selects and evaluates paths.  

### **Code 1 – Hotspot Selection + Bounded A\***  
custom_astar_single_hotspot_planner.py
- **Hotspot-based planning:** Finds a promising hotspot (cell with high value + high-value neighbors and applying extra weight for nearby and non-visited hotspots) near the drone.  
- **Bounded A\***: Runs a time-limited A\*-like search toward that hotspot.  
  - Uses Chebyshev distance for movement costs.  
  - Keeps track of score-per-step to balance distance vs. reward.  
  - Time-limited with `T` to remain responsive.  
- **Fallback:** If no path is found in time, defaults to a greedy move toward the neighbor with the highest current value.  

**Strengths:**  
- More thorough path exploration thanks to A\*.  
- Incorporates both distance penalty (`alpha`) and exploration bonus (`beta`).  

**Weaknesses:**  
- Higher computational overhead due to bounded A\* search.  
- May scale worse for very large grids or extremely tight time budgets.  

---

### **Code 2 – Hotspot Ranking + Greedy Lookahead**
greedy_lookahead_multi_hotspot_planner.py
- **All hotspot ranking:** Scans the drone’s local neighborhood to rank *all* candidate hotspots using value, neighbor average, distance, and exploration bonus.  
- **Greedy lookahead pathing:** For each hotspot, builds a greedy path that always reduces Chebyshev distance, with a small rollout (`lookahead_depth`) for tie-breaking.  
- **Evaluation:** Selects the hotspot path with the highest average reward per step.  
- **Fallback:** If no hotspot is viable, moves to the best-value neighbor.  

**Strengths:**  
- Faster and simpler than bounded A\*.  
- Scales better for larger grids and stricter time limits.  
- Flexible through `lookahead_depth` parameter.  

**Weaknesses:**  
- Less optimal than A\* in complex reward landscapes.  
- Paths may be short-sighted if the greedy rollout is too shallow.  

---

## Summary of Differences  

| Feature                  | Code 1 (Bounded A\*)             | Code 2 (Greedy Lookahead)        |
|--------------------------|----------------------------------|----------------------------------|
| **Hotspot search**       | Single best hotspot              | All candidate hotspots ranked    |
| **Pathfinding**          | Time-limited bounded A\*         | Greedy with short lookahead      |
| **Exploration**          | Distance penalty + exploration bonus | Same, but applied to all hotspots |
| **Scalability**          | More computationally expensive   | Faster, more scalable            |
| **Fallback strategy**    | Best neighbor if path fails      | Best neighbor if no hotspot path |

---

## Running the Simulation  

1. Run the following line in the terminal:
   ```
   pip install -r requirements.txt
   ```

2. Tune the parameters in the main function accordingly and run codes custom_astar_single_hotspot_planner.py and greedy_lookahead_multi_hotspot_planner.py.

3. Ignore greedy_multi_hotspot_planner.py and greedy_single_hotspot_planner.py