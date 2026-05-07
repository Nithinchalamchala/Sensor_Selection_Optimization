# Sensor Selection Optimization for Smart Systems

**Optimization Techniques — Course Project**

A complete implementation of the Sensor Selection Problem for a smart building environment, solved using four metaheuristic algorithms: Genetic Algorithm (GA), Simulated Annealing (SA), Binary Particle Swarm Optimization (PSO), and a Hybrid GA+SA (SAGA). Includes an interactive Streamlit web application for visualization and experimentation.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Environment Model](#3-environment-model)
4. [Algorithms](#4-algorithms)
5. [Project Structure](#5-project-structure)
6. [Installation](#6-installation)
7. [Usage](#7-usage)
8. [Output Files](#8-output-files)
9. [Results](#9-results)
10. [Discussion](#10-discussion)

---

## 1. Problem Overview

Given a set of candidate sensor locations in a 2D smart building floor plan, select an optimal subset that simultaneously:

- Maximizes priority-weighted area coverage across zones
- Minimizes total deployment cost
- Minimizes redundant overlap between sensor coverage areas

Subject to hard constraints:

- Total cost must not exceed a given budget
- Every critical zone must be covered by at least 2 sensors (fault tolerance)
- Selected sensors must form a connected communication graph

This is a binary combinatorial optimization problem with a search space of 2^N solutions. For N = 150, exhaustive search is infeasible, making metaheuristics the appropriate approach.

---

## 2. Mathematical Formulation

### Decision Variables

```
x_i in {0, 1},  i = 1, ..., N
```

`x_i = 1` means sensor `i` is selected for deployment.

### Sensor Properties

Each sensor `i` has:
- Position `(p_ix, p_iy)` on the building grid
- Type `t_i` in {motion, thermal, acoustic}
- Coverage radius `r_i` (type-dependent, with 15% random variation)
- Deployment cost `c_i` (type-dependent, with 20% random variation)
- Line-of-sight coverage — walls block sensor range

| Sensor Type | Base Radius | Base Cost |
|-------------|-------------|-----------|
| Motion      | 6 units     | 10        |
| Thermal     | 8 units     | 18        |
| Acoustic    | 10 units    | 25        |

### Objective Functions

**f1 — Priority-Weighted Coverage (maximize):**

```
f1(x) = sum over covered cells of w(cell) / sum over all cells of w(cell)
```

where `w(cell)` is the priority weight of the zone containing that cell.

**f2 — Total Cost (minimize):**

```
f2(x) = sum(c_i * x_i) / sum(c_i)
```

**f3 — Coverage Overlap (minimize):**

```
f3(x) = overlapping cell-visits / total cell-visits
```

### Combined Scalar Fitness (minimize)

```
F(x) = -alpha * f1(x) + beta * f2(x) + gamma * f3(x) + lambda * P(x)
```

Default weights: `alpha = 0.60`, `beta = 0.25`, `gamma = 0.15`

### Constraints

**Budget:**
```
sum(c_i * x_i) <= B
```

**Critical Zone Redundancy:**
```
For every critical zone z: |{i : x_i = 1 and Area(i) intersects z}| >= 2
```

**Network Connectivity:**
```
G(x) is connected, where edge (i,j) exists iff distance(i,j) <= R_comm
```

Communication range `R_comm = 15` grid units.

### Penalty Function

Infeasible solutions are penalized:

```
P(x) = (budget_violation / B) + 0.1 * redundancy_violation + 0.05 * connectivity_violation
```

---

## 3. Environment Model

### Building Layout

A 50x50 grid representing a smart building floor plan with 8 zones, 5 obstacle walls, and 100-200 randomly placed candidate sensors.

```
Zone          Priority    Critical
-----------   --------    --------
Lobby         1.0         No
Corridor_1    1.5         No
Office_A      2.0         Yes
Office_B      2.0         Yes
Lab           3.0         Yes
Server_Room   3.5         Yes
Corridor_2    1.0         No
Meeting_Room  2.5         Yes
```

### Line-of-Sight

Sensor coverage is computed using Bresenham's line algorithm to check whether walls block the path between a sensor and each candidate cell. Coverage sets are precomputed once at startup for efficiency.

---

## 4. Algorithms

### 4.1 Genetic Algorithm (GA)

File: `ga.py`

| Parameter | Value |
|-----------|-------|
| Population size | 100 |
| Generations | 300 |
| Crossover rate | 0.85 |
| Mutation rate | 0.05 to 0.01 (adaptive) |
| Selection | Tournament (k=5) |
| Elitism | Top 5% |
| Crossover type | Uniform + Two-point (randomly chosen) |

Key design choices:
- Mutation rate decreases linearly over generations to shift from exploration to exploitation
- Repair operator greedily removes the lowest-value sensors when budget is violated
- Elitism ensures the best solutions are never lost

### 4.2 Simulated Annealing (SA)

File: `sa.py`

| Parameter | Value |
|-----------|-------|
| Initial temperature | 5.0 |
| Minimum temperature | 1e-4 |
| Cooling rate | 0.995 (geometric) |
| Max iterations | 50,000 |
| Reheat factor | 2.0x |
| Reheat patience | 500 iterations |

Key design choices:
- Two neighbourhood moves: bit-flip (70% probability) and block-swap (30%)
- Reheating: when no improvement is found for 500 iterations, temperature is multiplied by 2x to escape local optima
- Repair operator applied after every accepted move

### 4.3 Binary PSO with Levy Flight

File: `pso.py`

| Parameter | Value |
|-----------|-------|
| Particles | 80 |
| Iterations | 300 |
| Inertia weight | 0.9 to 0.4 (linear decay) |
| c1 (cognitive) | 2.0 |
| c2 (social) | 2.0 |
| Levy perturbation probability | 0.15 |
| Stagnation limit | 30 iterations |

Key design choices:
- Continuous velocity is converted to selection probability via sigmoid transfer function
- Levy-flight perturbation is applied to the global best when stagnation is detected, providing long-range jumps to escape local optima
- Inertia weight decays linearly to balance exploration and exploitation

### 4.4 Hybrid GA+SA (SAGA)

File: `hybrid_saga.py`

Strategy:
1. Phase 1 (GA): Run GA for 150 generations to broadly explore the search space
2. Phase 2 (SA): Use the best GA solution as SA's starting point for fine-grained local refinement

This combines GA's global exploration strength with SA's local exploitation precision.

---

## 5. Project Structure

```
sensor-selection/
|
|-- app.py              Streamlit web application
|-- main.py             Command-line orchestrator
|-- environment.py      Building grid, zones, obstacles, sensor model, LOS
|-- fitness.py          Multi-objective fitness, constraints, repair operator
|-- ga.py               Genetic Algorithm
|-- sa.py               Simulated Annealing with reheating
|-- pso.py              Binary PSO with Levy-flight perturbation
|-- hybrid_saga.py      Hybrid GA+SA
|-- pareto.py           Pareto front generation
|-- experiments.py      Sensitivity analysis experiments
|-- visualization.py    All plotting utilities
|-- requirements.txt    Python dependencies
|-- README.md           This file
|
|-- results/            Generated CSV and JSON data files
|   |-- ga_history.csv
|   |-- sa_history.csv
|   |-- pso_history.csv
|   |-- saga_ga_history.csv
|   |-- saga_sa_history.csv
|   |-- pareto_data.json
|   |-- sensitivity_budget.csv
|   |-- sensitivity_weights.csv
|   `-- sensitivity_n_sensors.csv
|
`-- report_figures/     Generated plots for report
    |-- 01_building_map.png
    |-- 02_coverage_ga.png
    |-- 02_coverage_sa.png
    |-- 02_coverage_pso.png
    |-- 02_coverage_saga.png
    |-- 03_connectivity_ga.png
    |-- 03_connectivity_sa.png
    |-- 03_connectivity_pso.png
    |-- 03_connectivity_saga.png
    |-- 04_convergence.png
    |-- 05_comparison_summary.png
    |-- 06_pareto_front.png
    `-- 07_sensitivity.png
```

---

## 6. Installation

### Requirements

- Python 3.9 or higher
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
numpy>=1.24.0
matplotlib>=3.7.0
networkx>=3.1
scipy>=1.10.0
streamlit>=1.28.0
pandas>=2.0.0
```

---

## 7. Usage

### Option A: Streamlit Web Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The app provides:

- Sidebar controls for all parameters (sensors, budget, objective weights)
- One-click algorithm execution with progress tracking
- Interactive coverage heatmaps and convergence plots
- Pareto front explorer with adjustable weight configurations
- CSV and JSON export for all results

**Tabs:**

| Tab | Content |
|-----|---------|
| Environment | Floor plan, zone table, sensor distribution |
| Run Algorithms | Algorithm selection, execution, progress |
| Results | Summary table, convergence curves, coverage heatmap |
| Pareto Front | Coverage vs cost tradeoff visualization |
| Export | Download CSV/JSON results |

### Option B: Command Line

**Full run (all algorithms + all experiments):**
```bash
python main.py
```

**Quick test run (reduced iterations, approximately 3 minutes):**
```bash
python main.py --quick
```

**Run a single algorithm:**
```bash
python main.py --algo GA
python main.py --algo SA
python main.py --algo PSO
python main.py --algo SAGA
```

**Custom configuration:**
```bash
python main.py --n-sensors 200 --seed 0
python main.py --no-pareto --no-sensitivity
```

**All CLI options:**

```
--quick              Reduced iterations for fast testing
--algo               Which algorithm to run: GA, SA, PSO, SAGA, ALL (default: ALL)
--no-pareto          Skip Pareto front generation
--no-sensitivity     Skip sensitivity analysis
--n-sensors INT      Number of candidate sensors (default: 150)
--seed INT           Random seed for reproducibility (default: 42)
```

---

## 8. Output Files

### Figures (`report_figures/`)

| File | Description |
|------|-------------|
| `01_building_map.png` | Floor plan with zones, obstacles, and all candidate sensors |
| `02_coverage_<algo>.png` | Coverage heatmap and zone breakdown for each algorithm |
| `03_connectivity_<algo>.png` | Sensor network communication graph for each algorithm |
| `04_convergence.png` | Best fitness and coverage over iterations for all algorithms |
| `05_comparison_summary.png` | Bar chart comparison of coverage, cost, sensor count, and time |
| `06_pareto_front.png` | Pareto-optimal solutions: coverage vs cost tradeoff |
| `07_sensitivity.png` | Sensitivity to budget, objective weights, and sensor count |

### Data (`results/`)

| File | Description |
|------|-------------|
| `ga_history.csv` | Per-generation statistics for GA |
| `sa_history.csv` | Per-iteration statistics for SA |
| `pso_history.csv` | Per-iteration statistics for PSO |
| `saga_ga_history.csv` | GA phase history for SAGA |
| `saga_sa_history.csv` | SA phase history for SAGA |
| `pareto_data.json` | All Pareto front points and configurations |
| `sensitivity_budget.csv` | Results across budget fractions |
| `sensitivity_weights.csv` | Results across alpha weight values |
| `sensitivity_n_sensors.csv` | Results across sensor counts |

---

## 9. Results

Full run results with N=150 sensors, 50x50 grid, budget set to 50% of total cost.

| Algorithm | Best Fitness | Coverage (f1) | Cost (f2) | Sensors Selected | Time (s) | Feasible |
|-----------|-------------|---------------|-----------|-----------------|----------|----------|
| GA        | -0.5261     | 0.9784        | 0.1290    | 21              | 43.8     | Yes |
| SA        | -0.5183     | 0.9805        | 0.1398    | 24              | 58.4     | Yes |
| PSO       | -0.4393     | 0.9908        | 0.2719    | 44              | 69.5     | Yes |
| SAGA      | -0.5169     | 0.9845        | 0.1511    | 23              | 59.9     | Yes |

**GA achieves the best overall fitness**, selecting only 21 sensors with 97.84% coverage at the lowest normalized cost (0.1290). All four algorithms produce fully feasible solutions satisfying budget, redundancy, and connectivity constraints.

### Sensitivity Analysis Findings

**Budget:** Coverage plateaus above 50% budget. Even at 30% budget, 97.4% coverage is achieved, indicating the algorithm efficiently selects high-value sensors first.

**Objective weights:** As alpha (coverage weight) increases from 0.2 to 0.8, coverage rises from 81.6% to 99.1% while cost increases from 8.7% to 16.7%, confirming the expected tradeoff.

**Sensor count:** Coverage improves significantly from N=50 (83.6%) to N=100 (97.0%), with marginal gains beyond N=125. More candidates reduce cost per unit coverage.

---

## 10. Discussion

### Why these algorithms suit this problem

The Sensor Selection Problem is a binary combinatorial optimization problem. For N=150, the search space contains approximately 10^45 solutions, making exhaustive search impossible.

- **GA** is well-suited because binary chromosomes naturally represent sensor selection, and crossover can combine good partial solutions from different parents. The population maintains diversity to avoid premature convergence.

- **SA** works well because the neighbourhood (bit-flip) is simple and the Metropolis acceptance criterion allows escaping local optima. Reheating provides a second chance when the algorithm stagnates.

- **PSO** adapts to binary spaces via sigmoid velocity transfer. Levy flights provide long-range exploration jumps that standard PSO cannot achieve, helping escape stagnation.

- **SAGA** leverages complementary strengths: GA's diversity for global search, SA's precision for local refinement. The two-phase approach consistently produces competitive results.

### Local optima analysis

All algorithms can get stuck in local optima. The mechanisms used to escape them:

- GA: population diversity and crossover recombination
- SA: reheating raises temperature when no improvement is found for 500 iterations
- PSO: Levy-flight perturbation on the global best when stagnation is detected for 30 iterations
- SAGA: inherits both mechanisms

### Constraint satisfaction

All algorithms produce feasible solutions thanks to the repair operator applied after every solution generation or modification. The repair greedily removes the lowest-value sensors (ranked by cost-to-coverage ratio) until the budget constraint is satisfied. The penalty function guides infeasible solutions toward feasibility during optimization.

### PSO behavior

PSO selects significantly more sensors (44 vs 21 for GA) with higher overlap (58% vs 19%). This indicates PSO prioritizes raw coverage over cost efficiency. The sigmoid transfer function tends to produce denser solutions because particles are attracted toward the global best, which often has many sensors active.

---

## Deployment

### Streamlit Community Cloud (free)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and set the main file to `app.py`
5. Click Deploy

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t sensor-optimizer .
docker run -p 8501:8501 sensor-optimizer
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Numerical operations, array handling |
| matplotlib | >=3.7.0 | All plotting and visualization |
| networkx | >=3.1 | Sensor connectivity graph analysis |
| scipy | >=1.10.0 | Levy flight distribution |
| streamlit | >=1.28.0 | Interactive web application |
| pandas | >=2.0.0 | Data export and tabular display |

---

## Known Limitations

- 2D environment only, no multi-floor support
- Circular sensor coverage model, no directional sensors
- Static environment, no dynamic obstacles or sensor failures
- Weighted scalarization for multi-objective optimization rather than true Pareto optimization (NSGA-II would be more rigorous)
- All sensors share the same communication range

---

## License

This project is developed for educational purposes as part of an Optimization Techniques course project.
