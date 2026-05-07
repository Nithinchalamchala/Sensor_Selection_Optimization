"""
app.py
======
Streamlit web application for the Sensor Selection Problem.

Features
--------
  - Interactive parameter tuning (budget, weights, N sensors)
  - Real-time algorithm execution
  - Live visualization of solutions
  - Algorithm comparison dashboard
  - Pareto front explorer
  - Export results
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import time
import io
import json

from environment import BuildingEnvironment, SENSOR_TYPES, COMM_RANGE
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm
from sa import SimulatedAnnealing
from pso import BinaryPSO
from hybrid_saga import HybridSAGA
from pareto import generate_pareto_front

# Page config
st.set_page_config(
    page_title="Sensor Selection Optimizer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'pareto_data' not in st.session_state:
    st.session_state.pareto_data = None

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def plot_building_interactive(env):
    """Plot building map with zones and sensors."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ZONE_COLORS = [
        "#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7",
        "#D2B4DE", "#FDFEFE", "#ABB2B9", "#F1948A",
    ]
    
    # Zones
    for i, z in enumerate(env.zones):
        rect = mpatches.Rectangle(
            (z.x1, z.y1), z.x2 - z.x1, z.y2 - z.y1,
            linewidth=1.5, edgecolor="black",
            facecolor=ZONE_COLORS[i % len(ZONE_COLORS)], alpha=0.4,
        )
        ax.add_patch(rect)
        cx = (z.x1 + z.x2) / 2
        cy = (z.y1 + z.y2) / 2
        label = f"{z.name}\n(p={z.priority}{'★' if z.critical else ''})"
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8, fontweight="bold")
    
    # Obstacles
    for (x1, y1, x2, y2) in env.obstacles:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0, facecolor="#2c3e50", alpha=0.85,
        )
        ax.add_patch(rect)
    
    # Sensors
    for s in env.sensors:
        color = SENSOR_TYPES[s.stype]["color"]
        ax.scatter(s.x, s.y, c=color, s=30, zorder=5, alpha=0.7)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=SENSOR_TYPES[t]["color"],
               markersize=10, label=t.capitalize())
        for t in SENSOR_TYPES
    ]
    legend_elements.append(
        mpatches.Patch(facecolor="#2c3e50", label="Obstacle/Wall")
    )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    ax.set_xlim(0, env.W)
    ax.set_ylim(0, env.H)
    ax.set_aspect("equal")
    ax.set_title("Smart Building Floor Plan", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
    plt.tight_layout()
    return fig


def plot_coverage_interactive(env, solution, title="Coverage Heatmap"):
    """Plot coverage heatmap for a solution."""
    selected = list(np.where(solution == 1)[0])
    cov_map = np.zeros((env.W, env.H))
    for idx in selected:
        for (cx, cy) in env.coverage_cells[idx]:
            cov_map[cx, cy] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        cov_map.T, origin="lower", cmap="YlOrRd",
        extent=[0, env.W, 0, env.H], aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Coverage count")
    
    # Obstacles
    for (x1, y1, x2, y2) in env.obstacles:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0, facecolor="#2c3e50", alpha=0.9,
        )
        ax.add_patch(rect)
    
    # Selected sensors
    for idx in selected:
        s = env.sensors[idx]
        color = SENSOR_TYPES[s.stype]["color"]
        ax.scatter(s.x, s.y, c=color, s=80, zorder=6, edgecolors="black", linewidths=1)
        circle = plt.Circle((s.x, s.y), s.radius, color=color, fill=False,
                           alpha=0.3, linewidth=1)
        ax.add_patch(circle)
    
    ax.set_xlim(0, env.W)
    ax.set_ylim(0, env.H)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    return fig


def plot_convergence_interactive(results):
    """Plot convergence curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ALGO_COLORS = {
        "GA": "#2ecc71",
        "SA": "#e74c3c",
        "PSO": "#3498db",
        "SAGA": "#9b59b6",
    }
    
    for algo, res in results.items():
        color = ALGO_COLORS.get(algo, "black")
        history = res.get("history", [])
        
        if not history:
            continue
        
        xs = [h.get("generation", h.get("iteration", i)) for i, h in enumerate(history)]
        ys = [h["best_fitness"] for h in history]
        cov = [h["f1_coverage"] for h in history]
        
        axes[0].plot(xs, ys, label=algo, color=color, linewidth=2)
        axes[1].plot(xs, cov, label=algo, color=color, linewidth=2)
    
    axes[0].set_title("Best Fitness over Iterations", fontsize=12)
    axes[0].set_xlabel("Iteration / Generation")
    axes[0].set_ylabel("Best Fitness (lower = better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Coverage Progression", fontsize=12)
    axes[1].set_xlabel("Iteration / Generation")
    axes[1].set_ylabel("Coverage (f1)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pareto_interactive(pareto_data):
    """Plot Pareto front."""
    all_pts = pareto_data["all_points"]
    pareto_pts = pareto_data["pareto_points"]
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    ax.scatter(
        [p[0] for p in all_pts],
        [p[1] for p in all_pts],
        c="#bdc3c7", s=80, label="All solutions", zorder=3, alpha=0.6,
    )
    
    pareto_sorted = sorted(pareto_pts, key=lambda p: p[0])
    ax.scatter(
        [p[0] for p in pareto_sorted],
        [p[1] for p in pareto_sorted],
        c="#e74c3c", s=150, zorder=5, label="Pareto front", 
        edgecolors="black", linewidths=1.5,
    )
    ax.plot(
        [p[0] for p in pareto_sorted],
        [p[1] for p in pareto_sorted],
        c="#e74c3c", linewidth=2, linestyle="--",
    )
    
    ax.set_xlabel("Coverage (f1) — higher is better", fontsize=12)
    ax.set_ylabel("Cost (f2) — lower is better", fontsize=12)
    ax.set_title("Pareto Front: Coverage vs Cost Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">📡 Sensor Selection Optimizer</div>', unsafe_allow_html=True)
st.markdown("**Metaheuristic Optimization for Smart Building Sensor Deployment**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Environment")
    n_sensors = st.slider("Number of candidate sensors", 50, 200, 150, 10)
    seed = st.number_input("Random seed", 0, 1000, 42, 1)
    
    st.subheader("Objectives")
    alpha = st.slider("α (Coverage weight)", 0.0, 1.0, 0.60, 0.05)
    beta = st.slider("β (Cost weight)", 0.0, 1.0, 0.25, 0.05)
    gamma = st.slider("γ (Overlap penalty)", 0.0, 1.0, 0.15, 0.05)
    
    # Normalize weights
    total = alpha + beta + gamma
    if total > 0:
        alpha, beta, gamma = alpha/total, beta/total, gamma/total
    
    st.subheader("Budget")
    budget_pct = st.slider("Budget (% of total cost)", 20, 100, 50, 5)
    
    st.subheader("Algorithms")
    run_ga = st.checkbox("Genetic Algorithm (GA)", value=True)
    run_sa = st.checkbox("Simulated Annealing (SA)", value=True)
    run_pso = st.checkbox("Binary PSO", value=True)
    run_saga = st.checkbox("Hybrid SAGA", value=False)
    
    st.subheader("Performance")
    quick_mode = st.checkbox("Quick mode (faster, less accurate)", value=True)
    
    if st.button("🚀 Generate Environment & Run"):
        with st.spinner("Building environment..."):
            st.session_state.env = BuildingEnvironment(n_sensors=n_sensors, seed=seed)
            budget = st.session_state.env.budget_default() * (budget_pct / 50.0)
            st.session_state.evaluator = FitnessEvaluator(
                st.session_state.env, 
                budget=budget,
                alpha=alpha, 
                beta=beta, 
                gamma=gamma
            )
        st.success("✅ Environment ready!")

# Main content
if st.session_state.env is None:
    st.info("👈 Configure parameters in the sidebar and click **Generate Environment & Run** to start.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏢 Environment", 
    "🎯 Run Algorithms", 
    "📊 Results", 
    "📈 Pareto Front",
    "💾 Export"
])

# ---------------------------------------------------------------------------
# Tab 1: Environment
# ---------------------------------------------------------------------------
with tab1:
    st.header("Building Environment")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Grid Size", f"{st.session_state.env.W}×{st.session_state.env.H}")
    with col2:
        st.metric("Zones", len(st.session_state.env.zones))
    with col3:
        st.metric("Obstacles", len(st.session_state.env.obstacles))
    with col4:
        st.metric("Candidate Sensors", st.session_state.env.n_sensors)
    
    st.subheader("Floor Plan")
    fig = plot_building_interactive(st.session_state.env)
    st.pyplot(fig)
    
    st.subheader("Zone Details")
    zone_data = []
    for z in st.session_state.env.zones:
        zone_data.append({
            "Zone": z.name,
            "Priority": z.priority,
            "Critical": "Yes" if z.critical else "No",
            "Area": f"{(z.x2-z.x1)*(z.y2-z.y1)} cells"
        })
    st.dataframe(pd.DataFrame(zone_data), width="stretch")
    
    st.subheader("Sensor Type Distribution")
    type_counts = {}
    for s in st.session_state.env.sensors:
        type_counts[s.stype] = type_counts.get(s.stype, 0) + 1
    st.bar_chart(pd.DataFrame.from_dict(type_counts, orient='index', columns=['Count']))

# ---------------------------------------------------------------------------
# Tab 2: Run Algorithms
# ---------------------------------------------------------------------------
with tab2:
    st.header("Algorithm Execution")
    
    if st.button("▶️ Run Selected Algorithms", type="primary"):
        env = st.session_state.env
        ev = st.session_state.evaluator
        results = {}
        
        # Hyperparameters
        if quick_mode:
            ga_pop, ga_gen = 50, 100
            sa_iter = 10_000
            pso_part, pso_iter = 40, 100
        else:
            ga_pop, ga_gen = 100, 300
            sa_iter = 50_000
            pso_part, pso_iter = 80, 300
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_algos = sum([run_ga, run_sa, run_pso, run_saga])
        current = 0
        
        # GA
        if run_ga:
            status_text.text("Running Genetic Algorithm...")
            ga = GeneticAlgorithm(ev, pop_size=ga_pop, n_generations=ga_gen, seed=0)
            results["GA"] = ga.run(verbose=False)
            current += 1
            progress_bar.progress(current / total_algos)
        
        # SA
        if run_sa:
            status_text.text("Running Simulated Annealing...")
            sa = SimulatedAnnealing(ev, max_iter=sa_iter, seed=1)
            results["SA"] = sa.run(verbose=False)
            current += 1
            progress_bar.progress(current / total_algos)
        
        # PSO
        if run_pso:
            status_text.text("Running Binary PSO...")
            pso = BinaryPSO(ev, n_particles=pso_part, n_iterations=pso_iter, seed=2)
            results["PSO"] = pso.run(verbose=False)
            current += 1
            progress_bar.progress(current / total_algos)
        
        # SAGA
        if run_saga:
            status_text.text("Running Hybrid SAGA...")
            saga = HybridSAGA(ev, ga_generations=ga_gen//2, sa_max_iter=sa_iter//2, seed=3)
            results["SAGA"] = saga.run(verbose=False)
            current += 1
            progress_bar.progress(current / total_algos)
        
        st.session_state.results = results
        status_text.text("✅ All algorithms completed!")
        progress_bar.progress(1.0)
        st.success(f"Completed {len(results)} algorithm(s)")
        st.balloons()

# ---------------------------------------------------------------------------
# Tab 3: Results
# ---------------------------------------------------------------------------
with tab3:
    st.header("Algorithm Results")
    
    if not st.session_state.results:
        st.info("Run algorithms in the **Run Algorithms** tab first.")
        st.stop()
    
    results = st.session_state.results
    
    # Summary table
    st.subheader("Performance Summary")
    summary_data = []
    for algo, res in results.items():
        info = res["best_info"]
        summary_data.append({
            "Algorithm": algo,
            "Fitness": f"{res['best_fitness']:.4f}",
            "Coverage": f"{info['f1_coverage']:.4f}",
            "Cost": f"{info['f2_cost']:.4f}",
            "Overlap": f"{info['f3_overlap']:.4f}",
            "Sensors": info['n_selected'],
            "Feasible": "✅" if info['feasible'] else "❌",
            "Time (s)": f"{res['elapsed']:.1f}"
        })
    st.dataframe(pd.DataFrame(summary_data), width="stretch")
    
    # Best algorithm
    best_algo = min(results, key=lambda a: results[a]["best_fitness"])
    st.success(f"🏆 Best Algorithm: **{best_algo}** (fitness: {results[best_algo]['best_fitness']:.4f})")
    
    # Convergence plot
    st.subheader("Convergence Curves")
    fig = plot_convergence_interactive(results)
    st.pyplot(fig)
    
    # Individual solutions
    st.subheader("Solution Visualizations")
    algo_select = st.selectbox("Select algorithm to visualize", list(results.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Coverage Heatmap**")
        fig = plot_coverage_interactive(
            st.session_state.env,
            results[algo_select]["best_solution"],
            title=f"{algo_select} — Coverage"
        )
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Solution Details**")
        info = results[algo_select]["best_info"]
        st.metric("Coverage (f1)", f"{info['f1_coverage']:.4f}")
        st.metric("Cost (f2)", f"{info['f2_cost']:.4f}")
        st.metric("Overlap (f3)", f"{info['f3_overlap']:.4f}")
        st.metric("Sensors Selected", info['n_selected'])
        st.metric("Raw Cost", f"{info['cost_raw']:.1f}")
        st.metric("Covered Cells", info['n_cells'])
        
        if info['feasible']:
            st.success("✅ All constraints satisfied")
        else:
            st.error("❌ Constraint violations detected")
            if info['budget_viol'] > 0:
                st.warning(f"Budget violation: {info['budget_viol']:.4f}")
            if info['redundancy_pen'] > 0:
                st.warning(f"Redundancy penalty: {info['redundancy_pen']:.2f}")
            if info['connect_pen'] > 0:
                st.warning(f"Connectivity penalty: {info['connect_pen']:.2f}")

# ---------------------------------------------------------------------------
# Tab 4: Pareto Front
# ---------------------------------------------------------------------------
with tab4:
    st.header("Pareto Front Analysis")
    
    st.markdown("""
    The Pareto front shows the tradeoff between **coverage** and **cost**. 
    Each point represents a solution with different objective weights.
    """)
    
    n_steps = st.slider("Number of weight configurations", 4, 12, 6)
    
    if st.button("🔍 Generate Pareto Front"):
        with st.spinner("Running Pareto analysis..."):
            pareto_data = generate_pareto_front(
                st.session_state.env,
                budget=st.session_state.evaluator.budget,
                n_weight_steps=n_steps,
                ga_pop=50,
                ga_gen=80,
                verbose=False
            )
            st.session_state.pareto_data = pareto_data
        st.success(f"✅ Found {len(pareto_data['pareto_points'])} Pareto-optimal solutions")
    
    if st.session_state.pareto_data:
        fig = plot_pareto_interactive(st.session_state.pareto_data)
        st.pyplot(fig)
        
        st.subheader("Pareto-Optimal Solutions")
        pareto_df = pd.DataFrame(
            st.session_state.pareto_data['pareto_points'],
            columns=['Coverage (f1)', 'Cost (f2)']
        )
        st.dataframe(pareto_df, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5: Export
# ---------------------------------------------------------------------------
with tab5:
    st.header("Export Results")
    
    if not st.session_state.results:
        st.info("No results to export. Run algorithms first.")
        st.stop()
    
    st.subheader("Download Options")
    
    # Export summary as CSV
    summary_data = []
    for algo, res in st.session_state.results.items():
        info = res["best_info"]
        summary_data.append({
            "Algorithm": algo,
            "Fitness": res['best_fitness'],
            "Coverage_f1": info['f1_coverage'],
            "Cost_f2": info['f2_cost'],
            "Overlap_f3": info['f3_overlap'],
            "Sensors_Selected": info['n_selected'],
            "Raw_Cost": info['cost_raw'],
            "Covered_Cells": info['n_cells'],
            "Feasible": info['feasible'],
            "Time_seconds": res['elapsed']
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Summary (CSV)",
        data=csv,
        file_name="sensor_selection_summary.csv",
        mime="text/csv"
    )
    
    # Export convergence histories
    for algo, res in st.session_state.results.items():
        history = res.get("history", [])
        if history:
            hist_df = pd.DataFrame(history)
            csv = hist_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {algo} History (CSV)",
                data=csv,
                file_name=f"{algo.lower()}_history.csv",
                mime="text/csv"
            )
    
    # Export Pareto data
    if st.session_state.pareto_data:
        pareto_json = json.dumps(st.session_state.pareto_data, indent=2)
        st.download_button(
            label="📥 Download Pareto Data (JSON)",
            data=pareto_json,
            file_name="pareto_data.json",
            mime="application/json"
        )
    
    st.subheader("Configuration Summary")
    config = {
        "n_sensors": st.session_state.env.n_sensors,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "budget_pct": budget_pct,
        "quick_mode": quick_mode,
    }
    st.json(config)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>Sensor Selection Problem — Optimization Techniques Course Project</p>
    <p>Built with Streamlit | Algorithms: GA, SA, PSO, SAGA</p>
</div>
""", unsafe_allow_html=True)
