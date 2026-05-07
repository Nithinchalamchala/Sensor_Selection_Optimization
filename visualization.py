"""
visualization.py
================
All plotting utilities for the Sensor Selection project.

Plots generated
---------------
  1. Building map with zones, obstacles, and sensor placements
  2. Coverage heatmap for a given solution
  3. Sensor network connectivity graph
  4. Convergence curves (all algorithms on one plot)
  5. Pareto front (coverage vs cost)
  6. Sensitivity analysis bar charts
  7. Algorithm comparison radar chart
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import networkx as nx
from typing import List, Dict, Optional
import os

from environment import BuildingEnvironment, SENSOR_TYPES, COMM_RANGE

# colour palette
ZONE_COLORS = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7",
    "#D2B4DE", "#FDFEFE", "#ABB2B9", "#F1948A",
]
ALGO_COLORS = {
    "GA":   "#2ecc71",
    "SA":   "#e74c3c",
    "PSO":  "#3498db",
    "SAGA": "#9b59b6",
}


# ---------------------------------------------------------------------------
# 1. Building map
# ---------------------------------------------------------------------------
def plot_building(env: BuildingEnvironment, save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 9))

    # zones
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
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7, fontweight="bold")

    # obstacles
    for (x1, y1, x2, y2) in env.obstacles:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0, facecolor="#2c3e50", alpha=0.85,
        )
        ax.add_patch(rect)

    # sensors
    for s in env.sensors:
        color = SENSOR_TYPES[s.stype]["color"]
        ax.scatter(s.x, s.y, c=color, s=25, zorder=5, alpha=0.7)

    # legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=SENSOR_TYPES[t]["color"],
               markersize=8, label=t.capitalize())
        for t in SENSOR_TYPES
    ]
    legend_elements.append(
        mpatches.Patch(facecolor="#2c3e50", label="Obstacle/Wall")
    )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlim(0, env.W)
    ax.set_ylim(0, env.H)
    ax.set_aspect("equal")
    ax.set_title("Smart Building Floor Plan — Candidate Sensors", fontsize=13)
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Coverage heatmap
# ---------------------------------------------------------------------------
def plot_coverage(
    env: BuildingEnvironment,
    solution: np.ndarray,
    title: str = "Coverage Heatmap",
    save_path: str = None,
):
    selected = list(np.where(solution == 1)[0])
    cov_map = np.zeros((env.W, env.H))
    for idx in selected:
        for (cx, cy) in env.coverage_cells[idx]:
            cov_map[cx, cy] += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # left: heatmap
    ax = axes[0]
    im = ax.imshow(
        cov_map.T, origin="lower", cmap="YlOrRd",
        extent=[0, env.W, 0, env.H], aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Coverage count")

    # overlay obstacles
    for (x1, y1, x2, y2) in env.obstacles:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0, facecolor="#2c3e50", alpha=0.9,
        )
        ax.add_patch(rect)

    # selected sensors
    for idx in selected:
        s = env.sensors[idx]
        color = SENSOR_TYPES[s.stype]["color"]
        ax.scatter(s.x, s.y, c=color, s=60, zorder=6, edgecolors="black", linewidths=0.5)
        circle = plt.Circle((s.x, s.y), s.radius, color=color, fill=False,
                             alpha=0.3, linewidth=0.8)
        ax.add_patch(circle)

    ax.set_xlim(0, env.W); ax.set_ylim(0, env.H)
    ax.set_title(f"{title} — Coverage Heatmap")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    # right: priority-weighted coverage per zone
    ax2 = axes[1]
    zone_names, zone_covs, zone_pris = [], [], []
    for z in env.zones:
        zone_cells = env.get_zone_cells(z)
        covered = sum(1 for c in zone_cells if cov_map[c[0], c[1]] > 0)
        pct = covered / len(zone_cells) * 100 if zone_cells else 0
        zone_names.append(z.name)
        zone_covs.append(pct)
        zone_pris.append(z.priority)

    colors = ["#e74c3c" if p >= 3 else "#e67e22" if p >= 2 else "#3498db"
              for p in zone_pris]
    bars = ax2.barh(zone_names, zone_covs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Coverage (%)")
    ax2.set_title("Zone Coverage Breakdown")
    ax2.set_xlim(0, 105)
    for bar, val in zip(bars, zone_covs):
        ax2.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=8)

    legend_p = [
        mpatches.Patch(color="#e74c3c", label="High priority (≥3)"),
        mpatches.Patch(color="#e67e22", label="Medium priority (≥2)"),
        mpatches.Patch(color="#3498db", label="Low priority (<2)"),
    ]
    ax2.legend(handles=legend_p, fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Connectivity graph
# ---------------------------------------------------------------------------
def plot_connectivity(
    env: BuildingEnvironment,
    solution: np.ndarray,
    title: str = "Sensor Network",
    save_path: str = None,
):
    selected = list(np.where(solution == 1)[0])
    sensors = env.sensors

    G = nx.Graph()
    for idx in selected:
        G.add_node(idx, pos=(sensors[idx].x, sensors[idx].y),
                   stype=sensors[idx].stype)

    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            si, sj = sensors[selected[i]], sensors[selected[j]]
            dist = np.hypot(si.x - sj.x, si.y - sj.y)
            if dist <= COMM_RANGE:
                G.add_edge(selected[i], selected[j], weight=dist)

    pos = {idx: (sensors[idx].x, sensors[idx].y) for idx in selected}
    node_colors = [SENSOR_TYPES[sensors[idx].stype]["color"] for idx in selected]

    # connected components
    components = list(nx.connected_components(G))
    n_comp = len(components)

    fig, ax = plt.subplots(figsize=(9, 9))

    # background: zones
    for i, z in enumerate(env.zones):
        rect = mpatches.Rectangle(
            (z.x1, z.y1), z.x2 - z.x1, z.y2 - z.y1,
            linewidth=1, edgecolor="grey",
            facecolor=ZONE_COLORS[i % len(ZONE_COLORS)], alpha=0.2,
        )
        ax.add_patch(rect)

    # obstacles
    for (x1, y1, x2, y2) in env.obstacles:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0, facecolor="#2c3e50", alpha=0.85,
        )
        ax.add_patch(rect)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color="#7f8c8d", width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=80, edgecolors="black", linewidths=0.5)

    ax.set_xlim(0, env.W); ax.set_ylim(0, env.H)
    ax.set_aspect("equal")
    ax.set_title(
        f"{title}\n{len(selected)} sensors | {G.number_of_edges()} links | "
        f"{n_comp} component{'s' if n_comp > 1 else ''}",
        fontsize=11,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=SENSOR_TYPES[t]["color"],
               markersize=9, label=t.capitalize())
        for t in SENSOR_TYPES
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Convergence curves
# ---------------------------------------------------------------------------
def plot_convergence(results: Dict, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for algo, res in results.items():
        color = ALGO_COLORS.get(algo, "black")
        history = res.get("history") or res.get("history_sa") or []

        if algo == "SAGA":
            # combine GA + SA history
            ga_h = res.get("history_ga", [])
            sa_h = res.get("history_sa", [])
            # normalise x-axis to [0,1]
            n_ga = len(ga_h)
            n_sa = len(sa_h)
            total = n_ga + n_sa
            xs = [i / total for i in range(n_ga)] + \
                 [(n_ga + i) / total for i in range(n_sa)]
            ys = [h["best_fitness"] for h in ga_h] + \
                 [h["best_fitness"] for h in sa_h]
            axes[0].plot(xs, ys, label=algo, color=color, linewidth=1.8)
            axes[1].plot(xs, [h["f1_coverage"] for h in ga_h] +
                         [h["f1_coverage"] for h in sa_h],
                         label=algo, color=color, linewidth=1.8)
        else:
            xs = [h.get("generation", h.get("iteration", i))
                  for i, h in enumerate(history)]
            ys = [h["best_fitness"] for h in history]
            cov = [h["f1_coverage"] for h in history]
            axes[0].plot(xs, ys, label=algo, color=color, linewidth=1.8)
            axes[1].plot(xs, cov, label=algo, color=color, linewidth=1.8)

    axes[0].set_title("Convergence — Best Fitness over Iterations", fontsize=12)
    axes[0].set_xlabel("Iteration / Generation")
    axes[0].set_ylabel("Best Fitness (lower = better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Coverage Progression", fontsize=12)
    axes[1].set_xlabel("Iteration / Generation")
    axes[1].set_ylabel("f1 Coverage (normalised)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Algorithm Convergence Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Pareto front
# ---------------------------------------------------------------------------
def plot_pareto(pareto_data: dict, save_path: str = None):
    all_pts = pareto_data["all_points"]
    pareto_pts = pareto_data["pareto_points"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # all solutions
    ax.scatter(
        [p[0] for p in all_pts],
        [p[1] for p in all_pts],
        c="#bdc3c7", s=60, label="All solutions", zorder=3,
    )
    # pareto front
    pareto_sorted = sorted(pareto_pts, key=lambda p: p[0])
    ax.scatter(
        [p[0] for p in pareto_sorted],
        [p[1] for p in pareto_sorted],
        c="#e74c3c", s=100, zorder=5, label="Pareto front", edgecolors="black",
    )
    ax.plot(
        [p[0] for p in pareto_sorted],
        [p[1] for p in pareto_sorted],
        c="#e74c3c", linewidth=1.5, linestyle="--",
    )

    ax.set_xlabel("f1 — Coverage (higher = better)", fontsize=11)
    ax.set_ylabel("f2 — Cost (normalised, lower = better)", fontsize=11)
    ax.set_title("Pareto Front: Coverage vs Cost Tradeoff", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Sensitivity analysis
# ---------------------------------------------------------------------------
def plot_sensitivity(sensitivity_data: dict, save_path: str = None):
    """
    sensitivity_data keys: 'budget', 'weights', 'n_sensors'
    Each value is a list of dicts with 'param_value', 'f1', 'f2', 'fitness'
    """
    n_plots = len(sensitivity_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    titles = {
        "budget":   "Budget Sensitivity",
        "weights":  "Weight (α) Sensitivity",
        "n_sensors": "N Sensors Sensitivity",
    }

    for ax, (key, data) in zip(axes, sensitivity_data.items()):
        xs = [d["param_value"] for d in data]
        f1s = [d["f1"] for d in data]
        f2s = [d["f2"] for d in data]

        ax2 = ax.twinx()
        ax.plot(xs, f1s, "o-", color="#2ecc71", linewidth=2, label="Coverage (f1)")
        ax2.plot(xs, f2s, "s--", color="#e74c3c", linewidth=2, label="Cost (f2)")

        ax.set_xlabel(key.replace("_", " ").title())
        ax.set_ylabel("Coverage (f1)", color="#2ecc71")
        ax2.set_ylabel("Cost (f2)", color="#e74c3c")
        ax.set_title(titles.get(key, key))
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.suptitle("Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Algorithm comparison summary
# ---------------------------------------------------------------------------
def plot_comparison_summary(results: Dict, save_path: str = None):
    algos = list(results.keys())
    metrics = ["f1_coverage", "f2_cost", "n_selected", "elapsed"]
    labels  = ["Coverage", "Cost (norm)", "# Sensors", "Time (s)"]

    data = {m: [] for m in metrics}
    for algo in algos:
        info = results[algo]["best_info"]
        data["f1_coverage"].append(info["f1_coverage"])
        data["f2_cost"].append(info["f2_cost"])
        data["n_selected"].append(info["n_selected"])
        data["elapsed"].append(results[algo]["elapsed"])

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = [ALGO_COLORS.get(a, "grey") for a in algos]

    for ax, metric, label in zip(axes, metrics, labels):
        vals = data[metric]
        bars = ax.bar(algos, vals, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.3f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=9,
            )
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Algorithm Comparison Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save_or_show(fig, save_path: str):
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
