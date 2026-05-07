"""
main.py
=======
End-to-end orchestrator for the Sensor Selection Problem project.

Usage
-----
  python main.py                  # full run (all algorithms + experiments)
  python main.py --quick          # fast run for testing
  python main.py --algo GA        # run only one algorithm
  python main.py --no-pareto      # skip Pareto front generation
  python main.py --no-sensitivity # skip sensitivity analysis

Output
------
  results/          CSV logs of convergence histories
  report_figures/   All publication-quality plots
"""

import argparse
import os
import csv
import json
import time
import numpy as np

from environment import BuildingEnvironment
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm
from sa import SimulatedAnnealing
from pso import BinaryPSO
from hybrid_saga import HybridSAGA
from pareto import generate_pareto_front
from experiments import budget_sensitivity, weight_sensitivity, nsensor_sensitivity
from visualization import (
    plot_building,
    plot_coverage,
    plot_connectivity,
    plot_convergence,
    plot_pareto,
    plot_sensitivity,
    plot_comparison_summary,
)

RESULTS_DIR = "results"
FIGURES_DIR = "report_figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def save_history_csv(history: list, path: str):
    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"  Saved CSV: {path}")


def print_result_summary(result: dict):
    info = result["best_info"]
    print(f"\n  ── {result['algorithm']} Results ──────────────────────────")
    print(f"  Best fitness   : {result['best_fitness']:.6f}")
    print(f"  Coverage (f1)  : {info['f1_coverage']:.4f}  ({info['n_cells']} cells)")
    print(f"  Cost (f2)      : {info['f2_cost']:.4f}  (raw: {info['cost_raw']:.1f})")
    print(f"  Overlap (f3)   : {info['f3_overlap']:.4f}")
    print(f"  Sensors selected: {info['n_selected']}")
    print(f"  Feasible       : {info['feasible']}")
    print(f"  Budget viol.   : {info['budget_viol']:.4f}")
    print(f"  Redundancy pen.: {info['redundancy_pen']:.2f}")
    print(f"  Connect. pen.  : {info['connect_pen']:.2f}")
    print(f"  Time           : {result['elapsed']:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sensor Selection Optimisation")
    parser.add_argument("--quick",          action="store_true", help="Fast run for testing")
    parser.add_argument("--algo",           type=str, default="ALL",
                        choices=["GA", "SA", "PSO", "SAGA", "ALL"])
    parser.add_argument("--no-pareto",      action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--n-sensors",      type=int, default=150)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    t_total = time.time()

    # ── quick-mode overrides ──────────────────────────────────────────────
    if args.quick:
        GA_POP, GA_GEN   = 40, 80
        SA_ITER          = 10_000
        PSO_PART, PSO_IT = 30, 80
        SAGA_GA_GEN      = 50
        SAGA_SA_ITER     = 8_000
        PARETO_STEPS     = 4
        print("⚡  Quick mode enabled (reduced iterations)")
    else:
        GA_POP, GA_GEN   = 100, 300
        SA_ITER          = 50_000
        PSO_PART, PSO_IT = 80, 300
        SAGA_GA_GEN      = 150
        SAGA_SA_ITER     = 30_000
        PARETO_STEPS     = 8

    # ── Environment ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SENSOR SELECTION PROBLEM — Smart Building Optimisation")
    print("="*60)
    print(f"\n[1/7] Building environment (N={args.n_sensors} sensors) ...")
    env = BuildingEnvironment(n_sensors=args.n_sensors, seed=args.seed)
    env.summary()
    budget = env.budget_default()
    print(f"  Default budget: {budget:.1f}")

    # ── Fitness evaluator ─────────────────────────────────────────────────
    ev = FitnessEvaluator(env, budget=budget)

    # ── Building map ──────────────────────────────────────────────────────
    print("\n[2/7] Generating building map ...")
    plot_building(env, save_path=f"{FIGURES_DIR}/01_building_map.png")

    # ── Run algorithms ────────────────────────────────────────────────────
    print("\n[3/7] Running optimisation algorithms ...")
    results = {}

    if args.algo in ("GA", "ALL"):
        print("\n  ▶ Genetic Algorithm")
        ga = GeneticAlgorithm(ev, pop_size=GA_POP, n_generations=GA_GEN, seed=0)
        results["GA"] = ga.run(verbose=True)
        save_history_csv(results["GA"]["history"],
                         f"{RESULTS_DIR}/ga_history.csv")
        print_result_summary(results["GA"])

    if args.algo in ("SA", "ALL"):
        print("\n  ▶ Simulated Annealing")
        sa = SimulatedAnnealing(ev, max_iter=SA_ITER, seed=1)
        results["SA"] = sa.run(verbose=True)
        save_history_csv(results["SA"]["history"],
                         f"{RESULTS_DIR}/sa_history.csv")
        print_result_summary(results["SA"])

    if args.algo in ("PSO", "ALL"):
        print("\n  ▶ Binary PSO with Lévy Flight")
        pso = BinaryPSO(ev, n_particles=PSO_PART, n_iterations=PSO_IT, seed=2)
        results["PSO"] = pso.run(verbose=True)
        save_history_csv(results["PSO"]["history"],
                         f"{RESULTS_DIR}/pso_history.csv")
        print_result_summary(results["PSO"])

    if args.algo in ("SAGA", "ALL"):
        print("\n  ▶ Hybrid GA+SA (SAGA)")
        saga = HybridSAGA(
            ev,
            ga_generations=SAGA_GA_GEN,
            sa_max_iter=SAGA_SA_ITER,
            seed=3,
        )
        results["SAGA"] = saga.run(verbose=True)
        save_history_csv(results["SAGA"].get("history_ga", []),
                         f"{RESULTS_DIR}/saga_ga_history.csv")
        save_history_csv(results["SAGA"].get("history_sa", []),
                         f"{RESULTS_DIR}/saga_sa_history.csv")
        print_result_summary(results["SAGA"])

    # ── Visualisations ────────────────────────────────────────────────────
    print("\n[4/7] Generating solution visualisations ...")

    # find best overall
    best_algo = min(results, key=lambda a: results[a]["best_fitness"])
    best_sol  = results[best_algo]["best_solution"]
    print(f"  Best overall algorithm: {best_algo}")

    for algo, res in results.items():
        sol = res["best_solution"]
        plot_coverage(
            env, sol,
            title=f"{algo} — Best Solution",
            save_path=f"{FIGURES_DIR}/02_coverage_{algo.lower()}.png",
        )
        plot_connectivity(
            env, sol,
            title=f"{algo} — Sensor Network",
            save_path=f"{FIGURES_DIR}/03_connectivity_{algo.lower()}.png",
        )

    # ── Convergence ───────────────────────────────────────────────────────
    print("\n[5/7] Plotting convergence curves ...")
    plot_convergence(results, save_path=f"{FIGURES_DIR}/04_convergence.png")
    plot_comparison_summary(results, save_path=f"{FIGURES_DIR}/05_comparison_summary.png")

    # ── Pareto front ──────────────────────────────────────────────────────
    if not args.no_pareto:
        print("\n[6/7] Generating Pareto front ...")
        pareto_data = generate_pareto_front(
            env, budget=budget,
            n_weight_steps=PARETO_STEPS,
            ga_pop=50, ga_gen=80,
            verbose=True,
        )
        plot_pareto(pareto_data, save_path=f"{FIGURES_DIR}/06_pareto_front.png")
        # save pareto data
        with open(f"{RESULTS_DIR}/pareto_data.json", "w") as f:
            json.dump({
                "all_points":    pareto_data["all_points"],
                "pareto_points": pareto_data["pareto_points"],
            }, f, indent=2)
    else:
        print("\n[6/7] Pareto front skipped.")

    # ── Sensitivity analysis ──────────────────────────────────────────────
    if not args.no_sensitivity:
        print("\n[7/7] Running sensitivity analysis ...")

        print("  Budget sensitivity ...")
        budget_data = budget_sensitivity(env, verbose=True)

        print("  Weight sensitivity ...")
        weight_data = weight_sensitivity(env, budget=budget, verbose=True)

        print("  N-sensor sensitivity ...")
        nsensor_data = nsensor_sensitivity(verbose=True)

        sensitivity_data = {
            "budget":    budget_data,
            "weights":   weight_data,
            "n_sensors": nsensor_data,
        }
        plot_sensitivity(sensitivity_data,
                         save_path=f"{FIGURES_DIR}/07_sensitivity.png")

        # save CSVs
        for key, data in sensitivity_data.items():
            save_history_csv(data, f"{RESULTS_DIR}/sensitivity_{key}.csv")
    else:
        print("\n[7/7] Sensitivity analysis skipped.")

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Algorithm':<10} {'Fitness':>10} {'Coverage':>10} {'Cost':>10} {'#Sensors':>10} {'Time(s)':>10}")
    print("  " + "-"*58)
    for algo, res in results.items():
        info = res["best_info"]
        print(
            f"  {algo:<10} {res['best_fitness']:>10.4f} "
            f"{info['f1_coverage']:>10.4f} {info['f2_cost']:>10.4f} "
            f"{info['n_selected']:>10} {res['elapsed']:>10.1f}"
        )
    print(f"\n  Total runtime: {elapsed_total:.1f}s")
    print(f"  Figures saved to: {FIGURES_DIR}/")
    print(f"  Data saved to:    {RESULTS_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
