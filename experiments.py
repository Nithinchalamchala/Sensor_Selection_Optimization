"""
experiments.py
==============
Sensitivity analysis experiments for the Sensor Selection Problem.

Experiments
-----------
  1. Budget sensitivity   : vary budget from 30% to 80% of total cost
  2. Weight sensitivity   : vary alpha (coverage weight) from 0.2 to 0.8
  3. N-sensor sensitivity : vary number of candidate sensors (50 to 200)

Each experiment runs GA (fast settings) and records f1, f2, fitness.
"""

import numpy as np
from typing import List, Dict
from environment import BuildingEnvironment
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm


def _quick_ga(ev: FitnessEvaluator, seed: int = 99) -> dict:
    """Run a quick GA for sensitivity experiments."""
    ga = GeneticAlgorithm(
        evaluator=ev,
        pop_size=60,
        n_generations=100,
        seed=seed,
    )
    return ga.run(verbose=False)


# ---------------------------------------------------------------------------
# 1. Budget sensitivity
# ---------------------------------------------------------------------------
def budget_sensitivity(
    env: BuildingEnvironment,
    fractions: List[float] = None,
    verbose: bool = True,
) -> List[dict]:
    if fractions is None:
        fractions = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    total_cost = sum(s.cost for s in env.sensors)
    results = []

    for frac in fractions:
        budget = total_cost * frac
        ev = FitnessEvaluator(env, budget=budget)
        res = _quick_ga(ev, seed=int(frac * 100))
        info = res["best_info"]
        entry = {
            "param_value": frac,
            "budget":      budget,
            "f1":          info["f1_coverage"],
            "f2":          info["f2_cost"],
            "fitness":     res["best_fitness"],
            "n_selected":  info["n_selected"],
            "feasible":    info["feasible"],
        }
        results.append(entry)
        if verbose:
            print(f"  Budget {frac*100:.0f}% | f1={info['f1_coverage']:.3f} "
                  f"| f2={info['f2_cost']:.3f} | n={info['n_selected']}")

    return results


# ---------------------------------------------------------------------------
# 2. Weight (alpha) sensitivity
# ---------------------------------------------------------------------------
def weight_sensitivity(
    env: BuildingEnvironment,
    budget: float = None,
    alphas: List[float] = None,
    verbose: bool = True,
) -> List[dict]:
    if alphas is None:
        alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if budget is None:
        budget = env.budget_default()

    results = []
    for alpha in alphas:
        beta  = (1 - alpha) * 0.7
        gamma = (1 - alpha) * 0.3
        ev = FitnessEvaluator(env, budget=budget, alpha=alpha, beta=beta, gamma=gamma)
        res = _quick_ga(ev, seed=int(alpha * 100))
        info = res["best_info"]
        entry = {
            "param_value": alpha,
            "alpha":       alpha,
            "beta":        beta,
            "gamma":       gamma,
            "f1":          info["f1_coverage"],
            "f2":          info["f2_cost"],
            "fitness":     res["best_fitness"],
            "n_selected":  info["n_selected"],
        }
        results.append(entry)
        if verbose:
            print(f"  alpha={alpha:.1f} | f1={info['f1_coverage']:.3f} "
                  f"| f2={info['f2_cost']:.3f} | n={info['n_selected']}")

    return results


# ---------------------------------------------------------------------------
# 3. N-sensor sensitivity
# ---------------------------------------------------------------------------
def nsensor_sensitivity(
    n_values: List[int] = None,
    verbose: bool = True,
) -> List[dict]:
    if n_values is None:
        n_values = [50, 75, 100, 125, 150, 175, 200]

    results = []
    for n in n_values:
        env = BuildingEnvironment(n_sensors=n, seed=42)
        budget = env.budget_default()
        ev = FitnessEvaluator(env, budget=budget)
        res = _quick_ga(ev, seed=n)
        info = res["best_info"]
        entry = {
            "param_value": n,
            "f1":          info["f1_coverage"],
            "f2":          info["f2_cost"],
            "fitness":     res["best_fitness"],
            "n_selected":  info["n_selected"],
            "elapsed":     res["elapsed"],
        }
        results.append(entry)
        if verbose:
            print(f"  N={n:3d} | f1={info['f1_coverage']:.3f} "
                  f"| f2={info['f2_cost']:.3f} | n_sel={info['n_selected']} "
                  f"| t={res['elapsed']:.1f}s")

    return results
