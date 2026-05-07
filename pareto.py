"""
pareto.py
=========
Pareto front generation and analysis for the Sensor Selection Problem.

Approach
--------
  Vary the objective weights (alpha, beta, gamma) systematically across
  a grid of values and run a fast GA for each weight combination.
  Collect (f1_coverage, f2_cost) pairs and extract the Pareto-optimal front.

This reveals the true tradeoff surface between coverage and cost.
"""

import numpy as np
import itertools
import time
from typing import List, Tuple
from environment import BuildingEnvironment
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm


def is_dominated(p: Tuple, others: List[Tuple]) -> bool:
    """Return True if point p is dominated by any point in others.
    p = (f1, f2) where f1 is to be maximised and f2 minimised.
    We convert to minimisation: (-f1, f2).
    """
    for q in others:
        if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
            return True
    return False


def extract_pareto_front(points: List[Tuple]) -> List[Tuple]:
    """Extract non-dominated points from a list of (neg_f1, f2) tuples."""
    pareto = []
    for i, p in enumerate(points):
        others = [points[j] for j in range(len(points)) if j != i]
        if not is_dominated(p, others):
            pareto.append(p)
    return pareto


def generate_pareto_front(
    env: BuildingEnvironment,
    budget: float,
    n_weight_steps: int = 6,
    ga_pop: int = 60,
    ga_gen: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run GA with varying (alpha, beta) weights and collect Pareto front.

    Parameters
    ----------
    n_weight_steps : number of steps for alpha in [0.1, 0.9]
    ga_pop, ga_gen : GA hyperparameters (kept small for speed)

    Returns
    -------
    dict with 'all_points', 'pareto_points', 'weight_configs'
    """
    alphas = np.linspace(0.1, 0.9, n_weight_steps)
    betas  = 1.0 - alphas
    gammas = np.zeros_like(alphas)   # fix gamma=0 for 2D Pareto

    all_points: List[Tuple] = []
    weight_configs: List[dict] = []
    t0 = time.time()

    for i, (a, b, g) in enumerate(zip(alphas, betas, gammas)):
        if verbose:
            print(f"  Pareto run {i+1}/{n_weight_steps} | alpha={a:.2f} beta={b:.2f}")
        ev = FitnessEvaluator(env, budget=budget, alpha=a, beta=b, gamma=g)
        ga = GeneticAlgorithm(ev, pop_size=ga_pop, n_generations=ga_gen, seed=i*7)
        result = ga.run(verbose=False)
        info = result["best_info"]
        all_points.append((info["f1_coverage"], info["f2_cost"]))
        weight_configs.append({"alpha": a, "beta": b, "gamma": g, "info": info})

    # convert to minimisation space for Pareto extraction: (-f1, f2)
    min_points = [(-p[0], p[1]) for p in all_points]
    pareto_min = extract_pareto_front(min_points)
    # convert back
    pareto_points = [(-p[0], p[1]) for p in pareto_min]

    elapsed = time.time() - t0
    if verbose:
        print(f"  Pareto front: {len(pareto_points)} non-dominated solutions in {elapsed:.1f}s")

    return {
        "all_points":     all_points,
        "pareto_points":  pareto_points,
        "weight_configs": weight_configs,
        "elapsed":        elapsed,
    }
