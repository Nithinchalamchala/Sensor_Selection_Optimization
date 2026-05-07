"""
hybrid_saga.py
==============
Hybrid GA + SA (SAGA) for the Sensor Selection Problem.

Strategy
--------
  Phase 1 – GA exploration  : run GA for a fixed number of generations
                               to find a good region of the search space.
  Phase 2 – SA exploitation : use the best GA solution as the SA starting
                               point and run SA for fine-grained local search.

This two-phase approach combines the global exploration strength of GA
with the local refinement capability of SA.
"""

import numpy as np
import time
from typing import List
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm
from sa import SimulatedAnnealing


class HybridSAGA:
    def __init__(
        self,
        evaluator: FitnessEvaluator,
        # GA phase
        ga_pop_size: int = 80,
        ga_generations: int = 150,
        # SA phase
        sa_T_init: float = 3.0,
        sa_T_min: float = 1e-4,
        sa_cooling: float = 0.997,
        sa_max_iter: int = 30_000,
        seed: int = 3,
    ):
        self.ev = evaluator
        self.seed = seed

        self.ga = GeneticAlgorithm(
            evaluator=evaluator,
            pop_size=ga_pop_size,
            n_generations=ga_generations,
            seed=seed,
        )
        self.sa = SimulatedAnnealing(
            evaluator=evaluator,
            T_init=sa_T_init,
            T_min=sa_T_min,
            cooling_rate=sa_cooling,
            max_iter=sa_max_iter,
            seed=seed + 10,
        )

        self.best_solution: np.ndarray = None
        self.best_fitness: float = np.inf
        self.best_info: dict = {}
        self.history_ga: List[dict] = []
        self.history_sa: List[dict] = []

    def run(self, verbose: bool = True) -> dict:
        t0 = time.time()

        # ---- Phase 1: GA ------------------------------------------------
        if verbose:
            print("  [SAGA] Phase 1: GA exploration ...")
        ga_result = self.ga.run(verbose=verbose)
        self.history_ga = ga_result["history"]

        # ---- Phase 2: SA starting from GA best --------------------------
        if verbose:
            print("  [SAGA] Phase 2: SA local search from GA best ...")
        # inject GA best as SA starting point
        self.sa.best_solution = ga_result["best_solution"].copy()
        self.sa.best_fitness  = ga_result["best_fitness"]
        self.sa.best_info     = ga_result["best_info"]

        # override SA's internal starting solution
        sa_result = self._run_sa_from(ga_result["best_solution"], verbose=verbose)
        self.history_sa = sa_result["history"]

        # pick overall best
        if sa_result["best_fitness"] < ga_result["best_fitness"]:
            self.best_solution = sa_result["best_solution"]
            self.best_fitness  = sa_result["best_fitness"]
            self.best_info     = sa_result["best_info"]
        else:
            self.best_solution = ga_result["best_solution"]
            self.best_fitness  = ga_result["best_fitness"]
            self.best_info     = ga_result["best_info"]

        elapsed = time.time() - t0
        if verbose:
            print(f"  SAGA finished in {elapsed:.1f}s | best fitness={self.best_fitness:.4f}")

        return {
            "algorithm":     "SAGA",
            "best_solution": self.best_solution,
            "best_fitness":  self.best_fitness,
            "best_info":     self.best_info,
            "history_ga":    self.history_ga,
            "history_sa":    self.history_sa,
            "elapsed":       elapsed,
        }

    # ------------------------------------------------------------------
    # Run SA from a given starting solution
    # ------------------------------------------------------------------
    def _run_sa_from(self, x0: np.ndarray, verbose: bool) -> dict:
        import math

        sa = self.sa
        ev = self.ev
        rng = sa.rng

        x = x0.copy()
        current_fit, current_info = ev.evaluate(x)
        best_x    = x.copy()
        best_fit  = current_fit
        best_info = current_info

        T = sa.T_init
        no_improve = 0
        n_reheats  = 0
        log_every  = max(1, sa.max_iter // 300)
        history    = []

        for it in range(sa.max_iter):
            x_new = sa._neighbour(x)
            x_new = ev.repair(x_new)
            new_fit, new_info = ev.evaluate(x_new)

            delta = new_fit - current_fit
            if delta < 0 or rng.random() < math.exp(-delta / (T + 1e-12)):
                x = x_new
                current_fit  = new_fit
                current_info = new_info

            if current_fit < best_fit:
                best_fit  = current_fit
                best_x    = x.copy()
                best_info = current_info
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= sa.reheat_patience and T < sa.T_init * 0.1:
                T = min(T * sa.reheat_factor, sa.T_init * 0.5)
                no_improve = 0
                n_reheats += 1

            T = max(T * sa.alpha, sa.T_min)

            if it % log_every == 0:
                history.append({
                    "iteration":    it,
                    "best_fitness": best_fit,
                    "curr_fitness": current_fit,
                    "temperature":  T,
                    "f1_coverage":  best_info["f1_coverage"],
                    "f2_cost":      best_info["f2_cost"],
                    "n_selected":   best_info["n_selected"],
                    "feasible":     best_info["feasible"],
                    "n_reheats":    n_reheats,
                })

            if verbose and it % 10000 == 0:
                print(
                    f"    SA-phase iter {it:6d} | T={T:.5f} | best={best_fit:.4f} "
                    f"| cov={best_info['f1_coverage']:.3f} | reheats={n_reheats}"
                )

        return {
            "best_solution": best_x,
            "best_fitness":  best_fit,
            "best_info":     best_info,
            "history":       history,
        }
