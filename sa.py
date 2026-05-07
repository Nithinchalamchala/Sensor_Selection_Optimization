"""
sa.py
=====
Enhanced Simulated Annealing for the Sensor Selection Problem.

Features
--------
  - Geometric cooling schedule
  - Reheating mechanism to escape local optima
  - Two neighbourhood moves:
      * bit-flip  : toggle a single sensor
      * block-swap: flip a random cluster of k sensors
  - Adaptive penalty weight (increases when infeasible solutions dominate)
  - Repair operator applied after each accepted move
  - Convergence history tracking
"""

import numpy as np
import time
import math
from typing import List
from fitness import FitnessEvaluator


class SimulatedAnnealing:
    def __init__(
        self,
        evaluator: FitnessEvaluator,
        T_init: float = 5.0,
        T_min: float = 1e-4,
        cooling_rate: float = 0.995,
        reheat_factor: float = 2.0,
        reheat_patience: int = 500,
        max_iter: int = 50_000,
        block_size: int = 3,
        seed: int = 1,
    ):
        self.ev = evaluator
        self.N = len(evaluator.env.sensors)
        self.T_init = T_init
        self.T_min = T_min
        self.alpha = cooling_rate
        self.reheat_factor = reheat_factor
        self.reheat_patience = reheat_patience
        self.max_iter = max_iter
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

        self.best_solution: np.ndarray = None
        self.best_fitness: float = np.inf
        self.best_info: dict = {}
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Neighbourhood moves
    # ------------------------------------------------------------------
    def _bit_flip(self, x: np.ndarray) -> np.ndarray:
        nb = x.copy()
        idx = self.rng.integers(0, self.N)
        nb[idx] = 1 - nb[idx]
        return nb

    def _block_swap(self, x: np.ndarray) -> np.ndarray:
        nb = x.copy()
        idxs = self.rng.integers(0, self.N, size=self.block_size)
        nb[idxs] = 1 - nb[idxs]
        return nb

    def _neighbour(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.7:
            return self._bit_flip(x)
        return self._block_swap(x)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, verbose: bool = True) -> dict:
        t0 = time.time()

        # initialise random feasible solution
        x = (self.rng.random(self.N) < 0.30).astype(int)
        x = self.ev.repair(x)
        current_fit, current_info = self.ev.evaluate(x)

        self.best_solution = x.copy()
        self.best_fitness  = current_fit
        self.best_info     = current_info

        T = self.T_init
        no_improve = 0
        n_reheats  = 0
        log_every  = max(1, self.max_iter // 300)

        for it in range(self.max_iter):
            # generate neighbour
            x_new = self._neighbour(x)
            x_new = self.ev.repair(x_new)
            new_fit, new_info = self.ev.evaluate(x_new)

            delta = new_fit - current_fit
            if delta < 0 or self.rng.random() < math.exp(-delta / (T + 1e-12)):
                x = x_new
                current_fit  = new_fit
                current_info = new_info

            # update global best
            if current_fit < self.best_fitness:
                self.best_fitness  = current_fit
                self.best_solution = x.copy()
                self.best_info     = current_info
                no_improve = 0
            else:
                no_improve += 1

            # reheating
            if no_improve >= self.reheat_patience and T < self.T_init * 0.1:
                T = min(T * self.reheat_factor, self.T_init * 0.5)
                no_improve = 0
                n_reheats += 1

            # cooling
            T = max(T * self.alpha, self.T_min)

            # log
            if it % log_every == 0:
                self.history.append({
                    "iteration":    it,
                    "best_fitness": self.best_fitness,
                    "curr_fitness": current_fit,
                    "temperature":  T,
                    "f1_coverage":  self.best_info["f1_coverage"],
                    "f2_cost":      self.best_info["f2_cost"],
                    "n_selected":   self.best_info["n_selected"],
                    "feasible":     self.best_info["feasible"],
                    "n_reheats":    n_reheats,
                })

            if verbose and it % 10000 == 0:
                print(
                    f"  SA iter {it:6d} | T={T:.5f} | best={self.best_fitness:.4f} "
                    f"| cov={self.best_info['f1_coverage']:.3f} "
                    f"| cost={self.best_info['cost_raw']:.1f} "
                    f"| reheats={n_reheats}"
                )

        elapsed = time.time() - t0
        if verbose:
            print(f"  SA finished in {elapsed:.1f}s | best fitness={self.best_fitness:.4f}")

        return {
            "algorithm":     "SA",
            "best_solution": self.best_solution,
            "best_fitness":  self.best_fitness,
            "best_info":     self.best_info,
            "history":       self.history,
            "elapsed":       elapsed,
        }
