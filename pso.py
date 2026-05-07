"""
pso.py
======
Binary Particle Swarm Optimisation with Lévy-flight perturbation
for the Sensor Selection Problem.

Features
--------
  - Velocity → probability via sigmoid transfer function
  - Linearly decreasing inertia weight (w: 0.9 → 0.4)
  - Cognitive (c1) and social (c2) acceleration coefficients
  - Lévy-flight perturbation on global best to escape stagnation
  - Repair operator for budget-infeasible particles
  - Convergence history tracking
"""

import numpy as np
import time
import math as _math
from typing import List
from fitness import FitnessEvaluator


def _levy_flight(size: int, beta: float = 1.5, rng=None) -> np.ndarray:
    """Generate Lévy-distributed step sizes (Mantegna's algorithm)."""
    if rng is None:
        rng = np.random.default_rng()
    sigma_u = (
        _math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (_math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u, size)
    v = rng.normal(0, 1, size)
    step = u / (np.abs(v) ** (1 / beta))
    return step


class BinaryPSO:
    def __init__(
        self,
        evaluator: FitnessEvaluator,
        n_particles: int = 80,
        n_iterations: int = 300,
        w_init: float = 0.9,
        w_final: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        levy_prob: float = 0.15,
        stagnation_limit: int = 30,
        seed: int = 2,
    ):
        self.ev = evaluator
        self.N = len(evaluator.env.sensors)
        self.n_particles = n_particles
        self.n_iter = n_iterations
        self.w_init = w_init
        self.w_final = w_final
        self.c1 = c1
        self.c2 = c2
        self.levy_prob = levy_prob
        self.stagnation_limit = stagnation_limit
        self.rng = np.random.default_rng(seed)

        self.best_solution: np.ndarray = None
        self.best_fitness: float = np.inf
        self.best_info: dict = {}
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Sigmoid transfer
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(v: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))

    # ------------------------------------------------------------------
    # Lévy perturbation on global best
    # ------------------------------------------------------------------
    def _levy_perturb(self, gbest: np.ndarray) -> np.ndarray:
        step = _levy_flight(self.N, rng=self.rng)
        prob = self._sigmoid(step * 0.1)
        new = gbest.copy()
        flip = self.rng.random(self.N) < prob * self.levy_prob
        new[flip] = 1 - new[flip]
        return self.ev.repair(new)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, verbose: bool = True) -> dict:
        t0 = time.time()

        # initialise positions (binary) and velocities (continuous)
        pos = (self.rng.random((self.n_particles, self.N)) < 0.30).astype(int)
        pos = np.array([self.ev.repair(p) for p in pos])
        vel = self.rng.uniform(-4, 4, (self.n_particles, self.N))

        # personal bests
        pbest_pos = pos.copy()
        pbest_fit, pbest_info = self.ev.evaluate_population(pbest_pos)

        # global best
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        gbest_info = pbest_info[gbest_idx]

        self.best_solution = gbest_pos.copy()
        self.best_fitness  = gbest_fit
        self.best_info     = gbest_info

        stagnation = 0

        for it in range(self.n_iter):
            # linearly decreasing inertia
            w = self.w_init - (self.w_init - self.w_final) * it / self.n_iter

            r1 = self.rng.random((self.n_particles, self.N))
            r2 = self.rng.random((self.n_particles, self.N))

            # velocity update
            vel = (
                w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )

            # binary position update via sigmoid
            prob = self._sigmoid(vel)
            pos = (self.rng.random((self.n_particles, self.N)) < prob).astype(int)
            pos = np.array([self.ev.repair(p) for p in pos])

            # evaluate
            fits, infos = self.ev.evaluate_population(pos)

            # update personal bests
            improved = fits < pbest_fit
            pbest_pos[improved] = pos[improved].copy()
            pbest_fit[improved] = fits[improved]
            for i in np.where(improved)[0]:
                pbest_info[i] = infos[i]

            # update global best
            best_idx = np.argmin(pbest_fit)
            if pbest_fit[best_idx] < gbest_fit:
                gbest_fit  = pbest_fit[best_idx]
                gbest_pos  = pbest_pos[best_idx].copy()
                gbest_info = pbest_info[best_idx]
                stagnation = 0
            else:
                stagnation += 1

            # Lévy perturbation on stagnation
            if stagnation >= self.stagnation_limit:
                candidate = self._levy_perturb(gbest_pos)
                c_fit, c_info = self.ev.evaluate(candidate)
                if c_fit < gbest_fit:
                    gbest_fit  = c_fit
                    gbest_pos  = candidate.copy()
                    gbest_info = c_info
                stagnation = 0

            if gbest_fit < self.best_fitness:
                self.best_fitness  = gbest_fit
                self.best_solution = gbest_pos.copy()
                self.best_info     = gbest_info

            self.history.append({
                "iteration":    it,
                "best_fitness": self.best_fitness,
                "mean_fitness": float(np.mean(fits)),
                "f1_coverage":  self.best_info["f1_coverage"],
                "f2_cost":      self.best_info["f2_cost"],
                "n_selected":   self.best_info["n_selected"],
                "feasible":     self.best_info["feasible"],
                "inertia":      w,
                "stagnation":   stagnation,
            })

            if verbose and it % 50 == 0:
                print(
                    f"  PSO iter {it:4d} | best={self.best_fitness:.4f} "
                    f"| cov={self.best_info['f1_coverage']:.3f} "
                    f"| cost={self.best_info['cost_raw']:.1f} "
                    f"| n={self.best_info['n_selected']}"
                )

        elapsed = time.time() - t0
        if verbose:
            print(f"  PSO finished in {elapsed:.1f}s | best fitness={self.best_fitness:.4f}")

        return {
            "algorithm":     "PSO",
            "best_solution": self.best_solution,
            "best_fitness":  self.best_fitness,
            "best_info":     self.best_info,
            "history":       self.history,
            "elapsed":       elapsed,
        }
