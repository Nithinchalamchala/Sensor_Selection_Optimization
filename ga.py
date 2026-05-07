"""
ga.py
=====
Enhanced Genetic Algorithm for the Sensor Selection Problem.

Features
--------
  - Binary chromosome (length = N sensors)
  - Tournament selection
  - Adaptive mutation rate (starts high, decays over generations)
  - Uniform crossover + two-point crossover (randomly chosen)
  - Elitism (top-k individuals survive unchanged)
  - Repair operator for budget-infeasible solutions
  - Convergence history tracking
"""

import numpy as np
import time
from typing import List, Tuple
from fitness import FitnessEvaluator


class GeneticAlgorithm:
    def __init__(
        self,
        evaluator: FitnessEvaluator,
        pop_size: int = 100,
        n_generations: int = 300,
        crossover_rate: float = 0.85,
        mutation_rate_init: float = 0.05,
        mutation_rate_final: float = 0.01,
        tournament_size: int = 5,
        elite_frac: float = 0.05,
        seed: int = 0,
    ):
        self.ev = evaluator
        self.N = len(evaluator.env.sensors)
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.cx_rate = crossover_rate
        self.mut_init = mutation_rate_init
        self.mut_final = mutation_rate_final
        self.tourn_size = tournament_size
        self.n_elite = max(1, int(elite_frac * pop_size))
        self.rng = np.random.default_rng(seed)

        # results
        self.best_solution: np.ndarray = None
        self.best_fitness: float = np.inf
        self.best_info: dict = {}
        self.history: List[dict] = []   # per-generation stats

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_population(self) -> np.ndarray:
        """Random binary population; each individual has ~30% sensors active."""
        pop = (self.rng.random((self.pop_size, self.N)) < 0.30).astype(int)
        # repair all
        pop = np.array([self.ev.repair(ind) for ind in pop])
        return pop

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _tournament_select(
        self, pop: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        idxs = self.rng.integers(0, self.pop_size, size=self.tourn_size)
        best = idxs[np.argmin(fitnesses[idxs])]
        return pop[best].copy()

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.cx_rate:
            return p1.copy(), p2.copy()

        if self.rng.random() < 0.5:
            # Uniform crossover
            mask = self.rng.random(self.N) < 0.5
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
        else:
            # Two-point crossover
            pts = sorted(self.rng.integers(0, self.N, size=2))
            c1 = np.concatenate([p1[:pts[0]], p2[pts[0]:pts[1]], p1[pts[1]:]])
            c2 = np.concatenate([p2[:pts[0]], p1[pts[0]:pts[1]], p2[pts[1]:]])
        return c1, c2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def _mutate(self, ind: np.ndarray, mut_rate: float) -> np.ndarray:
        mask = self.rng.random(self.N) < mut_rate
        ind = ind.copy()
        ind[mask] = 1 - ind[mask]
        return ind

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, verbose: bool = True) -> dict:
        t0 = time.time()
        pop = self._init_population()
        fitnesses, infos = self.ev.evaluate_population(pop, apply_repair=True)

        for gen in range(self.n_gen):
            # adaptive mutation rate
            progress = gen / self.n_gen
            mut_rate = self.mut_init + (self.mut_final - self.mut_init) * progress

            # elitism: keep top individuals
            elite_idxs = np.argsort(fitnesses)[:self.n_elite]
            elites = pop[elite_idxs].copy()

            # build new population
            new_pop = []
            while len(new_pop) < self.pop_size - self.n_elite:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1, mut_rate)
                c2 = self._mutate(c2, mut_rate)
                c1 = self.ev.repair(c1)
                c2 = self.ev.repair(c2)
                new_pop.extend([c1, c2])

            new_pop = np.array(new_pop[:self.pop_size - self.n_elite])
            pop = np.vstack([elites, new_pop])
            fitnesses, infos = self.ev.evaluate_population(pop, apply_repair=False)

            # track best
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = pop[best_idx].copy()
                self.best_info = infos[best_idx]

            self.history.append({
                "generation":   gen,
                "best_fitness": self.best_fitness,
                "mean_fitness": float(np.mean(fitnesses)),
                "f1_coverage":  self.best_info["f1_coverage"],
                "f2_cost":      self.best_info["f2_cost"],
                "n_selected":   self.best_info["n_selected"],
                "feasible":     self.best_info["feasible"],
                "mut_rate":     mut_rate,
            })

            if verbose and gen % 50 == 0:
                print(
                    f"  GA Gen {gen:4d} | best={self.best_fitness:.4f} "
                    f"| cov={self.best_info['f1_coverage']:.3f} "
                    f"| cost={self.best_info['cost_raw']:.1f} "
                    f"| n={self.best_info['n_selected']} "
                    f"| feasible={self.best_info['feasible']}"
                )

        elapsed = time.time() - t0
        if verbose:
            print(f"  GA finished in {elapsed:.1f}s | best fitness={self.best_fitness:.4f}")

        return {
            "algorithm":     "GA",
            "best_solution": self.best_solution,
            "best_fitness":  self.best_fitness,
            "best_info":     self.best_info,
            "history":       self.history,
            "elapsed":       elapsed,
        }
