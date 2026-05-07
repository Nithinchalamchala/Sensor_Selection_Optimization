"""
fitness.py
==========
Multi-objective fitness evaluation for the Sensor Selection Problem.

Objectives
----------
  f1 : Maximize priority-weighted coverage  (normalised to [0,1])
  f2 : Minimize total deployment cost       (normalised to [0,1])
  f3 : Minimize coverage overlap ratio      (normalised to [0,1])

Combined scalar fitness (to minimise):
  F(x) = -alpha*f1 + beta*f2 + gamma*f3

Constraints
-----------
  C1 : sum(cost_i * x_i) <= Budget
  C2 : every critical zone covered by >= 2 sensors
  C3 : selected sensors form a connected communication graph

Infeasible solutions are penalised via an adaptive penalty term.
"""

import numpy as np
from typing import List, Tuple, Set
from environment import BuildingEnvironment, COMM_RANGE


# ---------------------------------------------------------------------------
# Default objective weights
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.60   # coverage importance
DEFAULT_BETA  = 0.25   # cost importance
DEFAULT_GAMMA = 0.15   # overlap penalty


# ---------------------------------------------------------------------------
# Core fitness function
# ---------------------------------------------------------------------------
class FitnessEvaluator:
    def __init__(
        self,
        env: BuildingEnvironment,
        budget: float = None,
        alpha: float = DEFAULT_ALPHA,
        beta:  float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        penalty_weight: float = 1e4,
    ):
        self.env = env
        self.budget = budget if budget is not None else env.budget_default()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.penalty_weight = penalty_weight

        # pre-compute normalisation denominators
        self.max_cost = sum(s.cost for s in env.sensors)
        self.total_weighted = env.total_weighted

        # pre-compute critical zone cell sets
        self.critical_zones = [
            (z, env.get_zone_cells(z))
            for z in env.zones if z.critical
        ]

    # ------------------------------------------------------------------
    # Objective components
    # ------------------------------------------------------------------
    def compute_coverage_and_overlap(
        self, selected: List[int]
    ) -> Tuple[float, float, float]:
        """
        Returns
        -------
        weighted_coverage : sum of priority_map values over uniquely covered cells
        overlap_ratio     : fraction of total covered cell-visits that are overlaps
        n_covered_cells   : raw count of unique covered cells
        """
        if not selected:
            return 0.0, 0.0, 0

        cell_count: dict = {}
        for idx in selected:
            for cell in self.env.coverage_cells[idx]:
                cell_count[cell] = cell_count.get(cell, 0) + 1

        weighted_cov = sum(
            self.env.priority_map[c[0], c[1]] for c in cell_count
        )
        total_visits = sum(cell_count.values())
        overlap_visits = sum(v - 1 for v in cell_count.values() if v > 1)
        overlap_ratio = overlap_visits / total_visits if total_visits > 0 else 0.0

        return weighted_cov, overlap_ratio, len(cell_count)

    def compute_cost(self, selected: List[int]) -> float:
        return sum(self.env.sensors[i].cost for i in selected)

    def check_redundancy(self, selected: List[int]) -> float:
        """
        Returns penalty proportional to under-covered critical zones.
        0 if all critical zones have >=2 sensors covering them.
        """
        if not selected:
            return float(len(self.critical_zones))

        penalty = 0.0
        for zone, zone_cells in self.critical_zones:
            covering = 0
            for idx in selected:
                if self.env.coverage_cells[idx] & zone_cells:
                    covering += 1
            if covering < 2:
                penalty += (2 - covering)
        return penalty

    def check_connectivity(self, selected: List[int]) -> float:
        """
        Returns 0 if selected sensors form a connected graph (or <=1 sensor),
        else returns the number of disconnected components - 1.
        """
        if len(selected) <= 1:
            return 0.0
        sensors = self.env.sensors
        # build adjacency
        adj = {i: [] for i in selected}
        sel_set = set(selected)
        for i in selected:
            for j in selected:
                if j <= i:
                    continue
                dist = np.hypot(sensors[i].x - sensors[j].x,
                                sensors[i].y - sensors[j].y)
                if dist <= COMM_RANGE:
                    adj[i].append(j)
                    adj[j].append(i)
        # BFS
        visited = set()
        queue = [selected[0]]
        visited.add(selected[0])
        while queue:
            node = queue.pop()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        disconnected = len(sel_set - visited)
        return float(disconnected)

    # ------------------------------------------------------------------
    # Repair operator
    # ------------------------------------------------------------------
    def repair(self, x: np.ndarray) -> np.ndarray:
        """
        Greedy repair: if budget is violated, iteratively remove the sensor
        with the worst cost-to-coverage ratio until feasible.
        """
        x = x.copy()
        selected = list(np.where(x == 1)[0])
        cost = self.compute_cost(selected)
        if cost <= self.budget:
            return x

        # rank by cost/coverage_size descending (worst value first)
        def value_ratio(idx):
            cov = len(self.env.coverage_cells[idx])
            return self.env.sensors[idx].cost / (cov + 1e-9)

        selected.sort(key=value_ratio, reverse=True)
        for idx in selected:
            if cost <= self.budget:
                break
            x[idx] = 0
            cost -= self.env.sensors[idx].cost
        return x

    # ------------------------------------------------------------------
    # Main fitness
    # ------------------------------------------------------------------
    def evaluate(
        self,
        x: np.ndarray,
        apply_repair: bool = False,
        penalty_weight: float = None,
    ) -> Tuple[float, dict]:
        """
        Parameters
        ----------
        x : binary array of length N
        apply_repair : if True, repair budget violations before scoring
        penalty_weight : override instance penalty weight

        Returns
        -------
        fitness : scalar (lower is better)
        info    : dict with individual components
        """
        pw = penalty_weight if penalty_weight is not None else self.penalty_weight

        if apply_repair:
            x = self.repair(x)

        selected = list(np.where(x == 1)[0])

        # --- objectives ---
        weighted_cov, overlap_ratio, n_cells = self.compute_coverage_and_overlap(selected)
        cost = self.compute_cost(selected)

        f1 = weighted_cov / (self.total_weighted + 1e-9)   # [0,1] higher=better
        f2 = cost / (self.max_cost + 1e-9)                 # [0,1] lower=better
        f3 = overlap_ratio                                  # [0,1] lower=better

        # --- constraint violations ---
        budget_viol    = max(0.0, cost - self.budget) / (self.budget + 1e-9)
        redundancy_pen = self.check_redundancy(selected)
        connect_pen    = self.check_connectivity(selected)

        total_penalty = pw * (budget_viol + redundancy_pen * 0.1 + connect_pen * 0.05)

        # --- combined fitness (minimise) ---
        fitness = (
            -self.alpha * f1
            + self.beta  * f2
            + self.gamma * f3
            + total_penalty
        )

        info = {
            "fitness":        fitness,
            "f1_coverage":    f1,
            "f2_cost":        f2,
            "f3_overlap":     f3,
            "cost_raw":       cost,
            "n_selected":     len(selected),
            "n_cells":        n_cells,
            "budget_viol":    budget_viol,
            "redundancy_pen": redundancy_pen,
            "connect_pen":    connect_pen,
            "feasible":       (budget_viol == 0 and redundancy_pen == 0 and connect_pen == 0),
        }
        return fitness, info

    def evaluate_population(
        self, population: np.ndarray, apply_repair: bool = False
    ) -> Tuple[np.ndarray, List[dict]]:
        fitnesses = []
        infos = []
        for x in population:
            f, info = self.evaluate(x, apply_repair=apply_repair)
            fitnesses.append(f)
            infos.append(info)
        return np.array(fitnesses), infos
