"""
environment.py
==============
Defines the smart building environment:
  - 2D grid with multiple priority-weighted zones
  - Obstacles / walls that block line-of-sight
  - Heterogeneous sensor types (motion, thermal, acoustic)
  - Random candidate sensor placement
  - Coverage computation with LOS checking
  - Communication-graph connectivity check
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set

# ---------------------------------------------------------------------------
# Sensor type definitions
# ---------------------------------------------------------------------------
SENSOR_TYPES = {
    "motion":   {"radius": 6,  "cost": 10, "color": "#e74c3c"},
    "thermal":  {"radius": 8,  "cost": 18, "color": "#e67e22"},
    "acoustic": {"radius": 10, "cost": 25, "color": "#9b59b6"},
}

COMM_RANGE = 15          # max distance for two sensors to communicate
GRID_W, GRID_H = 50, 50  # building grid dimensions


# ---------------------------------------------------------------------------
# Zone definition
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    priority: float          # higher = more important to cover
    critical: bool = False   # must be covered by >=2 sensors


# ---------------------------------------------------------------------------
# Sensor candidate
# ---------------------------------------------------------------------------
@dataclass
class Sensor:
    idx: int
    x: float
    y: float
    stype: str
    radius: float = field(init=False)
    cost: float = field(init=False)

    def __post_init__(self):
        self.radius = SENSOR_TYPES[self.stype]["radius"]
        self.cost   = SENSOR_TYPES[self.stype]["cost"]
        # small per-sensor cost variation ±20 %
        self.cost   = round(self.cost * random.uniform(0.8, 1.2), 2)
        self.radius = self.radius * random.uniform(0.85, 1.15)


# ---------------------------------------------------------------------------
# Building environment
# ---------------------------------------------------------------------------
class BuildingEnvironment:
    """
    Represents a 50×50 smart-building floor plan with:
      - 8 named zones of varying priority
      - Rectangular obstacle walls
      - N randomly placed heterogeneous candidate sensors
    """

    def __init__(self, n_sensors: int = 150, seed: int = 42):
        self.W = GRID_W
        self.H = GRID_H
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # ---- zones --------------------------------------------------------
        self.zones: List[Zone] = [
            Zone("Lobby",       0,  0, 15, 20, priority=1.0, critical=False),
            Zone("Corridor_1", 15,  0, 25, 50, priority=1.5, critical=False),
            Zone("Office_A",    0, 20, 15, 50, priority=2.0, critical=True),
            Zone("Office_B",   25,  0, 40, 25, priority=2.0, critical=True),
            Zone("Lab",        25, 25, 50, 50, priority=3.0, critical=True),
            Zone("Server_Room",40,  0, 50, 25, priority=3.5, critical=True),
            Zone("Corridor_2",  0,  0, 50,  5, priority=1.0, critical=False),
            Zone("Meeting_Rm", 40, 25, 50, 50, priority=2.5, critical=True),
        ]

        # ---- obstacles (list of (x1,y1,x2,y2) rectangles) ----------------
        self.obstacles: List[Tuple[int,int,int,int]] = [
            (15,  0, 17, 30),   # vertical wall between lobby & corridor
            (25,  5, 27, 50),   # wall separating corridor from offices
            (40,  0, 42, 50),   # wall around server room / meeting room
            (10, 20, 15, 22),   # small partition in office A
            (30, 25, 35, 27),   # partition in lab
        ]

        # ---- pre-compute obstacle mask ------------------------------------
        self.obstacle_mask = self._build_obstacle_mask()

        # ---- zone priority map -------------------------------------------
        self.priority_map = self._build_priority_map()

        # ---- candidate sensors -------------------------------------------
        self.n_sensors = n_sensors
        self.sensors: List[Sensor] = self._place_sensors()

        # ---- pre-compute coverage sets -----------------------------------
        # coverage_cells[i] = set of (x,y) grid cells covered by sensor i
        self.coverage_cells: List[Set[Tuple[int,int]]] = self._precompute_coverage()

        # total weighted cells (denominator for f1)
        self.total_weighted = float(np.sum(self.priority_map))

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _build_obstacle_mask(self) -> np.ndarray:
        mask = np.zeros((self.W, self.H), dtype=bool)
        for (x1, y1, x2, y2) in self.obstacles:
            mask[x1:x2, y1:y2] = True
        return mask

    def _build_priority_map(self) -> np.ndarray:
        pmap = np.ones((self.W, self.H), dtype=float) * 0.5  # default low
        for z in self.zones:
            pmap[z.x1:z.x2, z.y1:z.y2] = z.priority
        # zero out obstacles
        pmap[self.obstacle_mask] = 0.0
        return pmap

    def _place_sensors(self) -> List[Sensor]:
        sensors = []
        stypes = list(SENSOR_TYPES.keys())
        attempts = 0
        while len(sensors) < self.n_sensors and attempts < self.n_sensors * 20:
            attempts += 1
            x = random.uniform(0, self.W - 1)
            y = random.uniform(0, self.H - 1)
            ix, iy = int(x), int(y)
            if self.obstacle_mask[ix, iy]:
                continue
            stype = random.choice(stypes)
            sensors.append(Sensor(idx=len(sensors), x=x, y=y, stype=stype))
        return sensors

    def _has_los(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Bresenham line-of-sight check between two float positions."""
        ix0, iy0 = int(round(x0)), int(round(y0))
        ix1, iy1 = int(round(x1)), int(round(y1))
        dx = abs(ix1 - ix0); dy = abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx - dy
        cx, cy = ix0, iy0
        while True:
            if 0 <= cx < self.W and 0 <= cy < self.H:
                if self.obstacle_mask[cx, cy]:
                    return False
            if cx == ix1 and cy == iy1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; cx += sx
            if e2 < dx:
                err += dx; cy += sy
        return True

    def _precompute_coverage(self) -> List[Set[Tuple[int,int]]]:
        coverage = []
        for s in self.sensors:
            cells: Set[Tuple[int,int]] = set()
            r = s.radius
            for cx in range(max(0, int(s.x - r) - 1),
                            min(self.W, int(s.x + r) + 2)):
                for cy in range(max(0, int(s.y - r) - 1),
                                min(self.H, int(s.y + r) + 2)):
                    if self.obstacle_mask[cx, cy]:
                        continue
                    dist = np.hypot(cx - s.x, cy - s.y)
                    if dist <= r and self._has_los(s.x, s.y, cx, cy):
                        cells.add((cx, cy))
            coverage.append(cells)
        return coverage

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_zone_cells(self, zone: Zone) -> Set[Tuple[int,int]]:
        cells = set()
        for x in range(zone.x1, zone.x2):
            for y in range(zone.y1, zone.y2):
                if not self.obstacle_mask[x, y]:
                    cells.add((x, y))
        return cells

    def budget_default(self) -> float:
        """~50 % of total sensor cost as default budget."""
        return sum(s.cost for s in self.sensors) * 0.50

    def summary(self):
        print(f"Building : {self.W}×{self.H} grid")
        print(f"Zones    : {len(self.zones)}")
        print(f"Obstacles: {len(self.obstacles)} rectangles")
        print(f"Sensors  : {self.n_sensors} candidates")
        type_counts = {}
        for s in self.sensors:
            type_counts[s.stype] = type_counts.get(s.stype, 0) + 1
        for t, c in type_counts.items():
            print(f"  {t:10s}: {c}")
        print(f"Budget   : {self.budget_default():.1f}")
