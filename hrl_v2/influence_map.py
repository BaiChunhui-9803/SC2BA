"""Multi-channel tactical Influence Map builder with world-to-grid mapping.

Supports 4 tactical channels (and 1-channel signed ablation matching legacy im-grid):
  Ch 0: Signed unit presence field (7x7 decay kernel [16, 9, 4, 1], normalized by influence_scale)
  Ch 1: Effective HP+shield weighted signed field (distinguishes healthy vs wounded squads)
  Ch 2: Directional weapon-range fire coverage field (friendly reach minus enemy reach)
  Ch 3: Heterogeneous tactical role & support field (medivac / melee frontline / heavy / ranged)

Also returns the exact world->normalized-grid transform so IM-CNN spatial feature maps
can be bilinearly sampled at each internal tactical cluster centroid.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from bch.observation import Observation, Unit


# 7x7 square-ring influence kernel matching legacy HRL-IM/CBS [16, 9, 4, 1]
def _make_ring_kernel() -> np.ndarray:
    k = np.zeros((7, 7), dtype=np.float32)
    c = 3
    for y in range(7):
        for x in range(7):
            r = max(abs(x - c), abs(y - c))
            if r == 0:
                k[y, x] = 16.0
            elif r == 1:
                k[y, x] = 9.0
            elif r == 2:
                k[y, x] = 4.0
            elif r == 3:
                k[y, x] = 1.0
    return k


_RING_KERNEL = _make_ring_kernel()
_DISK_CACHE = {}


def _disk_offsets(radius_int: int) -> Tuple[np.ndarray, np.ndarray]:
    r = max(1, int(radius_int))
    if r not in _DISK_CACHE:
        coords = np.arange(-r, r + 1)
        dy, dx = np.meshgrid(coords, coords, indexing="ij")
        mask = (dx * dx + dy * dy) <= (r * r)
        _DISK_CACHE[r] = (dy[mask], dx[mask])
    return _DISK_CACHE[r]


def _unit_role_weight(unit: Unit) -> float:
    """Pure attribute-derived role weight (no scenario-specific hardcoding)."""
    if unit.is_medivac:
        return 1.5
    if unit.attack_range < 2.0:
        return 0.9  # melee frontline (Zealot / Zergling / Baneling)
    if unit.effective_health_max >= 120.0:
        return 0.6  # heavy / high-HP ranged (Marauder / Stalker / Colossus)
    return 0.3      # light ranged (Marine / Hydralisk)


@dataclass(frozen=True)
class InfluenceGrid:
    tensor: np.ndarray        # [C, H, W] float32
    center_x: float           # world x mapped to grid center
    center_y: float           # world y mapped to grid center
    cells_per_world: float    # scale factor (1.0 normally; <1.0 if engagement exceeds grid)
    grid_size: int

    def world_to_norm(self, x: float, y: float) -> Tuple[float, float]:
        """Map world (x, y) to PyTorch grid_sample normalized coords in [-1, 1] x [-1, 1].

        grid_sample expects (gx, gy) where gx=-1 is left column (col=0) and gy=-1 is top row (row=0).
        Note: row increases as world y decreases (top = higher y).
        """
        half = self.grid_size / 2.0
        col = half + (x - self.center_x) * self.cells_per_world
        row = half - (y - self.center_y) * self.cells_per_world
        gx = float(np.clip((col + 0.5) / half - 1.0, -1.0, 1.0))
        gy = float(np.clip((row + 0.5) / half - 1.0, -1.0, 1.0))
        return gx, gy


def _stamp_kernel(canvas: np.ndarray, row: int, col: int, kernel: np.ndarray, weight: float) -> None:
    h, w = canvas.shape
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    y0, y1 = row - ry, row + ry + 1
    x0, x1 = col - rx, col + rx + 1
    ky0, kx0 = max(0, -y0), max(0, -x0)
    cy0, cx0 = max(0, y0), max(0, x0)
    cy1, cx1 = min(h, y1), min(w, x1)
    if cy0 < cy1 and cx0 < cx1:
        canvas[cy0:cy1, cx0:cx1] += weight * kernel[ky0:ky0 + (cy1 - cy0), kx0:kx0 + (cx1 - cx0)]


def build_tactical_influence_map(
    obs: Observation,
    *,
    grid_size: int = 64,
    channels: int = 4,
) -> InfluenceGrid:
    """Build a centered multi-channel tactical influence grid from an Observation."""
    if channels not in (1, 4):
        raise ValueError(f"im_channels must be 1 or 4, got {channels}.")
    grid = np.zeros((channels, grid_size, grid_size), dtype=np.float32)
    all_units = obs.friendly + obs.enemy
    if not all_units:
        return InfluenceGrid(grid, 0.0, 0.0, 1.0, grid_size)

    xs = np.array([u.x for u in all_units], dtype=np.float32)
    ys = np.array([u.y for u in all_units], dtype=np.float32)
    center_x = float(0.5 * (xs.min() + xs.max()))
    center_y = float(0.5 * (ys.min() + ys.max()))
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    # Keep 1 cell = 1 world unit whenever engagement fits with margin; shrink smoothly if huge.
    usable = float(grid_size - 14)
    cells_per_world = 1.0 if span <= usable else usable / max(span, 1e-6)
    half = grid_size / 2.0
    scale = max(float(obs.influence_scale), 16.0)
    fire_norm = max(1.0, 0.5 * len(all_units))

    for sign, side in ((1.0, obs.friendly), (-1.0, obs.enemy)):
        for unit in side:
            col = int(np.clip(round(half + (unit.x - center_x) * cells_per_world), 0, grid_size - 1))
            row = int(np.clip(round(half - (unit.y - center_y) * cells_per_world), 0, grid_size - 1))

            # Ch 0: Signed position presence
            _stamp_kernel(grid[0], row, col, _RING_KERNEL, sign / scale)

            if channels == 4:
                # Ch 1: Effective HP+shield weighted presence
                hp_ratio = unit.effective_health / max(unit.effective_health_max, 1.0)
                _stamp_kernel(grid[1], row, col, _RING_KERNEL, (sign * hp_ratio) / scale)

                # Ch 2: Weapon-range fire coverage
                if not unit.is_medivac and unit.attack_range > 0.0:
                    rad = max(1, int(round(max(unit.attack_range, 1.0) * cells_per_world)))
                    dy, dx = _disk_offsets(rad)
                    rr = row + dy
                    cc = col + dx
                    valid = (rr >= 0) & (rr < grid_size) & (cc >= 0) & (cc < grid_size)
                    grid[2, rr[valid], cc[valid]] += sign / fire_norm

                # Ch 3: Tactical role / support field
                role_w = _unit_role_weight(unit)
                _stamp_kernel(grid[3], row, col, _RING_KERNEL, (sign * role_w) / scale)

    # Soft clip extreme fire-overlap values for CNN numerical stability
    if channels == 4:
        grid[2] = np.tanh(grid[2])
    return InfluenceGrid(grid, center_x, center_y, cells_per_world, grid_size)
