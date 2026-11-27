"""The original HRL-IM/CBS short-term reward for lower Q updates."""

import numpy as np
from scipy.spatial.distance import cdist


def _health_ratio(units):
    return sum(int(unit.effective_health / unit.effective_health_max * 255) for unit in units)


def _coverage(observation):
    if not observation.friendly or not observation.enemy:
        return 0, 0
    distance = cdist([unit.xy for unit in observation.friendly], [unit.xy for unit in observation.enemy])
    own_ranges = np.array([0.0 if unit.is_medivac else unit.attack_range for unit in observation.friendly])[:, None]
    enemy_ranges = np.array([0.0 if unit.is_medivac else unit.attack_range for unit in observation.enemy])[None, :]
    radii = np.array([unit.radius for unit in observation.friendly])[:, None] + np.array(
        [unit.radius for unit in observation.enemy]
    )[None, :]
    own_thresh = np.where(own_ranges < 2.0, own_ranges + radii, own_ranges)
    enemy_thresh = np.where(enemy_ranges < 2.0, enemy_ranges + radii, enemy_ranges)
    own_reaches = (own_ranges > 0.0) & (distance < own_thresh)
    enemy_reaches = (enemy_ranges > 0.0) & (distance < enemy_thresh)
    return int(own_reaches.sum(axis=0).max()), int(enemy_reaches.sum(axis=1).max())


def short_reward(before, after):
    own_before, enemy_before = _health_ratio(before.friendly), _health_ratio(before.enemy)
    own_after, enemy_after = _health_ratio(after.friendly), _health_ratio(after.enemy)
    before_gap, after_gap = own_before - enemy_before, own_after - enemy_after
    # The legacy tests compare the signed gaps, including a lead change.
    inferior = 10 if before_gap < 0 and after_gap > before_gap else 0
    dominant = -10 if before_gap > 0 and after_gap < before_gap else 0
    own_coverage, enemy_coverage = _coverage(before)
    next_own_coverage, next_enemy_coverage = _coverage(after)
    fire_coverage = 5 * max(0, next_own_coverage - own_coverage)
    if next_enemy_coverage < enemy_coverage:
        fire_coverage = 5 * (enemy_coverage - next_enemy_coverage)
    net_kills = (len(before.enemy) - len(after.enemy)) - (len(before.friendly) - len(after.friendly))
    return 5 * net_kills + inferior + dominant + (after_gap - before_gap) / 10 + fire_coverage
