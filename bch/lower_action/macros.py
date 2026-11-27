"""Translate the eleven historical macros into ordered SC2-independent commands."""

from dataclasses import dataclass
from math import dist
from typing import Tuple, Union

from .grouping import center


@dataclass(frozen=True)
class Command:
    ability: str
    units: Tuple[int, ...]
    target: Union[int, Tuple[float, float]]
    queued: bool = False


# Unit types that can attack air units (Marine=1, Stalker=4, Hydralisk=8).
_ANTI_AIR_TYPES = {1, 4, 8}


def _order(ability, units, target, queued=False):
    return Command(ability, tuple(unit.tag for unit in units), target, queued)


def _split_support(units):
    combat = tuple(unit for unit in units if not unit.is_medivac)
    medivacs = tuple(unit for unit in units if unit.is_medivac)
    return combat, medivacs


def _valid_enemies(unit, enemies):
    if unit.unit_type in _ANTI_AIR_TYPES:
        return enemies
    ground = tuple(enemy for enemy in enemies if not enemy.is_medivac)
    return ground


def _medivac_support(medivacs, combat, *, retreat_point=None):
    if not medivacs:
        return []
    if retreat_point is not None:
        return [_order("move", medivacs, retreat_point)]
    if not combat:
        return []
    wounded = tuple(unit for unit in combat if unit.health < unit.health_max)
    commands = []
    anchor = center(combat)
    for medivac in medivacs:
        if wounded and medivac.energy >= 2.0:
            patient = min(wounded, key=lambda unit: (unit.health / unit.health_max, dist(medivac.xy, unit.xy), unit.tag))
            commands.append(_order("heal", (medivac,), patient.tag))
        else:
            commands.append(_order("move", (medivac,), anchor))
    return commands


def _nearest(position, units, *, prefer_health=False):
    return min(units, key=lambda unit: (dist(position, unit.xy), unit.effective_health if prefer_health else 0, unit.tag))


def _in_reach(unit, enemy):
    if unit.is_medivac:
        return False
    threshold = max(unit.attack_range, 1.0) + (unit.radius + enemy.radius if unit.attack_range < 2.0 else 0.0)
    return dist(unit.xy, enemy.xy) < threshold


def _group_target(units, enemies, weak=False):
    def rank(enemy):
        distances = [dist(unit.xy, enemy.xy) for unit in units]
        return (
            -sum(_in_reach(unit, enemy) for unit in units),
            enemy.effective_health if weak else sum(distances),
            enemy.tag,
        )
    return min(enemies, key=rank)


def _attack_unit_or_move(unit, target_pool, fallback_point, *, prefer_health=False, queued=False):
    if target_pool:
        target = _nearest(unit.xy, target_pool, prefer_health=prefer_health)
        return _order("smart", (unit,), target.tag, queued=queued)
    return _order("move", (unit,), fallback_point, queued=queued)


def _nearest_per_unit(friendly, enemy, groups):
    combat, medivacs = _split_support(friendly)
    enemy_center = center(enemy)
    commands = [
        _attack_unit_or_move(unit, _valid_enemies(unit, enemy), enemy_center)
        for unit in combat
    ]
    return commands + _medivac_support(medivacs, combat)


def _nearest_per_group(friendly, enemy, groups):
    return _group_attacks(friendly, groups, enemy)


def _nearest_weakest(friendly, enemy, groups):
    combat, medivacs = _split_support(friendly)
    if not combat:
        return _medivac_support(medivacs, combat)
    ground_enemies = tuple(unit for unit in enemy if not unit.is_medivac)
    origin = center(combat)
    enemy_center = center(enemy)
    commands = []
    if ground_enemies:
        target = _nearest(origin, ground_enemies, prefer_health=True)
        commands.append(_order("smart", combat, target.tag))
    else:
        anti_air = tuple(unit for unit in combat if unit.unit_type in _ANTI_AIR_TYPES)
        ground_only = tuple(unit for unit in combat if unit.unit_type not in _ANTI_AIR_TYPES)
        if anti_air:
            target = _nearest(origin, enemy, prefer_health=True)
            commands.append(_order("smart", anti_air, target.tag))
        if ground_only:
            commands.append(_order("move", ground_only, enemy_center))
    return commands + _medivac_support(medivacs, combat)


def _weakest_per_group(friendly, enemy, groups):
    return _group_attacks(friendly, groups, enemy, weak=True)


def _group_attacks(friendly, groups, enemy, weak=False):
    combat, medivacs = _split_support(friendly)
    ground_enemies = tuple(unit for unit in enemy if not unit.is_medivac)
    enemy_center = center(enemy)
    commands = []
    for group in groups:
        g_combat = tuple(unit for unit in group.units if not unit.is_medivac)
        if not g_combat:
            continue
        if ground_enemies:
            tag = _group_target(g_combat, ground_enemies, weak).tag
            commands.extend(_order("smart", (unit,), tag) for unit in g_combat)
        else:
            anti_air = tuple(unit for unit in g_combat if unit.unit_type in _ANTI_AIR_TYPES)
            ground_only = tuple(unit for unit in g_combat if unit.unit_type not in _ANTI_AIR_TYPES)
            if anti_air:
                tag = _group_target(anti_air, enemy, weak).tag
                commands.extend(_order("smart", (unit,), tag) for unit in anti_air)
            if ground_only:
                commands.extend(_order("move", (unit,), enemy_center) for unit in ground_only)
    return commands + _medivac_support(medivacs, combat)


def _threatening(friendly, enemy, groups):
    combat, medivacs = _split_support(friendly)
    if not combat:
        return _medivac_support(medivacs, combat)
    ground_enemies = tuple(unit for unit in enemy if not unit.is_medivac)
    origin = center(combat)
    enemy_center = center(enemy)
    commands = []
    if ground_enemies:
        target = min(ground_enemies, key=lambda unit: (-unit.effective_health, dist(origin, unit.xy), unit.tag))
        commands.append(_order("smart", combat, target.tag))
    else:
        anti_air = tuple(unit for unit in combat if unit.unit_type in _ANTI_AIR_TYPES)
        ground_only = tuple(unit for unit in combat if unit.unit_type not in _ANTI_AIR_TYPES)
        if anti_air:
            target = min(enemy, key=lambda unit: (-unit.effective_health, dist(origin, unit.xy), unit.tag))
            commands.append(_order("smart", anti_air, target.tag))
        if ground_only:
            commands.append(_order("move", ground_only, enemy_center))
    return commands + _medivac_support(medivacs, combat)


def _withdraw(friendly, enemy, groups):
    commands = []
    for group in groups:
        threats = {_nearest(unit.xy, enemy).tag for unit in group.units}
        threat = center(tuple(unit for unit in enemy if unit.tag in threats))
        point = tuple(x + 3 * (x - y) for x, y in zip(group.center, threat))
        commands.extend(_order("move" if unit.is_medivac else "smart", (unit,), point) for unit in group.units)
    return commands


def _gather(friendly, enemy, groups):
    commands = []
    for group in groups:
        if group.crowding < 0.4:
            g_combat, g_meds = _split_support(group.units)
            if g_combat:
                commands.append(_order("smart", g_combat, group.center))
            if g_meds:
                commands.append(_order("move", g_meds, group.center))
    return commands + _nearest_per_unit(friendly, enemy, groups)


def _lure(friendly, enemy, groups):
    combat, medivacs = _split_support(friendly)
    if not combat:
        return _medivac_support(medivacs, combat)
    target = center(enemy)
    separated = tuple(
        unit for group in groups if group.uniformity < 0.4 for unit in group.units if not unit.is_medivac
    )
    lure = _nearest(target, separated if len(separated) > 1 else combat)
    rest = tuple(unit for unit in combat if unit.tag != lure.tag)
    commands = [_order("smart", (lure,), target)]
    own = center(combat)
    commands.extend(_order("smart", (unit,), own) for unit in rest)
    return commands + _medivac_support(medivacs, combat)


def _sacrifice_lure(friendly, enemy, groups):
    combat, medivacs = _split_support(friendly)
    if not combat:
        return _medivac_support(medivacs, combat)
    own, opposing = center(combat), center(enemy)
    lure = _nearest(opposing, combat, prefer_health=True)
    ground_enemies = tuple(unit for unit in enemy if not unit.is_medivac)
    pool = ground_enemies if ground_enemies else enemy
    target = _nearest(own, pool, prefer_health=True).tag
    commands = [
        _order("smart", (lure,), target)
        if (ground_enemies or lure.unit_type in _ANTI_AIR_TYPES)
        else _order("move", (lure,), opposing)
    ]
    rest = tuple(unit for unit in combat if unit.tag != lure.tag)
    if rest:
        point = tuple((6 * x - y) / 5 for x, y in zip(own, opposing))
        if ground_enemies:
            commands += [_order("smart", rest, target, queued=True), _order("move", rest, point)]
        else:
            anti_air = tuple(unit for unit in rest if unit.unit_type in _ANTI_AIR_TYPES)
            if anti_air:
                commands.append(_order("smart", anti_air, target, queued=True))
            commands.append(_order("move", rest, point))
    return commands + _medivac_support(medivacs, combat)


def _attack_point(friendly, enemy, groups):
    # In the source, do_randomly is a point attack at feature coordinate (50, 50).
    combat, medivacs = _split_support(friendly)
    commands = [_order("attack", combat, (50.0, 50.0))] if combat else []
    return commands + _medivac_support(medivacs, combat)


def _idle(friendly, enemy, groups):
    return []


_MACROS = {
    "action_ATK_nearest": _nearest_per_unit,
    "action_ATK_clu_nearest": _nearest_per_group,
    "action_ATK_nearest_weakest": _nearest_weakest,
    "action_ATK_clu_nearest_weakest": _weakest_per_group,
    "action_ATK_threatening": _threatening,
    "action_DEF_clu_nearest": _withdraw,
    "action_MIX_gather": _gather,
    "action_MIX_lure": _lure,
    "action_MIX_sacrifice_lure": _sacrifice_lure,
    "do_randomly": _attack_point,
    "do_nothing": _idle,
}
MACROS = tuple(_MACROS)


def macro_commands(observation, formation, name):
    if not observation.friendly or not observation.enemy:
        return []
    return _MACROS[name](observation.friendly, observation.enemy, formation.groups)
