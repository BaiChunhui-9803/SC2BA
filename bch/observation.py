"""Plain data shared by the environment, encoder and action library."""

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class Unit:
    tag: int
    x: float
    y: float
    health: float
    health_max: float
    radius: float
    cooldown: float
    shield: float = 0.0
    shield_max: float = 0.0
    unit_type: int = 1
    attack_range: float = 5.0
    is_medivac: bool = False
    energy: float = 0.0

    @property
    def xy(self):
        return self.x, self.y

    @property
    def effective_health(self):
        return self.health + self.shield

    @property
    def effective_health_max(self):
        return self.health_max + self.shield_max


@dataclass(frozen=True)
class Observation:
    friendly: Tuple[Unit, ...]
    enemy: Tuple[Unit, ...]
    step: int
    game_loop: int
    map_height: int
    influence_scale: float

    @property
    def margin(self):
        return (
            sum(unit.effective_health for unit in self.friendly)
            - sum(unit.effective_health for unit in self.enemy)
        )

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, row):
        row = dict(row)
        for side in ("friendly", "enemy"):
            row[side] = tuple(Unit(**unit) for unit in row[side])
        return cls(**row)
