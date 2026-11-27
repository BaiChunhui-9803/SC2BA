"""Run the macro library in SC2BA while retaining its opponent and battle logic."""

from contextlib import redirect_stdout
from io import StringIO

import torch
from absl import flags
from s2clientprotocol import common_pb2 as common
from s2clientprotocol import debug_pb2 as debug
from s2clientprotocol import error_pb2 as errors
from s2clientprotocol import raw_pb2 as raw
from s2clientprotocol import sc2api_pb2 as sc

from sc2ba.enemycontrol.enemy_control import enemy_name
from sc2ba.env import Agent, Bot, Camp, StarCraft2BAEnv, difficulties
from sc2ba.env.starcraft2.maps import get_map_params

from .observation import Observation, Unit

OPPONENTS = {"bot": 0, **{name: code for name, code in enemy_name.items() if code}}
ABILITIES = {"smart": 1, "attack": 3674, "move": 3794, "heal": 386}


def _macro_ability_ids(available):
    abilities = dict(ABILITIES)
    if abilities["move"] not in available:
        legacy_move = available.get(16)
        if (legacy_move is None or not legacy_move.available
                or legacy_move.link_name.casefold() != "move"
                or legacy_move.button_name.casefold() != "move"):
            raise RuntimeError("SC2 provides neither GeneralMove 3794 nor a verified Move ability 16")
        abilities["move"] = 16
    return abilities


# Standard SC2 weapon ranges by StarCraft2BAEnv.get_unit_type_id()
UNIT_ATTACK_RANGE = {
    1: 5.0,  # marine
    2: 6.0,  # marauder
    3: 4.0,  # medivac (heal range, non-combat)
    4: 6.0,  # stalker
    5: 0.5,  # zealot (melee)
    6: 6.0,  # colossus
    7: 0.5,  # zergling
    8: 5.0,  # hydralisk
    9: 0.5,  # baneling
}


class _MixedActions:
    """Join ordered red macro commands to the blue actions in SC2BA's step RPC."""

    def __init__(self, controller, actions):
        self.controller = controller
        self.red_actions = actions

    def actions(self, request):
        actions = sc.RequestAction(actions=[*self.red_actions, *request.actions])
        response = self.controller.actions(actions)
        if response is not None:
            rejected = [(index, errors.ActionResult.Name(result))
                        for index, result in enumerate(response.result[:len(self.red_actions)])
                        if result != errors.Success]
            if rejected:
                raise RuntimeError(f"SC2 rejected red macro commands: {rejected}")
        return response

    def __getattr__(self, name):
        return getattr(self.controller, name)


class Arena(StarCraft2BAEnv):
    def __init__(self, scenario, seed, *, step_mul=8):
        self.abilities = dict(ABILITIES)
        params = get_map_params(scenario.name)
        allowed_ids = params.get("blue_control_models", [0])
        for opponent in scenario.opponents:
            if OPPONENTS.get(opponent) not in allowed_ids:
                raise ValueError(f"{opponent} is unavailable in the {scenario.name} opponent pool.")
        bounds = params["playable_area"]
        self.bounds = tuple(zip(bounds["lower_left"], bounds["upper_right"]))
        for layout in scenario.layouts:
            self._validate_layout(layout, params)
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["bch"])
        players = [
            Agent(params["a_race"], Camp.RED.name),
            Bot(params["b_race"], Camp.BLUE.name, difficulties["7"]),
        ]
        use_models = any(OPPONENTS[name] for name in scenario.opponents)
        super().__init__(map_name=scenario.name, players=players, seed=seed, step_mul=step_mul,
                         random_enemy=use_models, window_size_x=320, window_size_y=240)
        self.default_blue = self.blue_start_position
        self.default_mirror = self.mirror_create_blue_camp
        self.default_blue_units = dict(params["blue_units"])
        self.initial_scale = 16 * max(self.n_red_agents, self.n_blue_agents)
        self.layout = None
        self._raw_step = False
        self.launches = 0
        self.last_end_loop = None

    def _validate_layout(self, layout, params):
        if layout.red_units:
            if (len(layout.red_units), len(layout.blue_units)) != (params["n_agents"], params["n_enemies"]):
                raise ValueError(f"{layout.id} must match the scenario's unit counts.")
            points = layout.red_units + layout.blue_units
        else:
            points = (layout.red, layout.blue) if layout.blue else (layout.red,)
        for point in points:
            if not all(lo <= value < hi for (lo, hi), value in zip(self.bounds, point)):
                raise ValueError(f"{layout.id} spawn {point} is outside the scenario's playable area.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _launch(self):
        super()._launch()
        self.abilities = _macro_ability_ids(self._controller.data().abilities)
        self.launches += 1
        if max(self.map_x, self.map_y) > 128:
            raise ValueError("The legacy 128x128 influence grid requires a SC2BA map at most 128 world units wide.")

    def create_all_units(self):
        if not self.layout.red_units:
            return super().create_all_units()
        commands = []
        for camp, positions, use_custom_unit in ((Camp.RED, self.layout.red_units, self.red_agent_flag),
                                                  (Camp.BLUE, self.layout.blue_units, self.blue_agent_flag)):
            unit_type = self._get_unit_type("marine", use_custom_unit)
            for x, y in positions:
                commands.append(debug.DebugCommand(create_unit=debug.DebugCreateUnit(
                    unit_type=unit_type, owner=camp.value, quantity=1, pos=common.Point2D(x=x, y=y),
                )))
        self._controller.debug(commands)
        for _ in range(40):
            self._controller.step(2)
            self._obs = self._controller.observe()
            if len(self._obs.observation.raw_data.units) == len(commands):
                return
        raise RuntimeError(f"SC2 did not spawn all units for {self.layout.id}.")

    def reset_round(self, layout, opponent):
        self.layout = layout
        self.random_enemy_seq = [OPPONENTS[opponent]]
        self.red_start_position = layout.red or self.red_start_position
        self.blue_start_position = layout.blue or self.default_blue
        self.mirror_create_blue_camp = self.default_mirror and layout.blue is None
        # Restore the declared blue roster (fixes stock maps whose mirrored spawn ignores blue_units).
        self.blue_units_config = dict(self.default_blue_units)
        restarts = self.force_restarts
        with redirect_stdout(StringIO()):
            super().reset()
        if self.force_restarts != restarts or (self.enemy_code or 0) != OPPONENTS[opponent]:
            raise RuntimeError("SC2BA reset did not produce the requested match.")
        observation = self.observe_battle()
        if self.last_end_loop is not None and observation.game_loop <= self.last_end_loop:
            raise RuntimeError("SC2 loaded a fresh game instead of reusing the current map.")
        for side, count in ((observation.friendly, self.n_red_agents), (observation.enemy, self.n_blue_agents)):
            if len(side) != count or any(
                unit.health_max <= 0
                or unit.health != unit.health_max
                or (unit.shield_max > 0 and unit.shield != unit.shield_max)
                for unit in side
            ):
                raise RuntimeError(f"Bad initial roster or HP for {self.map_name}/{layout.id}.")
        return observation

    def observe_battle(self):
        def read_side(side):
            result = []
            for unit in sorted(side.values(), key=lambda item: item.tag):
                if unit.health <= 0:
                    continue
                utype = self.get_unit_type_id(unit) or 1
                result.append(
                    Unit(
                        int(unit.tag),
                        float(unit.pos.x),
                        float(unit.pos.y),
                        float(unit.health),
                        float(unit.health_max),
                        float(unit.radius),
                        float(unit.weapon_cooldown),
                        shield=float(getattr(unit, "shield", 0.0)),
                        shield_max=float(getattr(unit, "shield_max", 0.0)),
                        unit_type=int(utype),
                        attack_range=float(UNIT_ATTACK_RANGE.get(utype, 5.0)),
                        is_medivac=(utype == 3),
                        energy=float(getattr(unit, "energy", 0.0)),
                    )
                )
            return tuple(result)
        return Observation(read_side(self.agents), read_side(self.enemies), self.episode_steps,
                           int(self._obs.observation.game_loop), self.map_y, self.initial_scale)

    def get_agent_action(self, a_id, action, camp=Camp.RED):
        if self._raw_step and camp == Camp.RED:
            return None
        return super().get_agent_action(a_id, action, camp)

    def _raw_action(self, command):
        raw_command = raw.ActionRawUnitCommand(
            ability_id=self.abilities[command.ability], unit_tags=command.units, queue_command=command.queued,
        )
        if isinstance(command.target, int):
            raw_command.target_unit_tag = command.target
        else:
            target = command.target
            if command.ability == "attack":
                # Convert the original fixed feature-grid point to the COMMON world.
                scale = max(self.map_x, self.map_y) / 128
                target = target[0] * scale, self.map_y - target[1] * scale
            x, y = [min(max(value, lo + 0.25), hi - 0.25) for value, (lo, hi) in zip(target, self.bounds)]
            raw_command.target_world_space_pos.CopyFrom(common.Point2D(x=x, y=y))
        return sc.Action(action_raw=raw.ActionRaw(unit_command=raw_command))

    def _macro_actions(self, commands):
        # Reissuing an active heal to the same target returns AlreadyTargeted.
        active_heals = {
            (unit.tag, unit.orders[0].target_unit_tag)
            for unit in self.agents.values()
            if unit.health > 0 and unit.orders and unit.orders[0].ability_id == self.abilities["heal"]
        }
        return [
            self._raw_action(command) for command in commands
            if command.ability != "heal"
            or not all((tag, command.target) in active_heals for tag in command.units)
        ]

    @torch.inference_mode()
    def step_macros(self, commands):
        controller = self._controller
        proxy = _MixedActions(controller, self._macro_actions(commands))
        self._controller = proxy
        self._raw_step = True
        try:
            placeholders = [int(self.agents[index].health > 0) for index in range(self.n_red_agents)]
            _, done, info = super().step(placeholders)
        finally:
            if self._controller is proxy:
                self._controller = controller
            self._raw_step = False
        if "dead_red" not in info:
            raise RuntimeError("SC2BA did not return a normal step after the macro.")
        observation = self.observe_battle()
        medivac_end = self.only_medivac_left(ally=True) or self.only_medivac_left(ally=False)
        if (
            done
            and observation.friendly
            and observation.enemy
            and observation.step != self.episode_limit
            and not medivac_end
        ):
            raise RuntimeError("Unexpected SC2BA terminal state.")
        if done:
            self.last_end_loop = observation.game_loop
        return observation, done, info

    def episode_details(self, first, final, info):
        outcome = ("win" if info["red_battle_won"] else "loss" if info["blue_battle_won"]
                   else "timeout" if final.friendly and final.enemy else "draw")
        return {
            "outcome": outcome, "steps": final.step, "step_limit": self.episode_limit,
            "final_game_loop": final.game_loop,
            "initial": first.record(), "final_hp": [sum(unit.effective_health for unit in side)
                                                     for side in (final.friendly, final.enemy)],
            "engine_pid": self._sc2_proc.pid, "engine_launches": self.launches,
            "engine_restarts": self.force_restarts, "sc2_build": self._run_config.version.build_version,
        }
