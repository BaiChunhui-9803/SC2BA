"""B01 frozen-policy round robin with both 3m camps controlled by agents."""

import argparse
import copy
import csv
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from pysc2 import run_configs
from pysc2.lib import protocol

import sc2ba
from sc2ba.enemycontrol.enemy_control import EnemyControl, enemy_name
from sc2ba.env import Agent, Camp, StarCraft2BAEnv


ROOT = Path(__file__).resolve().parents[1]
LAYOUT_SOURCE = ROOT / "docs" / "data" / "test-outcomes.csv"
GAME_VERSION = "5.0.16.97425"
POLICIES = tuple(name for name, code in enemy_name.items() if code != 0)
CAMPS = (Camp.RED, Camp.BLUE)
MOVE_PAIRS = {2: 3, 3: 2, 4: 5, 5: 4}
MOVE_VECTORS = {2: (0, 1), 3: (0, -1), 4: (1, 0), 5: (-1, 0)}
MATCH_FIELDS = (
    "worker", "session", "session_seed", "layout_id", "x", "y", "red", "blue",
    "red_result", "steps", "red_return", "blue_return", "dead_red", "dead_blue",
)


class InterruptedGame(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_layouts():
    with LAYOUT_SOURCE.open(encoding="utf-8", newline="") as file:
        points = sorted({(float(row["x"]), float(row["y"])) for row in csv.DictReader(file)})
    if len(points) != 144:
        raise ValueError(f"Expected 144 predeclared layouts, got {len(points)}")
    return tuple(enumerate(points, start=1))


def create_env(seed):
    return StarCraft2BAEnv(
        map_name="3m", players=[Agent("T", camp.name) for camp in CAMPS],
        random_enemy=False, seed=seed, window_size_x=320, window_size_y=240,
    )


class DuelArena:
    def __init__(self, seed):
        self.seed = seed
        self.session = 0
        self.retries = 0
        self.version = None
        self.env = create_env(seed)
        info = self.env.get_env_info()
        if (info["n_agents"], info["n_enemies"], info["obs_red_shape"], info["obs_blue_shape"],
                info["n_red_actions"], info["n_blue_actions"]) != (3, 3, 30, 30, 9, 9):
            raise ValueError(f"Unexpected 3m observation or action dimensions: {info}")
        with redirect_stdout(io.StringIO()):
            pool = EnemyControl(info, "3m")
        if not all(pool.mac_pool[enemy_name[name]].args.test_greedy for name in POLICIES):
            raise ValueError("Both camps require greedy action selection")
        self.controllers = (pool, copy.deepcopy(pool))

    def close(self):
        self.env.close()

    def mirror_audit(self):
        env = self.env
        positions = [
            [(unit.pos.x, unit.pos.y) for unit in units.values()]
            for units in (env.agents, env.enemies)
        ]
        red, blue = positions
        if len(red) != 3 or len(blue) != 3:
            raise ValueError("Both camps must spawn three Marines")
        expected = [(2 * env.center_x - x, 2 * env.center_y - y) for x, y in red]
        error = min(max(math.hypot(x - bx, y - by) for (x, y), (bx, by) in zip(expected, order))
                    for order in itertools.permutations(blue))
        if error > 2.0:
            raise ValueError(f"Expected mirrored unit positions, max error {error}")
        checks = 0
        for camp, units in zip(CAMPS, (env.agents, env.enemies)):
            for unit_id, unit in units.items():
                mask = env.get_avail_agent_actions(unit_id, camp=camp)
                for action, opposite in MOVE_PAIRS.items():
                    if not mask[action] or (camp == Camp.BLUE and not mask[opposite]):
                        continue
                    command = env.get_agent_action(unit_id, action, camp=camp)
                    pos = command.action_raw.unit_command.target_world_space_pos
                    expected_move = opposite if camp == Camp.BLUE else action
                    dx, dy = MOVE_VECTORS[expected_move]
                    if not np.allclose((pos.x - unit.pos.x, pos.y - unit.pos.y),
                                       (dx * env._move_amount, dy * env._move_amount), atol=1e-5):
                        raise ValueError("SC2BA movement does not follow the expected camp orientation")
                    checks += 1
        if not checks:
            raise ValueError("No legal mirrored moves were available for the audit")
        return error, checks

    @torch.inference_mode()
    def play(self, red, blue, layout, *, audit=False):
        layout_id, point = layout
        env = self.env
        env.red_start_position = point
        restarts = env.force_restarts
        with redirect_stdout(io.StringIO()):
            env.reset()
        if env.force_restarts != restarts:
            raise InterruptedGame("SC2 engine restarted during reset")
        if self.version is None:
            self.version = env._controller.ping().game_version
            if self.version != GAME_VERSION:
                raise RuntimeError(f"Expected SC2 {GAME_VERSION}, found {self.version}")
        for controller, name in zip(self.controllers, (red, blue)):
            controller.reset(enemy_name[name])
        position_error, moves_checked = self.mirror_audit() if audit else (0.0, 0)
        previous = None
        returns = [0.0, 0.0]
        for step in range(1, env.episode_limit + 1):
            t = step - 1
            decisions = []
            for index, (camp, controller) in enumerate(zip(CAMPS, self.controllers)):
                obs = np.asarray(env.get_obs(camp), dtype=np.float32)
                available = np.asarray(env.get_avail_actions(camp), dtype=np.int32)
                if audit:
                    if obs.shape != (3, 30) or available.shape != (3, 9) or not np.isfinite(obs).all():
                        raise ValueError("Invalid policy inputs")
                    if previous is not None:
                        history = controller.batch["actions_onehot"][0, t - 1]
                        if history.argmax(dim=-1).tolist() != previous[index] or history.sum().item() != 3:
                            raise ValueError("Previous actions are misaligned in recurrent policy history")
                controller.batch.update({
                    "obs": obs[None, ...], "avail_actions": available[None, ...],
                }, ts=t)
                actions = controller.mac.select_actions(controller.batch, t_ep=t, test_mode=True)[0].tolist()
                if len(actions) != 3 or not available[np.arange(3), actions].all():
                    raise ValueError(f"Policy selected an unavailable action: {camp.name} {actions}")
                decisions.append(actions)
            rewards, done, info = env.step(decisions[0] + decisions[1])
            if env.force_restarts != restarts or "red_battle_won" not in info or "blue_battle_won" not in info:
                raise InterruptedGame("No valid SC2 outcome information")
            for index, controller in enumerate(self.controllers):
                returns[index] += rewards[index]
                controller.update_action_reward(decisions[index], rewards[index], done, t)
            previous = decisions
            if done:
                break
        else:
            raise ValueError("Episode did not terminate at the scenario limit")
        if not np.isfinite(returns).all() or (info["red_battle_won"] and info["blue_battle_won"]):
            raise ValueError(f"Invalid terminal state: {info}")
        outcome = "win" if info["red_battle_won"] else "loss" if info["blue_battle_won"] else "draw"
        return {
            "session": self.session, "session_seed": self.seed + self.session,
            "layout_id": layout_id, "x": point[0], "y": point[1],
            "red": red, "blue": blue, "red_result": outcome, "steps": step,
            "red_return": float(returns[0]), "blue_return": float(returns[1]),
            "dead_red": info["dead_red"], "dead_blue": info["dead_blue"],
        }, (position_error, moves_checked)

    def replay_if_interrupted(self, red, blue, layout):
        try:
            return self.play(red, blue, layout)[0]
        except (protocol.ProtocolError, protocol.ConnectionError, InterruptedGame) as error:
            self.close()
            self.retries += 1
            self.session += 1
            self.env = create_env(self.seed + self.session)
            self.version = None
            print(f"Relaunched SC2 after {type(error).__name__} for {red}/{blue} layout {layout[0]}", flush=True)
            return self.play(red, blue, layout)[0]


def configure_actor(seed):
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("Use the CPU environment for B01 actors")
    torch.manual_seed(seed)


def smoke(output, seed, layouts):
    configure_actor(seed)
    arena = DuelArena(seed)
    rows = []
    audits = []
    try:
        for index, red in enumerate(POLICIES):
            blue = POLICIES[(index + 1) % len(POLICIES)]
            for first, second in ((red, blue), (blue, red)):
                row, audit = arena.play(first, second, layouts[index], audit=True)
                rows.append(row)
                audits.append(audit)
    finally:
        arena.close()
    path = output / "preflight.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, {
        "passed": True, "game_version": arena.version, "seed": seed,
        "games": len(rows), "max_mirror_error": max(value for value, _ in audits),
        "movement_checks": sum(checks for _, checks in audits), "matches": rows,
    })
    print(f"B01a passed: {len(rows)} games, {sum(checks for _, checks in audits)} verified moves", flush=True)


def run_worker(index, seed, layouts, output):
    configure_actor(seed)
    arena = DuelArena(seed)
    output_path = output / f"worker-{index:02}.csv"
    combinations = tuple(itertools.combinations(POLICIES, 2))
    matchups = tuple(pair for left, right in combinations for pair in ((left, right), (right, left)))
    matchups += tuple((name, name) for name in POLICIES)
    completed = 0
    try:
        with output_path.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=MATCH_FIELDS)
            writer.writeheader()
            for layout in layouts:
                # Alternate the order of reciprocal games across the fixed layouts.
                for first, second in matchups if layout[0] % 2 else reversed(matchups):
                    row = arena.replay_if_interrupted(first, second, layout)
                    writer.writerow({"worker": index, **row})
                    completed += 1
                file.flush()
                if completed % 192 == 0 or completed == len(layouts) * len(matchups):
                    print(f"Worker {index}: {completed}/{len(layouts) * len(matchups)} games", flush=True)
    finally:
        arena.close()
    return output_path, arena.retries


def aggregate_matches(paths, layouts, output):
    expected = {(ident, red, blue) for ident, _ in layouts for red in POLICIES for blue in POLICIES}
    rows = []
    totals = Counter()
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (int(row["layout_id"]), row["red"], row["blue"])
                if key not in expected or key in seen:
                    raise ValueError(f"Unknown or duplicate match: {key}")
                expected_point = dict(layouts)[key[0]]
                if (float(row["x"]), float(row["y"])) != expected_point:
                    raise ValueError(f"Incorrect layout: {key}")
                seen.add(key)
                rows.append(row)
                totals[row["red"], row["blue"], row["red_result"]] += 1
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} matches")

    matches = len(layouts)
    cells = []
    for policy, opponent in itertools.permutations(POLICIES, 2):
        red = {key: totals[policy, opponent, key] for key in ("win", "loss", "draw")}
        blue = {key: totals[opponent, policy, reverse] for key, reverse in
                (("win", "loss"), ("loss", "win"), ("draw", "draw"))}
        if sum(red.values()) != matches or sum(blue.values()) != matches:
            raise ValueError(f"Outcome counts differ from the required side balance: {policy}, {opponent}")
        cells.append({
            "policy": policy, "opponent": opponent, "games": 2 * matches,
            "wins": red["win"] + blue["win"], "losses": red["loss"] + blue["loss"],
            "draws": red["draw"] + blue["draw"], "win_rate": (red["win"] + blue["win"]) / (2 * matches),
            "red_wins": red["win"], "red_losses": red["loss"], "red_draws": red["draw"],
            "blue_wins": blue["win"], "blue_losses": blue["loss"], "blue_draws": blue["draw"],
        })
    lookup = {(cell["policy"], cell["opponent"]): cell for cell in cells}
    for key, value in lookup.items():
        reverse = lookup[key[::-1]]
        if value["wins"] != reverse["losses"] or value["draws"] != reverse["draws"]:
            raise ValueError(f"Reciprocal matrix entries disagree: {key}")

    diagonal = [{
        "policy": name, "games": matches, "red_wins": totals[name, name, "win"],
        "blue_wins": totals[name, name, "loss"], "draws": totals[name, name, "draw"],
    } for name in POLICIES]
    if any(row["red_wins"] + row["blue_wins"] + row["draws"] != matches for row in diagonal):
        raise ValueError("Invalid self-play totals")
    ranking = sorted(
        ({"policy": policy, "mean_win_rate": sum(lookup[policy, opponent]["win_rate"]
                                                for opponent in POLICIES if policy != opponent)
                                           / (len(POLICIES) - 1)}
         for policy in POLICIES),
        key=lambda row: (-row["mean_win_rate"], row["policy"]),
    )
    write_csv(output / "matches.csv", sorted(rows, key=lambda row: (
        int(row["layout_id"]), row["red"], row["blue"])), MATCH_FIELDS)
    write_csv(output / "pairwise.csv", cells, tuple(cells[0]))
    write_csv(output / "win-rate-matrix.csv", [
        {"policy": policy, **{opponent: "" if policy == opponent
                              else round(100 * lookup[policy, opponent]["win_rate"], 6)
                              for opponent in POLICIES}}
        for policy in POLICIES
    ], ("policy", *POLICIES))
    write_csv(output / "self-play.csv", diagonal, tuple(diagonal[0]))
    return {
        "policies": POLICIES, "layouts": matches, "cells": cells, "ranking": ranking, "self_play": diagonal,
        "pairwise_games": len(rows) - sum(row["games"] for row in diagonal), "games": len(rows),
        "environment_steps": sum(int(row["steps"]) for row in rows),
    }


def run_matrix(args, layouts):
    preflight = json.loads((args.output / "preflight.json").read_text(encoding="utf-8"))
    if not preflight["passed"] or preflight["seed"] != args.seed or preflight["game_version"] != GAME_VERSION:
        raise ValueError("Run the matching preflight before the matrix")
    if any(args.output.glob("worker-*.csv")) or (args.output / "report.json").exists():
        raise FileExistsError("The output already contains a matrix run")
    started = time.monotonic()
    source_map = Path(sc2ba.__file__).parent / "env" / "starcraft2" / "maps" / "SC2BA_Maps" / "COMMON.SC2Map"
    installed_map = Path(run_configs.get().data_dir) / "Maps" / "SC2BA_Maps" / "COMMON.SC2Map"
    if sha256(source_map) != sha256(installed_map):
        raise ValueError("The installed SC2BA map differs from this checkout")
    weight_root = Path(sc2ba.__file__).parent / "enemycontrol" / "models" / "3m"
    fingerprints = {
        "map_sha256": sha256(source_map), "layout_source_sha256": sha256(LAYOUT_SOURCE),
        "script_sha256": sha256(Path(__file__)),
        "policy_sha256": {name: sha256(weight_root / name / "agent.th") for name in POLICIES},
        "config_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted((ROOT / "sc2ba" / "enemycontrol" / "config").rglob("*.yaml"))
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write_json(args.output / "protocol.json", {
        "scenario": "3m dual Agent", "game_version": GAME_VERSION, "seed": args.seed,
        "workers": args.workers, "layouts": layouts, "swap_colors": True, "test_greedy": True,
        **fingerprints,
    })
    partitions = [layouts[worker::args.workers] for worker in range(args.workers)]
    print(f"B01b starting: {len(layouts) * len(POLICIES) ** 2:,} games, {args.workers} CPU actors", flush=True)
    files = []
    retries = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_worker, index, args.seed + 1000 * index, partition, args.output)
                   for index, partition in enumerate(partitions, start=1)]
        for future in as_completed(futures):
            file, retry_count = future.result()
            files.append(file)
            retries += retry_count
    report = aggregate_matches(files, layouts, args.output)
    write_json(args.output / "report.json", {
        "batch": "b01-pairwise-20261125", "version": "frozen-round-robin-v1",
        "scenario": "3m", "game_version": GAME_VERSION, "seed": args.seed, "workers": args.workers,
        "retries": retries, "elapsed_seconds": round(time.monotonic() - started, 2), **report,
    })
    print(f"B01b validated: {report['games']:,} matches, {report['environment_steps']:,} steps, {retries} retries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "matrix"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26112501)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if Path(sc2ba.__file__).resolve().parent.parent != ROOT:
        parser.error(f"Load sc2ba from this checkout using PYTHONPATH={ROOT}")
    layouts = load_layouts()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        smoke(args.output, args.seed, layouts)
    else:
        run_matrix(args, layouts)
