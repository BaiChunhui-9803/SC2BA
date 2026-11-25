"""Supplement B01 with a fixed-camp reference against SC2's native Bot."""

import argparse
import csv
import io
import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from run_b01_pairwise import (
    GAME_VERSION, POLICIES, ROOT, configure_actor, load_layouts, sha256, write_csv, write_json,
)

import sc2ba
from sc2ba.enemycontrol.enemy_control import EnemyControl, enemy_name
from sc2ba.env import Agent, Bot, Camp, StarCraft2BAEnv, difficulties


DIFFICULTY = "7"
FIELDS = (
    "layout_id", "x", "y", "policy", "agent_camp", "agent_result", "steps", "session_seed",
    "agent_return", "bot_return", "dead_agent", "dead_bot",
)


class BotReference:
    def __init__(self, seed):
        self.seed = seed
        self.version = None
        self.players = [Agent("T", Camp.RED.name), Bot("T", Camp.BLUE.name, difficulties[DIFFICULTY])]
        self.env = StarCraft2BAEnv(
            map_name="3m", players=self.players, seed=seed,
            random_enemy=False, window_size_x=320, window_size_y=240,
        )
        with redirect_stdout(io.StringIO()):
            policies = EnemyControl(self.env.get_env_info(), "3m")
        if not all(policies.mac_pool[enemy_name[name]].args.test_greedy for name in POLICIES):
            raise ValueError("B01 requires greedy policies")
        self.policies = policies

    def close(self):
        self.env.close()

    @torch.inference_mode()
    def play(self, layout, name):
        env = self.env
        layout_id, point = layout
        env.red_start_position = point
        restarts = env.force_restarts
        with redirect_stdout(io.StringIO()):
            env.reset()
        if self.version is None:
            self.version = env._controller.ping().game_version
            if self.version != GAME_VERSION:
                raise ValueError(f"Expected SC2 {GAME_VERSION}, found {self.version}")
        if env._obs.observation.player_common.player_id != Camp.RED.value:
            raise ValueError("SC2 did not assign the participant to the intended camp")
        if len(env.agents) != 3 or len(env.enemies) != 3 or env.force_restarts != restarts:
            raise ValueError("Invalid reset for the built-in Bot comparison")
        if not np.allclose(
            np.mean([(u.pos.x, u.pos.y) for u in env.agents.values()], axis=0)
            + np.mean([(u.pos.x, u.pos.y) for u in env.enemies.values()], axis=0),
            (2 * env.center_x, 2 * env.center_y), atol=0.05,
        ):
            raise ValueError("Spawn points are not mirrored")

        self.policies.reset(enemy_name[name])
        controller = self.policies
        returns = [0.0, 0.0]
        for step in range(1, env.episode_limit + 1):
            t = step - 1
            obs = np.asarray(env.get_obs(Camp.RED), dtype=np.float32)
            available = np.asarray(env.get_avail_actions(Camp.RED), dtype=np.int32)
            if obs.shape != (3, 30) or available.shape != (3, 9) or not np.isfinite(obs).all():
                raise ValueError("Invalid observation or availability during Bot reference")
            controller.batch.update({"obs": obs[None, ...], "avail_actions": available[None, ...]}, ts=t)
            actions = controller.mac.select_actions(controller.batch, t_ep=t, test_mode=True)[0].tolist()
            if len(actions) != 3 or not available[np.arange(3), actions].all():
                raise ValueError("Agent policy selected an unavailable action")
            rewards, done, info = env.step(actions)
            if env.force_restarts != restarts or "red_battle_won" not in info or "blue_battle_won" not in info:
                raise ValueError("SC2 returned an interrupted match")
            controller.update_action_reward(actions, rewards[0], done, t)
            for i, reward in enumerate(rewards):
                returns[i] += reward
            if done:
                break
        else:
            raise ValueError("Bot reference did not terminate at the scenario limit")
        won = [info["red_battle_won"], info["blue_battle_won"]]
        if all(won) or not np.isfinite(returns).all():
            raise ValueError("Invalid outcome in built-in Bot reference")
        outcome = "win" if won[0] else "loss" if won[1] else "draw"
        dead = [info["dead_red"], info["dead_blue"]]
        return {
            "layout_id": layout_id, "x": point[0], "y": point[1],
            "policy": name, "agent_camp": "red", "agent_result": outcome,
            "steps": step, "session_seed": self.seed, "agent_return": returns[0],
            "bot_return": returns[1], "dead_agent": dead[0], "dead_bot": dead[1],
        }


def run_reference(seed, layouts, output_path, names):
    configure_actor(seed)
    arena = BotReference(seed)
    try:
        with output_path.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for layout in layouts:
                for name in names:
                    writer.writerow(arena.play(layout, name))
                file.flush()
    finally:
        arena.close()
    print(f"{output_path.stem}: {len(layouts) * len(names)} games", flush=True)
    return output_path


def summarize(paths, layouts, output, workers, elapsed):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            rows.extend(csv.DictReader(file))
    keys = {(int(row["layout_id"]), row["policy"], row["agent_camp"]) for row in rows}
    expected = {(index, name, camp.name.lower()) for index, _ in layouts for name in POLICIES for camp in (Camp.RED,)}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError("Missing or duplicate layouts, policies or camps in the Bot reference")
    points = dict(layouts)
    if any((float(row["x"]), float(row["y"])) != points[int(row["layout_id"])] for row in rows):
        raise ValueError("Bot reference changed the frozen layout set")
    counts = Counter((row["policy"], row["agent_camp"], row["agent_result"]) for row in rows)
    records = []
    for name in POLICIES:
        record = {"policy": name, "games": len(layouts)}
        for outcome, key in (("win", "wins"), ("loss", "losses"), ("draw", "draws")):
            record[key] = counts[name, "red", outcome]
        if sum(record[key] for key in ("wins", "losses", "draws")) != record["games"]:
            raise ValueError(f"Invalid result totals for {name}")
        record["win_rate"] = record["wins"] / record["games"]
        records.append(record)
    records.sort(key=lambda row: (-row["win_rate"], row["policy"]))
    write_csv(output / "bot-matches.csv", sorted(rows, key=lambda row: (
        int(row["layout_id"]), row["policy"], row["agent_camp"])), FIELDS)
    write_csv(output / "bot-reference.csv", records, tuple(records[0]))
    write_json(output / "bot-report.json", {
        "scenario": "3m", "game_version": GAME_VERSION, "bot_difficulty": DIFFICULTY,
        "agent_camp": "red", "layouts": len(layouts), "games": len(rows), "workers": workers, "elapsed_seconds": elapsed,
        "rows": records, "script_sha256": sha256(Path(__file__)),
    })
    print(f"B01 native Bot reference verified: {len(rows):,} matches.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "full"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=26112502)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    if Path(sc2ba.__file__).resolve().parent.parent != ROOT:
        parser.error(f"Load sc2ba from this checkout using PYTHONPATH={ROOT}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    layouts = load_layouts()
    if args.stage == "smoke":
        run_reference(args.seed, layouts[:2], output / "preflight-red.csv", POLICIES)
    else:
        if not (output / "preflight-red.csv").is_file():
            raise ValueError("Verify the native Bot setup in the preflight")
        if any(output.glob("bot-worker-*.csv")):
            raise FileExistsError("Existing built-in reference output")
        start = time.monotonic()
        print(f"Starting {len(layouts) * len(POLICIES):,} Bot reference games", flush=True)
        paths = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(run_reference, args.seed + 1000 * i, layouts[i::args.workers],
                                output / f"bot-worker-{i:02d}.csv", POLICIES)
                    for i in range(args.workers)]
            for job in as_completed(jobs):
                paths.append(job.result())
        summarize(paths, layouts, output, args.workers, round(time.monotonic() - start, 2))
