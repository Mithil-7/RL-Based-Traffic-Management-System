"""Train the DQN agent(s) against `TrafficGridEnv` (the real, networked city
graph -- not the single-intersection toy env) and write checkpoints to
`models/` in the layout `brain_service.py` expects: one `{intersection_id}.npz`
(or `.pt`, with `--backend torch`) file per intersection.

Two training modes:

`--share-weights` (default, recommended for scaling to a real city): every
intersection is controlled by the *same* network during the rollout, and
every intersection's transitions land in one shared replay buffer. This is
standard practice for many-agent traffic control (you cannot realistically
train hundreds of independent networks to convergence) and is exactly what
`agents/dqn_agent.py`'s docstring describes. The single trained checkpoint
is duplicated under every intersection's filename so the brain service's
per-intersection loading code works unmodified; nothing stops a later
`--share-weights=false` run once you have the compute budget to specialize.

`--no-share-weights`: one independent agent per intersection, each learning
only from its own local transitions. Higher potential ceiling per
intersection, far more sample-inefficient, only worth it for a handful of
unusually-shaped intersections.

Example:
    python scripts/train_dqn.py --episodes 200 --backend numpy
    python scripts/train_dqn.py --episodes 500 --backend torch --no-share-weights
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from traffic_system.agents.dqn_agent import IntersectionDQNAgent
from traffic_system.common.config import get_settings
from traffic_system.common.logging import configure_logging, get_logger
from traffic_system.env.city_graph import CityGraph
from traffic_system.env.traffic_grid_env import ACTION_PHASES, OBS_DIM, SimConfig, TrafficGridEnv

logger = get_logger(__name__)


def evaluate(env: TrafficGridEnv, act_fn, episode_steps: int, seed: int) -> dict[str, float]:
    """Runs one full greedy (no-exploration) episode and returns aggregate KPIs."""
    obs = env.reset(seed=seed)
    total_queue, emergency_steps = 0.0, 0
    for _ in range(episode_steps):
        actions = {iid: act_fn(iid, obs[iid]) for iid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for info in infos.values():
            total_queue += info["total_queue"]
            emergency_steps += int(info["has_emergency"])
        if all(truncs.values()):
            break
    final_discharged = sum(s.total_discharged for s in env.states.values())
    n = len(env.agents) * episode_steps
    return {
        "avg_queue_per_intersection_per_step": total_queue / n,
        "total_vehicles_discharged": final_discharged,
        "emergency_intersection_steps": emergency_steps,
    }


def train(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    city_graph = CityGraph.load(args.city_graph)
    sim_config = SimConfig(dt_s=settings.decision_interval_seconds, max_episode_steps=args.steps_per_episode)
    env = TrafficGridEnv(city_graph, sim_config, seed=args.seed)

    n_actions = len(ACTION_PHASES)
    if args.share_weights:
        shared_agent = IntersectionDQNAgent(OBS_DIM, n_actions, intersection_id="shared", backend=args.backend, seed=args.seed)
        agents = {iid: shared_agent for iid in env.agents}
    else:
        agents = {
            iid: IntersectionDQNAgent(OBS_DIM, n_actions, intersection_id=iid, backend=args.backend, seed=args.seed + i)
            for i, iid in enumerate(env.agents)
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    logger.info(
        "train.start",
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        share_weights=args.share_weights,
        backend=args.backend,
        intersections=len(env.agents),
    )

    t_start = time.monotonic()
    for episode in range(args.episodes):
        obs = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        losses: list[float] = []

        for _ in range(args.steps_per_episode):
            actions = {iid: agents[iid].act(obs[iid], explore=True) for iid in env.agents}
            next_obs, rewards, terms, truncs, infos = env.step(actions)

            for iid in env.agents:
                done = terms[iid] or truncs[iid]
                agents[iid].remember(obs[iid], actions[iid], rewards[iid], next_obs[iid], done)
                episode_reward += rewards[iid]

            # One gradient step per unique agent per environment step (not
            # per intersection, when weight-sharing -- a shared agent should
            # not take 9x the gradient steps of an independent one per env step).
            for agent in {id(a): a for a in agents.values()}.values():
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

            obs = next_obs
            if all(truncs.values()):
                break

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        record = {
            "episode": episode,
            "total_reward": round(episode_reward, 1),
            "avg_loss": round(avg_loss, 5),
            "epsilon": round(next(iter(agents.values())).epsilon, 4),
        }
        history.append(record)

        if episode % args.log_every == 0 or episode == args.episodes - 1:
            elapsed = time.monotonic() - t_start
            logger.info("train.episode", elapsed_s=round(elapsed, 1), **record)

        if args.eval_every and (episode % args.eval_every == 0) and episode > 0:
            kpis = evaluate(
                env,
                act_fn=lambda iid, o: agents[iid].act(o, explore=False),
                episode_steps=args.steps_per_episode,
                seed=999_000 + episode,
            )
            logger.info("train.eval", episode=episode, **kpis)

    # --- Save checkpoints in the layout brain_service.py expects ---
    ext = "npz" if args.backend in ("numpy", "auto") and not _is_torch_backend(next(iter(agents.values()))) else "pt"
    if args.share_weights:
        shared_path = output_dir / f"shared.{ext}"
        shared_agent.save(str(shared_path))
        for iid in env.agents:
            target = output_dir / f"{iid}.{ext}"
            shutil.copyfile(shared_path, target)
        logger.info("train.saved_shared_checkpoint", path=str(shared_path), duplicated_for=len(env.agents))
    else:
        for iid, agent in agents.items():
            path = output_dir / f"{iid}.{ext}"
            agent.save(str(path))
        logger.info("train.saved_independent_checkpoints", count=len(agents), dir=str(output_dir))

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    logger.info("train.done", total_time_s=round(time.monotonic() - t_start, 1), history_file=str(history_path))


def _is_torch_backend(agent: IntersectionDQNAgent) -> bool:
    return type(agent.online).__name__ == "TorchDuelingQNetwork"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings = get_settings()
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps-per-episode", type=int, default=360, help="360 steps * 10s dt = 1 simulated hour")
    parser.add_argument("--backend", choices=["torch", "numpy", "auto"], default=settings.qnet_backend)
    parser.add_argument("--share-weights", dest="share_weights", action="store_true", default=True)
    parser.add_argument("--no-share-weights", dest="share_weights", action="store_false")
    parser.add_argument("--city-graph", default=str(settings.city_graph_path))
    parser.add_argument("--output-dir", default=str(settings.model_dir))
    parser.add_argument("--eval-every", type=int, default=25, help="0 disables periodic evaluation")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
