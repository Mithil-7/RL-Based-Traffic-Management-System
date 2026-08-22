"""Runs one EdgeAgent per intersection in the city graph, all in a single
process using a lightweight round-robin loop (real deployments run one
process per Raspberry Pi; this is the convenient all-in-one-box mode for
local development, demos, and this repo's smoke tests).
"""
from __future__ import annotations

import argparse
import time

from traffic_system.common.config import get_settings
from traffic_system.common.logging import configure_logging, get_logger
from traffic_system.edge.edge_agent import build_simulated_agent
from traffic_system.env.city_graph import CityGraph

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run simulated edge agents for every intersection in the city graph.")
    parser.add_argument("--city-graph", default=str(settings.city_graph_path))
    args = parser.parse_args()

    city_graph = CityGraph.load(args.city_graph)
    agents = [build_simulated_agent(iid, seed=i) for i, iid in enumerate(city_graph.intersection_ids)]
    for agent in agents:
        agent.start()
    logger.info("edge_fleet.started", count=len(agents))

    try:
        while True:
            for agent in agents:
                agent.step()
            time.sleep(settings.decision_interval_seconds)
    except KeyboardInterrupt:
        logger.info("edge_fleet.shutting_down")
    finally:
        for agent in agents:
            agent.video_source.release()
            agent.mqtt.disconnect()


if __name__ == "__main__":
    main()
