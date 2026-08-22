"""Streams every `redis_state.publish_update(...)` event (telemetry,
signal_command, emergency, incident) to connected WebSocket clients in
real time. The dashboard's live map/KPI widgets subscribe here instead of
polling the REST endpoints.

Runs the blocking Redis pubsub `listen()` loop in a thread (via
`run_in_threadpool`) so it doesn't block the asyncio event loop -- redis-py's
pubsub client is synchronous.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from traffic_system.api.deps import get_redis_state
from traffic_system.common.logging import get_logger
from traffic_system.ingestion.redis_state import RedisState

logger = get_logger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_updates(websocket: WebSocket, redis_state: RedisState = Depends(get_redis_state)) -> None:
    await websocket.accept()
    pubsub = redis_state.subscribe_updates()
    queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    stop = False

    def _pump() -> None:
        # Runs in a worker thread: blocking-poll Redis pubsub and hand
        # messages back to the event loop via call_soon_threadsafe.
        while not stop:
            message = pubsub.get_message(timeout=1.0)
            if message and message["type"] == "message":
                loop.call_soon_threadsafe(queue.put_nowait, message["data"])

    pump_task = asyncio.create_task(run_in_threadpool(_pump))

    try:
        while True:
            data = await queue.get()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        logger.info("websocket.client_disconnected")
    finally:
        stop = True
        pump_task.cancel()
        pubsub.close()
