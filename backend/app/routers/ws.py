"""WebSocket router — live presence for the /maps page.

`manager` is a module-level singleton so other routers can::

    from .ws import manager
    await manager.broadcast({"type": "update", ...})

to push map-data changes to all connected clients.
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts JSON messages to all of them."""

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)

    @property
    def count(self) -> int:
        return len(self._active)

    async def broadcast(self, message: dict) -> None:
        """Send `message` as JSON to every active connection; silently drop dead ones."""
        dead: list[WebSocket] = []
        for ws in list(self._active):  # snapshot — avoids mutation-during-iteration
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.discard(ws)


manager = ConnectionManager()

router = APIRouter()


@router.websocket("/ws/maps")
async def websocket_maps(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        await manager.broadcast({"type": "presence", "count": manager.count})
        while True:
            await ws.receive_text()  # keep-alive; ignore inbound text / echo pings here
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)
        await manager.broadcast({"type": "presence", "count": manager.count})
