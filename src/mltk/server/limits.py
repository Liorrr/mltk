"""ASGI middleware for limiting request body sizes."""
from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    """Reject invalid, oversized, and chunked request bodies above a fixed limit."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    def _too_large_detail(self) -> str:
        mib = 1024 * 1024
        if self.max_bytes % mib == 0:
            return f"Request body too large (max {self.max_bytes // mib}MB)"
        return f"Request body too large (max {self.max_bytes} bytes)"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                length = int(raw_length.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                await self._respond(send, 400, "Invalid Content-Length header")
                return
            if length < 0:
                await self._respond(send, 400, "Invalid Content-Length header")
                return
            if length > self.max_bytes:
                await self._respond(send, 413, self._too_large_detail())
                return
            await self.app(scope, receive, send)
            return

        total = 0
        rejected = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal total, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    rejected = True
                    if not response_started:
                        await self._respond(send, 413, self._too_large_detail())
                    return {"type": "http.disconnect"}
            return message

        async def limited_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, limited_receive, limited_send)

    async def _respond(self, send: Send, status_code: int, detail: str) -> None:
        import json

        body = json.dumps({"detail": detail}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
