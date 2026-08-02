"""FastAPI server: WebRTC signalling plus the static client.

Streamlit cannot do this. Real-time bidirectional audio needs a peer
connection that stays open for the whole call, and Streamlit's execution
model re-runs the script on every interaction. That is the reason this
project is a FastAPI app.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pipecat.transports.smallwebrtc.connection import IceServer
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from interview.report import render_markdown
from interview.store import store

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

STATIC_DIR = Path(__file__).parent / "static"
_background_tasks: set[asyncio.Task] = set()

TURN_USER = os.getenv("TURN_USERNAME")
TURN_PASS = os.getenv("TURN_CREDENTIAL")

if not TURN_USER or not TURN_PASS:
    log.warning(
        "TURN credentials missing (TURN_USERNAME / TURN_CREDENTIAL). "
        "Local calls will work, but remote calls will fail ICE with 401."
    )
else:
    log.info("TURN credentials loaded for user %s***", TURN_USER[:6])

TURN_ONLY = os.getenv("TURN_ONLY", "1") == "1"

if TURN_ONLY:
    # Relay-only: forces every candidate through TURN. Slower to gather but
    # survives symmetric NAT on both ends, which is what breaks direct paths.
    ICE_SERVERS = [
        IceServer(
            urls="turn:global.relay.metered.ca:443?transport=tcp",
            username=TURN_USER,
            credential=TURN_PASS,
        ),
        IceServer(
            urls="turn:global.relay.metered.ca:80?transport=tcp",
            username=TURN_USER,
            credential=TURN_PASS,
        ),
    ]
    log.info("ICE mode: TURN relay only (%d servers)", len(ICE_SERVERS))
else:
    ICE_SERVERS = [
        IceServer(urls="stun:stun.l.google.com:19302"),
        IceServer(
            urls="turn:global.relay.metered.ca:443?transport=tcp",
            username=TURN_USER,
            credential=TURN_PASS,
        ),
    ]
    log.info("ICE mode: STUN + TURN (%d servers)", len(ICE_SERVERS))

webrtc_handler = SmallWebRTCRequestHandler(ice_servers=ICE_SERVERS)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Interview agent ready on http://localhost:%s", os.getenv("PORT", "7860"))
    yield
    await webrtc_handler.close()
    for task in list(_background_tasks):
        task.cancel()


app = FastAPI(title="Voice Interview Agent", lifespan=lifespan)


async def _launch_bot(connection) -> None:
    # Imported lazily so a missing API key surfaces per-call, not at boot.
    from bot import run_bot

    try:
        await run_bot(connection)
    except Exception:
        log.exception("Interview session ended with an error")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest):
    """WebRTC signalling endpoint. The browser posts an SDP offer here."""

    async def on_connection(connection):
        task = asyncio.create_task(_launch_bot(connection))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    answer = await webrtc_handler.handle_web_request(request, on_connection)
    return JSONResponse(answer)


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    """Poll target for the client. Returns a status until the report is ready."""
    record = store.get(session_id)
    if record is None:
        return JSONResponse({"status": "unknown"}, status_code=404)
    return JSONResponse(record.public())


@app.get("/api/report/{session_id}/download")
async def download_report(session_id: str):
    """Markdown version, as a file the candidate can keep."""
    record = store.get(session_id)
    if record is None or not record.report:
        return JSONResponse({"error": "Report not available"}, status_code=404)
    return PlainTextResponse(
        render_markdown(record.report),
        headers={
            "Content-Disposition": f'attachment; filename="interview-{session_id}.md"'
        },
    )


@app.get("/api/sessions")
async def list_sessions():
    """Recent sessions. This is the seed of the admin view."""
    return JSONResponse([r.summary() for r in store.list_recent()])


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "role": os.getenv("INTERVIEW_ROLE", "Machine Learning Engineer"),
    }


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=bool(os.getenv("DEV_RELOAD")),
    )