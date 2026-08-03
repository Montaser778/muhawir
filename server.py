"""FastAPI server: serves the client and brokers Pipecat Cloud sessions.

The bot no longer runs here. It runs on Pipecat Cloud over Daily, which is
what fixed the ICE failures this architecture used to hit on Fly. What is
left in this process is deliberately small:

  * serve static/index.html
  * exchange a POST /api/connect for a Daily room + token

The Pipecat public API key stays server-side. Calling /start straight from
the browser would expose it in devtools to anyone who opens the page.
"""

import logging
import os
from pathlib import Path

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from report import render_markdown
from store import store

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

STATIC_DIR = Path(__file__).parent / "static"

AGENT_NAME = os.getenv("PIPECAT_AGENT_NAME", "muhawir")
PIPECAT_API_KEY = os.getenv("PIPECAT_API_KEY", "")
START_URL = f"https://api.pipecat.daily.co/v1/public/{AGENT_NAME}/start"

# Shared secret between this server and the bot. Without it, anyone could POST
# a fabricated report for any session id.
INGEST_TOKEN = os.getenv("REPORT_INGEST_TOKEN", "")

if not PIPECAT_API_KEY:
    log.warning("PIPECAT_API_KEY is unset — /api/connect will fail with 500.")


app = FastAPI(title="Muhawir")


@app.post("/api/connect")
async def connect():
    """Start an agent session and hand the browser a room to join.

    Cold starts are real: with min_agents at 0 the first request waits for a
    container, so the client should show a spinner rather than assume this
    returns instantly.
    """
    if not PIPECAT_API_KEY:
        return JSONResponse({"error": "Server missing PIPECAT_API_KEY"}, status_code=500)

    payload = {
        "createDailyRoom": True,
        # Anything in "body" arrives at the bot as runner_args.body — the hook
        # for per-call config (candidate name, role) once the client sends it.
        "body": {},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                START_URL,
                headers={"Authorization": f"Bearer {PIPECAT_API_KEY}"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    log.error("Pipecat /start returned %s: %s", resp.status, data)
                    return JSONResponse(
                        {"error": data.get("error", "Failed to start session")},
                        status_code=resp.status,
                    )
    except Exception:
        log.exception("Could not reach Pipecat Cloud")
        return JSONResponse({"error": "Could not reach Pipecat Cloud"}, status_code=502)

    log.info("Session started for agent %s", AGENT_NAME)
    return JSONResponse(
        {
            "room_url": data.get("dailyRoom"),
            "token": data.get("dailyToken"),
            "session_id": data.get("sessionId"),
        }
    )


# ── Reports ──────────────────────────────────────────────────────────────
# The bot runs in a different container, so it pushes its finished report
# here rather than the browser reading the bot's own store. Storage is this
# process's memory: enough for the minute of polling that follows a call,
# and lost on restart. Durable storage is a change to store.py alone.


@app.post("/api/report/{session_id}")
async def ingest_report(session_id: str, request: Request):
    """Called by the bot when scoring finishes. Not for browsers."""
    if not INGEST_TOKEN or request.headers.get("X-Ingest-Token") != INGEST_TOKEN:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    status = body.get("status", "ready")

    if status == "ready" and body.get("report"):
        store.save(session_id, body["report"])
    else:
        store.create(session_id, body.get("role", ""), body.get("language", ""))
        store.set_status(session_id, status, body.get("error"))

    log.info("Report ingested for %s (%s)", session_id, status)
    return JSONResponse({"ok": True})



@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    record = store.get(session_id)
    if record is None:
        return JSONResponse({"status": "unknown"}, status_code=404)
    return JSONResponse(record.public())


@app.get("/api/report/{session_id}/download")
async def download_report(session_id: str):
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
    return JSONResponse([r.summary() for r in store.list_recent()])


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agent": AGENT_NAME,
        "key_configured": bool(PIPECAT_API_KEY),
    }


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=bool(os.getenv("DEV_RELOAD")),
    )