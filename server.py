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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from report import render_html, render_markdown
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


# ── Health monitoring (phase 1) ─────────────────────────────────────────
# Alerting moved out to .github/workflows/agent-health-check.yml +
# scripts/check_agent_health.py — a monitor that lives inside the thing it
# monitors goes silent exactly when it matters most (server.py crashing,
# Fly sleeping the front-end, a bad deploy). What stays here is read-only
# and on-demand: /api/health can report live agent reachability when asked,
# but nothing in this process alerts or runs on a schedule any more.
PIPECAT_TOKEN = os.getenv("PIPECAT_TOKEN", "")
EXPECTED_IMAGE = os.getenv("MUHAWIR_IMAGE", "eng7montaser/muhawir:latest")
PIPECAT_STATUS_API = "https://api.pipecat.daily.co"

if not PIPECAT_TOKEN:
    log.warning("PIPECAT_TOKEN is unset — /api/health's agent fields will be empty.")

_org_cache: str | None = None


async def _pipecat_status_get(session: aiohttp.ClientSession, path: str) -> dict | None:
    async with session.get(
        f"{PIPECAT_STATUS_API}{path}",
        headers={"Authorization": f"Bearer {PIPECAT_TOKEN}"},
        timeout=aiohttp.ClientTimeout(total=15),
    ) as resp:
        if resp.status != 200:
            log.warning("Pipecat status API %s returned %s", path, resp.status)
            return None
        return await resp.json()


async def _resolve_org(session: aiohttp.ClientSession) -> str | None:
    """The org a PAT belongs to, cached after the first successful lookup —
    it does not change between requests."""
    global _org_cache
    if _org_cache:
        return _org_cache
    data = await _pipecat_status_get(session, "/v1/organizations")
    orgs = (data or {}).get("organizations") or []
    if not orgs:
        return None
    _org_cache = orgs[0]["name"]
    return _org_cache


async def _get_agent_health() -> dict:
    """Live, on-demand snapshot for /api/health. No caching, no alerting —
    every call to /api/health makes a fresh request to Pipecat Cloud. That's
    fine for an endpoint meant to be checked occasionally by a human, not
    polled in a hot path.
    """
    if not PIPECAT_TOKEN:
        return {"healthy": None, "image": None, "error": "PIPECAT_TOKEN not set"}

    async with aiohttp.ClientSession() as session:
        org = await _resolve_org(session)
        if not org:
            return {"healthy": None, "image": None, "error": "Could not resolve Pipecat organization"}

        data = await _pipecat_status_get(
            session, f"/v1/organizations/{org}/services/{AGENT_NAME}"
        )
        if data is None:
            return {"healthy": None, "image": None, "error": "Could not reach Pipecat Cloud status API"}

        available = bool(data.get("available"))
        ready = bool(data.get("ready") or data.get("activeDeploymentReady"))
        image = (
            data.get("deployment", {}).get("manifest", {}).get("spec", {}).get("image", "")
        )
        return {
            "healthy": available and ready,
            "image": image,
            "error": None,
        }


app = FastAPI(title="Muhawir")


@app.post("/api/connect")
async def connect(request: Request):
    """Start an agent session and hand the browser a room to join.

    Cold starts are real: with min_agents at 0 the first request waits for a
    container, so the client should show a spinner rather than assume this
    returns instantly.
    """
    if not PIPECAT_API_KEY:
        return JSONResponse({"error": "Server missing PIPECAT_API_KEY"}, status_code=500)

    try:
        client_body = await request.json()
    except Exception:
        client_body = {}

    # Candidate-chosen topic/level. Capped so a pasted essay doesn't end up
    # bloating the system prompt it flows into (see Settings.for_session,
    # prompts.py). Missing/blank values fall back to the bot's env defaults.
    role = str(client_body.get("role") or "").strip()[:120]
    seniority = str(client_body.get("seniority") or "").strip()[:40]

    payload = {
        "createDailyRoom": True,
        # Arrives at the bot as runner_args.body — read in bot.py's bot().
        "body": {"role": role, "seniority": seniority},
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

    session_id = data.get("sessionId")
    if session_id:
        # Learned the moment the call starts, not just at the end — this is
        # what lets /report/{session_id} (and /api/report/{session_id}) tell
        # "still in progress" apart from "this session never existed". Before
        # this, the front-end store only heard about a session once, from
        # bot._finalise's single push at call end, so "in progress" was
        # never a state the front-end could distinguish from "unknown".
        store.create(session_id, role or "Interview", "en")

    log.info("Session started for agent %s", AGENT_NAME)
    return JSONResponse(
        {
            "room_url": data.get("dailyRoom"),
            "token": data.get("dailyToken"),
            "session_id": session_id,
        }
    )


# ── Reports ──────────────────────────────────────────────────────────────
# The bot runs in a different container, so it pushes its finished report
# here rather than the browser reading the bot's own store. Storage is this
# process's memory: enough for the minute of polling that follows a call,
# and lost on restart. Durable storage is a change to store.py alone.


@app.post("/api/report/{session_id}")
async def ingest_report(session_id: str, request: Request):
    """Called by the bot — at scoring start now, as well as at the end.
    Not for browsers."""
    if not INGEST_TOKEN or request.headers.get("X-Ingest-Token") != INGEST_TOKEN:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    status = body.get("status", "ready")

    if status == "ready" and body.get("report"):
        store.save(session_id, body["report"])
    else:
        # /api/connect already creates this record the moment the call
        # starts (status "running"). Only create here as a fallback for a
        # session this process never saw start — never overwrite it, or
        # every "scoring" push would wipe the record /api/connect made and
        # reset created_at, right when a candidate might be viewing it.
        if store.get(session_id) is None:
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


def _report_status_page(
    title: str, message: str, *, refresh: bool = False, http_status: int = 200
) -> HTMLResponse:
    """Shared shell for every non-ready state of /report/{session_id}.

    Six distinct states exist (unknown, running, scoring, empty, error,
    ready) and none of them collapse into another — "still scoring" and
    "this link is wrong" must never look the same page to someone who
    shared a link before the call ended.
    """
    refresh_tag = '<meta http-equiv="refresh" content="5">' if refresh else ""
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
{refresh_tag}
<title>{title} — Muhawir</title>
<style>
  body {{
    margin: 0; min-height: 100svh; display: grid; place-items: center;
    background: #0a0b0d; color: #e6e8eb; padding: 24px;
    font-family: "Space Grotesk", system-ui, sans-serif;
  }}
  main {{ max-width: 46ch; text-align: center; }}
  h1 {{ font-size: 22px; margin: 0 0 10px; }}
  p {{ color: #6b7280; font-size: 15px; line-height: 1.6; margin: 0; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p>{message}</p>
</main>
</body>
</html>""",
        status_code=http_status,
    )


@app.get("/report/{session_id}")
async def report_page(session_id: str):
    """Shareable permalink — the human-facing counterpart to the JSON
    /api/report/{session_id}. No auth: session_id is an unguessable
    Pipecat/Daily session UUID, the same trust model the JSON and
    download endpoints already use.
    """
    record = store.get(session_id)

    if record is None:
        return _report_status_page(
            "Report not found",
            "No interview report exists at this link. Check that you copied "
            "the full URL.",
            http_status=404,
        )

    if record.status in ("running", "scoring"):
        return _report_status_page(
            "Scoring in progress",
            "This interview just ended and the report is still being "
            "generated. This page will refresh automatically.",
            refresh=True,
        )

    if record.status == "empty":
        return _report_status_page(
            "No answers were scored",
            "The call ended before any question was answered, so there is "
            "nothing to report.",
        )

    if record.status == "error":
        return _report_status_page(
            "Report could not be generated",
            record.error or "Scoring failed for this session.",
            http_status=500,
        )

    if record.status != "ready" or not record.report:
        # Any future status this endpoint doesn't know about yet — fail
        # into an honest "not ready" rather than a misleading 404 or a
        # crash on record.report being None.
        return _report_status_page(
            "Report not ready",
            f"This report's status is '{record.status}'.",
        )

    return HTMLResponse(render_html(record.report))


@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse([r.summary() for r in store.list_recent()])


@app.get("/api/health")
async def health():
    # Live on-demand check, not a cached background result — see
    # _get_agent_health's docstring. Scheduled alerting lives entirely in
    # .github/workflows/agent-health-check.yml now, independent of this
    # endpoint and of this process being up at all.
    agent = await _get_agent_health()
    return {
        "status": "ok",
        "agent": AGENT_NAME,
        "key_configured": bool(PIPECAT_API_KEY),
        # Defaults for the pre-call topic/level form (static/index.html).
        # Mirrors config.py's own env-var defaults on the bot side.
        "role": os.getenv("INTERVIEW_ROLE", "Machine Learning Engineer"),
        "seniority": os.getenv("INTERVIEW_LEVEL", "mid"),
        "agent_healthy": agent["healthy"],
        "agent_image": agent["image"],
        "agent_check_error": agent["error"],
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