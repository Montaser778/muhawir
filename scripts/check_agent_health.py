#!/usr/bin/env python3
"""Scheduled health check for the muhawir Pipecat Cloud agent.

Runs as a GitHub Action (.github/workflows/agent-health-check.yml) on a
cron schedule, deliberately OUTSIDE server.py: a monitor that lives inside
the thing it's monitoring goes silent exactly when it matters most —
server.py crashing, Fly sleeping the front-end machine, a bad deploy. This
script has no dependency on the front-end being up at all, and no
dependency beyond the Python standard library.

State (how many consecutive checks have come back unhealthy, and whether
an alert has already been sent for the current outage) is persisted
between runs as a GitHub Actions repository variable, HEALTH_CHECK_STATE,
via `gh variable`. Each workflow run is a fresh process with nothing else
to remember it by — without this, a sustained outage would re-alert every
single run (every 5 minutes, for as long as it lasts) instead of once.

Run locally to test: set PIPECAT_TOKEN / TELEGRAM_BOT_TOKEN /
TELEGRAM_CHAT_ID and run directly. State read/write is best-effort — if
`gh` isn't installed or authenticated, it warns and behaves as though
there is no prior state, rather than crashing.
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

PIPECAT_TOKEN = os.environ.get("PIPECAT_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
AGENT_NAME = os.environ.get("PIPECAT_AGENT_NAME", "muhawir")
EXPECTED_IMAGE = os.environ.get("MUHAWIR_IMAGE", "eng7montaser/muhawir:latest")

STATE_VAR = "HEALTH_CHECK_STATE"

# A single unlucky sample can land inside a normal cold-start window: the
# service scales to zero between calls (min_agents=0 by design), so
# available/ready legitimately read false for ~10-20s after a new call
# starts a fresh replica. That is not a failure. Requiring the SAME
# unhealthy result twice in a row, one check interval apart, makes it
# essentially impossible to false-positive on a cold start while still
# catching a real outage within two intervals.
FAILURE_THRESHOLD = 2


def _get_json(url: str, token: str) -> dict | None:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"GET {url} failed: {exc}", file=sys.stderr)
        return None


def resolve_org() -> str | None:
    data = _get_json("https://api.pipecat.daily.co/v1/organizations", PIPECAT_TOKEN)
    orgs = (data or {}).get("organizations") or []
    return orgs[0]["name"] if orgs else None


def fetch_docker_hub_digest(image: str) -> str | None:
    """Best-effort current digest for a public Docker Hub tag. Never raises
    — a lookup failure here must not itself cause a false alert."""
    repo, _, tag = image.partition(":")
    tag = tag or "latest"
    if "/" not in repo:
        return None
    try:
        with urllib.request.urlopen(
            f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}", timeout=10
        ) as resp:
            data = json.loads(resp.read())
            images = data.get("images") or []
            return images[0].get("digest") if images else None
    except Exception:
        return None


def load_state() -> dict:
    default = {"consecutive_failures": 0, "alerted": False, "last_error_code": ""}
    try:
        result = subprocess.run(
            ["gh", "variable", "get", STATE_VAR],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0 or not result.stdout.strip():
            print("No prior health-check state found — starting fresh.")
            return default
        return {**default, **json.loads(result.stdout)}
    except Exception as exc:
        print(f"Could not read prior state ({exc}); starting fresh.", file=sys.stderr)
        return default


def save_state(state: dict) -> None:
    try:
        subprocess.run(
            ["gh", "variable", "set", STATE_VAR, "--body", json.dumps(state)],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except Exception as exc:
        print(f"Could not persist state ({exc}) — next run may re-alert.", file=sys.stderr)


def send_telegram(message: str) -> None:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        print(f"Telegram not configured; would have sent:\n{message}", file=sys.stderr)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    body = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                print(f"Telegram send returned HTTP {resp.status}", file=sys.stderr)
    except Exception as exc:
        print(f"Could not send Telegram alert: {exc}", file=sys.stderr)


def main() -> int:
    if not PIPECAT_TOKEN:
        print("PIPECAT_TOKEN not set — cannot check agent health.", file=sys.stderr)
        return 1

    org = resolve_org()
    if not org:
        print("Could not resolve Pipecat organization.", file=sys.stderr)
        return 1

    data = _get_json(
        f"https://api.pipecat.daily.co/v1/organizations/{org}/services/{AGENT_NAME}",
        PIPECAT_TOKEN,
    )
    if data is None:
        print("Could not reach Pipecat Cloud status API.", file=sys.stderr)
        return 1
    body = data.get("body", data)

    available = bool(body.get("available"))
    ready = bool(body.get("ready") or body.get("activeDeploymentReady"))
    healthy = available and ready
    errors = body.get("errors") or []
    image = body.get("deployment", {}).get("manifest", {}).get("spec", {}).get("image", "")

    state = load_state()

    if healthy and not errors:
        if state["consecutive_failures"] or state["alerted"]:
            print("Agent recovered.")
            send_telegram(f"\U0001F7E2 Muhawir agent '{AGENT_NAME}' is healthy again.")
        state = {"consecutive_failures": 0, "alerted": False, "last_error_code": ""}

        # Only checked once healthy — no point flagging a stale image on a
        # deployment that's already unhealthy for a more important reason.
        expected_digest = fetch_docker_hub_digest(EXPECTED_IMAGE)
        if expected_digest and expected_digest not in image:
            send_telegram(
                f"\U0001F7E1 Muhawir agent '{AGENT_NAME}' is healthy but running an "
                f"unexpected image.\nRunning: {image}\nExpected: {EXPECTED_IMAGE}"
            )

        save_state(state)
        print(f"Healthy. available={available} ready={ready}")
        return 0

    first = errors[0] if errors else {}
    error_code = first.get("code", "")
    state["consecutive_failures"] += 1
    print(
        f"Unhealthy (check {state['consecutive_failures']}/{FAILURE_THRESHOLD}): "
        f"available={available} ready={ready} errors={[e.get('code') for e in errors]}"
    )

    if state["consecutive_failures"] >= FAILURE_THRESHOLD and not state["alerted"]:
        detail = f"\n{error_code}: {first.get('message', '')}" if first else ""
        send_telegram(
            f"\U0001F534 Muhawir agent '{AGENT_NAME}' has been unhealthy for "
            f"{state['consecutive_failures']} consecutive checks "
            f"(available={available} ready={ready}).{detail}\n"
            f"Check: pipecat cloud agent status {AGENT_NAME}"
        )
        state["alerted"] = True
        print("Alert sent.")
    elif state["alerted"]:
        print("Still unhealthy — already alerted for this outage, not repeating.")

    state["last_error_code"] = error_code
    save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
