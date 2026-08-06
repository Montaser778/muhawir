#!/usr/bin/env python3
"""End-to-end check for the report pipeline: bot.py's real _finalise()
against a real, locally-running server.py.

This exists because the previous version of this check (a hand-written
curl script asserting each state independently) proved insufficient: it
tested that server.py's /api/report endpoint behaves correctly when POSTed
to directly, but never once called the actual functions bot.py uses to
produce those POSTs. A real session failed to show a report while that
check reported a pass, because the check was never exercising bot.py's
code at all — only a simulation of it.

This script instead imports bot.py for real and calls its actual
_finalise() (and therefore its actual _push_report(), twice, exactly as a
real call does: once for "scoring", once for the final state) against a
server.py this script starts itself. If bot.py and server.py agree with
each other, this passes. If they don't — wrong field name, wrong URL
construction, a silently-swallowed exception — this fails loudly, because
it's the same code path a real call uses, not a paraphrase of it.

Run: python scripts/check_report_pipeline.py
Exit 0 = pipeline verified end to end. Exit 1 = it isn't, with the reason.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 8099
BASE_URL = f"http://localhost:{PORT}"
TEST_SESSION_ID = "checklist-pipeline-test"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _cleanup_test_artifacts() -> None:
    for ext in ("json", "md"):
        f = REPORTS_DIR / f"{TEST_SESSION_ID}.{ext}"
        if f.exists():
            f.unlink()


def main() -> int:
    project_root = Path(__file__).parent.parent
    env = os.environ.copy()
    env["PORT"] = str(PORT)

    print(f"Starting server.py on :{PORT} ...")
    server = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=project_root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    try:
        # Wait for the server to actually accept connections rather than
        # sleeping a guessed duration.
        for _ in range(50):
            try:
                urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        else:
            print("FAIL: server.py did not come up in time.", file=sys.stderr)
            return 1

        # Point THIS process's bot module at the server we just started —
        # exactly the role REPORT_SINK_URL plays for a real deployed bot
        # container talking to the real front-end.
        env_before = os.environ.get("REPORT_SINK_URL")
        os.environ["REPORT_SINK_URL"] = BASE_URL
        sys.path.insert(0, str(project_root))
        from dotenv import load_dotenv
        load_dotenv()
        os.environ["REPORT_SINK_URL"] = BASE_URL  # load_dotenv must not undo the override

        import bot as botmod
        import importlib
        importlib.reload(botmod)  # re-read REPORT_SINK_URL with the override in effect

        from report import Session
        from rubric import TurnEvaluation

        session = Session(
            session_id=TEST_SESSION_ID, role="Backend Engineer", language="en"
        )
        session.turns = [("What is a hash map?", "A key-value data structure.")]

        class FakeEvaluator:
            results = [
                TurnEvaluation(
                    question="What is a hash map?",
                    answer="A key-value data structure.",
                    scores={"relevance": {"score": 4, "why": "addresses the question"}},
                    is_clarification=False,
                )
            ]

            async def drain(self, timeout: float = 20.0):
                return self.results

        print("Running bot._finalise() for real against the local server...")
        asyncio.run(botmod._finalise(session, FakeEvaluator()))

        status, body = _get(f"{BASE_URL}/api/report/{TEST_SESSION_ID}")
        if status != 200:
            print(f"FAIL: GET /api/report/{TEST_SESSION_ID} returned {status}: {body}", file=sys.stderr)
            return 1
        if body.get("status") != "ready":
            print(f"FAIL: expected status 'ready', got {body.get('status')!r}. Full body: {body}", file=sys.stderr)
            return 1
        report = body.get("report") or {}
        if report.get("questions_answered") != 1 or report.get("overall_score") != 4.0:
            print(f"FAIL: report content is wrong: {report}", file=sys.stderr)
            return 1

        # /report/{id} (the shareable HTML page) must also resolve, not
        # just the JSON API — this is what a candidate actually opens.
        html_status, _ = (lambda: (
            urllib.request.urlopen(f"{BASE_URL}/report/{TEST_SESSION_ID}", timeout=5).status, None
        ))()
        if html_status != 200:
            print(f"FAIL: GET /report/{TEST_SESSION_ID} (HTML) returned {html_status}", file=sys.stderr)
            return 1

        print("PASS: bot._finalise()'s real scoring+ready push sequence reaches "
              "server.py and /api/report + /report both resolve correctly.")
        return 0

    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        output = server.stdout.read() if server.stdout else ""
        if "Traceback" in output or "ERROR" in output:
            print("--- server.py output (contained errors) ---", file=sys.stderr)
            print(output, file=sys.stderr)
        if env_before is None:
            os.environ.pop("REPORT_SINK_URL", None)
        else:
            os.environ["REPORT_SINK_URL"] = env_before
        _cleanup_test_artifacts()


if __name__ == "__main__":
    raise SystemExit(main())
