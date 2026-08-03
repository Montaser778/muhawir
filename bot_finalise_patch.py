# ─────────────────────────────────────────────────────────────────────────
# Add near the top of bot.py, with the other imports:
# ─────────────────────────────────────────────────────────────────────────

import aiohttp  # noqa: E402

REPORT_SINK_URL = os.getenv("REPORT_SINK_URL", "")
REPORT_INGEST_TOKEN = os.getenv("REPORT_INGEST_TOKEN", "")


# ─────────────────────────────────────────────────────────────────────────
# Add this helper anywhere above _finalise:
# ─────────────────────────────────────────────────────────────────────────


async def _push_report(session_id: str, payload: dict) -> None:
    """Hand the finished report to the front-end server.

    The browser polls that server, not this container — which is a different
    machine that stops as soon as the call ends. Failure here is logged and
    swallowed: a report that cannot be delivered is not worth crashing a
    session that has already finished successfully.
    """
    if not REPORT_SINK_URL or not REPORT_INGEST_TOKEN:
        log.warning("REPORT_SINK_URL / REPORT_INGEST_TOKEN unset; report not pushed.")
        return

    url = f"{REPORT_SINK_URL.rstrip('/')}/api/report/{session_id}"
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                url,
                headers={"X-Ingest-Token": REPORT_INGEST_TOKEN},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    log.error("Report push returned %s", resp.status)
                else:
                    log.info("Report pushed for session %s", session_id)
    except Exception:
        log.exception("Could not push report for session %s", session_id)


# ─────────────────────────────────────────────────────────────────────────
# Replace the whole existing _finalise with this version. The only change is
# the three _push_report calls — the scoring logic is untouched.
# ─────────────────────────────────────────────────────────────────────────


async def _finalise(session: Session, evaluator: Evaluator) -> None:
    """Wait for in-flight scoring, then write the report."""
    store.set_status(session.session_id, "scoring")
    try:
        evaluations = await asyncio.wait_for(evaluator.drain(), timeout=30)
    except asyncio.TimeoutError:
        log.warning("Scoring did not finish in time; writing partial report.")
        evaluations = evaluator.results

    if not evaluations:
        log.info("No scored turns for session %s; skipping report.", session.session_id)
        store.set_status(session.session_id, "empty")
        await _push_report(
            session.session_id,
            {"status": "empty", "role": session.role, "language": session.language},
        )
        return

    report = build_report(session, evaluations)
    json_path, md_path = save(report)
    store.save(session.session_id, report)
    log.info(
        "Report written: %s (overall %.2f/5, p50 latency %.0f ms)",
        md_path,
        report["overall_score"],
        report["p50_response_latency_ms"],
    )
    await _push_report(session.session_id, {"status": "ready", "report": report})
