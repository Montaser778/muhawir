"""The voice interview agent.

Pipeline shape:

    mic ─▶ transport.in ─▶ VAD ─▶ STT ─▶ user ctx ─▶ LLM ─▶ TTS ─▶ transport.out
                                                    │
                                              (transcript tap)
                                                    │
                                            background scorer

The scorer hangs off the side of the pipeline, never inside it. Anything
placed inline between STT and TTS is paid for in latency on every single
turn, so only the three services that must be there are there.

Verified against pipecat-ai 1.6.0.

Runs in two places:
  * locally, over SmallWebRTC (server.py drives it)
  * on Pipecat Cloud, over Daily (the `bot` coroutine at the bottom)
"""

import os
import sys

# The Pipecat Cloud runner imports this module from outside /app, so the
# project root is not on sys.path and `import interview.*` fails with
# ModuleNotFoundError. Must run before any first-party import below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import aiohttp  # noqa: E402

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.cartesia.tts import CartesiaTTSService  # noqa: E402
from pipecat.services.groq.llm import GroqLLMService  # noqa: E402
from pipecat.services.groq.stt import GroqSTTService  # noqa: E402
from pipecat.transports.base_transport import BaseTransport, TransportParams  # noqa: E402

from config import Settings, settings  # noqa: E402
from prompts import (  # noqa: E402
    cannot_continue_line,
    interviewer_system_prompt,
    opening_brief,
    opening_instruction,
)
from report import Session, build_report, save  # noqa: E402
from rubric import Evaluator, looks_like_clarification  # noqa: E402
from store import store  # noqa: E402

log = logging.getLogger(__name__)


def _clean_env(name: str, *, redact: bool = False) -> str:
    """Read an env var and strip the debris that secret managers pick up.

    `pipecat cloud secrets set NAME = value` (with spaces around the equals)
    stores the literal string "= value". The leading "= " then survives
    rstrip('/') and produces a URL that aiohttp rejects outright with
    InvalidUrlClientError before any connection is attempted — which looks
    exactly like a network failure in the logs but is not one. Same story
    for values pasted with surrounding quotes or a trailing newline.
    """
    raw = os.getenv(name, "")
    value = raw.strip().strip('"').strip("'").strip()
    if value.startswith("="):
        value = value.lstrip("=").strip()

    if raw and raw != value:
        if redact:
            log.warning("%s contained stray characters; cleaned before use.", name)
        else:
            log.warning("%s cleaned: %r -> %r", name, raw, value)
    return value


# Where the finished report goes. The browser polls the front-end server, not
# this container — which is a different machine that stops as soon as the call
# ends — so the report has to be pushed there rather than read from here.
REPORT_SINK_URL = _clean_env("REPORT_SINK_URL")
REPORT_INGEST_TOKEN = _clean_env("REPORT_INGEST_TOKEN", redact=True)

# This variable has silently pointed to the wrong place three times now
# (once malformed with a stray "=", twice pointing somewhere unexpected).
# _clean_env already warns when it had to repair the value; this states
# the final, resolved destination unconditionally, once, at startup — the
# question "where are reports actually going right now" should never
# require reading source code to answer.
log.info(
    "Reports will be pushed to: %s",
    REPORT_SINK_URL or "(unset — reports will be written locally only, never pushed)",
)


def _sink_config_problem() -> str | None:
    """Return a human-readable reason the report cannot be delivered, or None.

    Checked at session start rather than only at push time: an unusable sink
    discovered after a full interview means the transcript is already gone.
    Better to see it in the logs the moment the candidate connects.
    """
    if not REPORT_SINK_URL:
        return "REPORT_SINK_URL is unset"
    if not REPORT_SINK_URL.startswith(("http://", "https://")):
        return f"REPORT_SINK_URL is not a valid URL: {REPORT_SINK_URL!r}"
    if not REPORT_INGEST_TOKEN:
        return "REPORT_INGEST_TOKEN is unset"
    return None


class TranscriptTap(FrameProcessor):
    """Passive observer: pairs questions with answers and measures latency.

    It forwards every frame untouched. If you find yourself wanting to
    block or transform here, put it in its own processor instead — this
    one must stay cheap.

    Answers are flushed when the bot starts its *next* turn (a new
    TTSTextFrame), not on UserStoppedSpeakingFrame. STT round-trips (Groq
    Whisper) routinely take several seconds — well longer than the gap
    between VAD silence and the next VAD pause — so a stopped-speaking
    flush regularly fires before the transcription it's supposed to catch
    has even arrived, silently dropping the turn.

    The answer text itself doesn't arrive here directly: LLMContextAggregatorPair's
    user aggregator consumes TranscriptionFrame to build LLM context and never
    forwards it downstream, so this tap (which sits after the LLM and TTS)
    would never see it. AnswerProbe, planted upstream of that aggregator,
    hands the text over via note_transcription() before it gets swallowed.

    The opening brief (spoken via a standalone TTSSpeakFrame before the
    interview itself begins) produces TTSTextFrame chunks exactly like any
    real question does, and nothing else in this class distinguishes "the
    fixed brief" from "a real question" — both would silently concatenate
    into the same _question_buffer since there is no user turn between them
    to trigger a flush. brief_done is the fix: run_session waits on it
    before ever asking a real question, and the first BotStoppedSpeakingFrame
    (guaranteed by that ordering to be the brief's own, and only the
    brief's) discards whatever accumulated during it.
    """

    def __init__(self, session: Session, evaluator: Evaluator):
        super().__init__()
        self._session = session
        self._evaluator = evaluator
        self._question_buffer: list[str] = []
        self._answer_buffer: list[str] = []
        self._user_stopped_at: float | None = None
        self._interview_started = False
        self.brief_done = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame):
            # A new bot turn starting means the previous answer is complete.
            self._flush_answer()
            self._question_buffer.append(frame.text)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_stopped_at = time.perf_counter()

        elif isinstance(frame, BotStartedSpeakingFrame):
            # End-to-end perceived latency: silence detected → first audio out.
            if self._user_stopped_at is not None:
                elapsed_ms = (time.perf_counter() - self._user_stopped_at) * 1000
                self._session.record_latency(elapsed_ms)
                log.info("Response latency: %.0f ms", elapsed_ms)
                self._user_stopped_at = None

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if not self._interview_started:
                self._interview_started = True
                # Discard whatever the brief's own TTSTextFrame chunks left
                # in here — none of it is a real question.
                self._question_buffer.clear()
                self.brief_done.set()

        await self.push_frame(frame, direction)

    def note_transcription(self, text: str) -> None:
        """Fed by AnswerProbe with text from a TranscriptionFrame it saw
        upstream of the LLM context aggregator."""
        text = text.strip()
        if not text:
            return
        # A new user utterance means the previous bot turn is complete.
        if self._question_buffer:
            self._session.record_question(" ".join(self._question_buffer))
            self._question_buffer.clear()
        if looks_like_clarification(text):
            # "Can you repeat that?" is not an answer. Leave it out of the
            # answer buffer entirely: _last_question stays set on the
            # Session, so whatever the candidate says once the bot has
            # repeated/explained still pairs with the original question.
            # The scorer's own is_clarification flag (rubric.py) is the
            # safety net for phrasings this regex misses.
            return
        self._answer_buffer.append(text)

    def flush(self) -> None:
        """Catch a final answer left buffered if the call ends mid-turn."""
        self._flush_answer()

    def _flush_answer(self) -> None:
        if not self._answer_buffer:
            return
        answer = " ".join(self._answer_buffer).strip()
        self._answer_buffer.clear()
        pair = self._session.record_answer(answer)
        if pair:
            self._evaluator.schedule(*pair)


class AnswerProbe(FrameProcessor):
    """Forwards every frame untouched; hands TranscriptionFrame text to the
    tap before LLMContextAggregatorPair's user aggregator consumes it.
    """

    def __init__(self, tap: TranscriptTap):
        super().__init__()
        self._tap = tap

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            self._tap.note_transcription(frame.text)
        await self.push_frame(frame, direction)


def build_pipeline(transport: BaseTransport, tap: TranscriptTap, call_settings: Settings):
    stt = GroqSTTService(
        api_key=call_settings.groq_api_key,
        model=call_settings.stt_model,
        language=call_settings.stt_language(),
    )

    llm = GroqLLMService(
        api_key=call_settings.groq_api_key,
        model=call_settings.llm_model,
    )

    tts_kwargs = {}
    if call_settings.tts_model:
        tts_kwargs["model"] = call_settings.tts_model

    tts = CartesiaTTSService(
        api_key=call_settings.cartesia_api_key,
        voice_id=call_settings.tts_voice_id,
        params=CartesiaTTSService.InputParams(language=call_settings.tts_language()),
        **tts_kwargs,
    )

    context = LLMContext(
        messages=[{"role": "system", "content": interviewer_system_prompt(call_settings)}]
    )
    aggregators = LLMContextAggregatorPair(context)

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=call_settings.vad_confidence,
                start_secs=call_settings.vad_start_secs,
                stop_secs=call_settings.vad_stop_secs,
            )
        )
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            AnswerProbe(tap),
            aggregators.user(),
            llm,
            tts,
            tap,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    # Returned alongside the pipeline so run_session can attach on_error
    # handlers directly to each backing service — see the "never silence"
    # handling there.
    return pipeline, context, (stt, llm, tts)


async def run_session(
    transport: BaseTransport, session_id: str, call_settings: Settings | None = None
) -> None:
    """One interview call, transport-agnostic.

    Takes a ready-made transport so the same body serves both SmallWebRTC
    locally and Daily on Pipecat Cloud.

    call_settings, when given, is a per-call copy (see Settings.for_session)
    carrying the topic/level chosen for this candidate. Falls back to the
    module-level settings singleton — today only bot()'s Pipecat Cloud path
    builds a per-call copy; run_bot's local path still uses the shared
    defaults, unchanged from before this existed.
    """
    s = call_settings or settings
    s.validate()

    problem = _sink_config_problem()
    if problem:
        # Loud, but not fatal: the interview itself still works, and a live
        # candidate should never be dropped over a misconfigured sink.
        log.error("Report delivery is misconfigured (%s) — the report will "
                  "be written locally but not pushed.", problem)

    session = Session(
        session_id=session_id,
        role=s.role,
        language=s.language,
    )
    store.create(session_id, s.role, s.language)
    evaluator = Evaluator(s.groq_api_key, s.scorer_model)
    tap = TranscriptTap(session, evaluator)

    pipeline, context, (stt, llm, tts) = build_pipeline(transport, tap, s)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        # Interruption support is what makes the agent feel alive. Without
        # it the candidate has to wait politely for a machine to finish.
        idle_timeout_secs=180,
    )

    # "Can't continue" state, shared across whichever service errors first.
    # Two strikes, not one: a single transient blip shouldn't end a call
    # that could otherwise finish normally — the same reasoning already
    # applied to the health-check's own FAILURE_THRESHOLD.
    failure_count = 0
    bailed_out = False

    async def _on_service_error(processor, error_frame):
        nonlocal failure_count, bailed_out
        failure_count += 1
        log.error(
            "%s error (%d/2): %s", processor, failure_count, error_frame.error
        )
        if bailed_out:
            return

        # First strike: recover, don't just return.
        #
        # This is what produced the "goes silent after question two" bug.
        # Groq returns 429 (rate limit) once the accumulated context grows
        # past the per-minute token allowance — usually on the third turn.
        # The failed turn emits no text and no audio, and the LLM is only
        # ever invoked again when the candidate speaks. So returning here
        # left the candidate waiting for a question that would never come,
        # and the second strike could never arrive either: dead air until
        # idle_timeout_secs (180s) killed the call. That matches the
        # observed session durations of 197s and 230s exactly.
        #
        # A 429 means "wait", not "broken". Say something so the candidate
        # knows the line is alive, pause long enough for the rate-limit
        # window to roll over, then re-run the turn.
        if failure_count == 1:
            await task.queue_frames([
                TTSSpeakFrame("One moment.", append_to_context=False),
            ])
            await asyncio.sleep(3)
            await task.queue_frames([LLMRunFrame()])
            return

        bailed_out = True
        log.error("Interview cannot continue — ending the call and finalising.")
        # Straight to TTS, not the LLM: the LLM is exactly what's likely
        # broken. EndFrame right behind it drives the same shutdown path
        # _on_disconnected already uses, so tap.flush()/_finalise() below
        # run exactly as they would for a normal end of call.
        await task.queue_frames([
            TTSSpeakFrame(cannot_continue_line(s), append_to_context=False),
            EndFrame(),
        ])

    for service in (stt, llm, tts):
        service.event_handler("on_error")(_on_service_error)

    @transport.event_handler("on_first_participant_joined")
    async def _on_connected(_transport, _participant):
        log.info("Candidate connected — session %s", session.session_id)
        # append_to_context=False: this is a fixed script, not something the
        # LLM said, and must never enter its memory of the conversation —
        # left at the (default) True, it silently becomes part of the
        # "assistant" turn the LLM sees on every subsequent call, which is
        # what previously corrupted the context enough to derail the rest
        # of the interview.
        await task.queue_frames([
            TTSSpeakFrame(opening_brief(s), append_to_context=False),
        ])
        # Wait for the brief to actually finish before asking the real first
        # question. Queuing both together used to let them run together as
        # one continuous utterance with no turn boundary between them —
        # which is how the brief's own text ended up folded into question
        # one in the report. brief_done is set by TranscriptTap on the
        # first BotStoppedSpeakingFrame it sees, which — because nothing
        # else has been queued yet — can only be the brief's.
        try:
            await asyncio.wait_for(tap.brief_done.wait(), timeout=60)
        except asyncio.TimeoutError:
            log.warning(
                "Opening brief did not report completion within 60s; "
                "asking the first question anyway rather than hanging."
            )
        context.add_message({"role": "user", "content": opening_instruction(s)})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        log.info("Candidate disconnected — session %s", session.session_id)
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception as exc:  # noqa: BLE001
        store.set_status(session.session_id, "error", str(exc))
        raise
    finally:
        tap.flush()
        await _finalise(session, evaluator)


async def run_bot(webrtc_connection) -> None:
    """Local entry point — kept so server.py keeps working unchanged."""
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

    # pc_id is handed back to the browser in the SDP answer, so using it as
    # the session id means the client can poll for its own report with no
    # extra plumbing.
    session_id = getattr(webrtc_connection, "pc_id", None) or uuid.uuid4().hex[:12]

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    await run_session(transport, session_id)


async def _push_report(session_id: str, payload: dict) -> None:
    """Hand the finished report to the front-end server.

    Failure here is logged and swallowed: a report that cannot be delivered
    is not worth crashing a session that has already finished successfully.
    One retry, because this runs during container shutdown and a single
    dropped connection should not cost the whole report.
    """
    problem = _sink_config_problem()
    if problem:
        log.warning("Report not pushed for session %s: %s", session_id, problem)
        return

    url = f"{REPORT_SINK_URL.rstrip('/')}/api/report/{session_id}"
    log.info("Pushing report for session %s to %s", session_id, url)

    for attempt in (1, 2):
        try:
            async with aiohttp.ClientSession() as http:
                async with http.post(
                    url,
                    headers={"X-Ingest-Token": REPORT_INGEST_TOKEN},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status >= 400:
                        # The body is where the server explains itself; without
                        # it a 4xx is indistinguishable from any other 4xx.
                        body = (await resp.text())[:500]
                        log.error(
                            "Report push returned %s for session %s: %s",
                            resp.status, session_id, body,
                        )
                    else:
                        log.info(
                            "Report pushed for session %s (HTTP %s)",
                            session_id, resp.status,
                        )
                    return
        except aiohttp.InvalidURL:
            # Malformed URL will never succeed; retrying only hides the cause.
            log.error(
                "Report sink URL is malformed: %r — check REPORT_SINK_URL "
                "for stray characters such as a leading '='.", url,
            )
            return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == 1:
                log.warning(
                    "Report push failed for session %s; retrying once.", session_id
                )
                await asyncio.sleep(1)
                continue
            log.exception("Could not push report for session %s", session_id)
        except Exception:
            log.exception("Could not push report for session %s", session_id)
            return


async def _finalise(session: Session, evaluator: Evaluator) -> None:
    """Wait for in-flight scoring, then write the report."""
    store.set_status(session.session_id, "scoring")
    # Pushed to the front-end too, not just set on this container's own
    # local store — this runs after runner.run(task) has already returned
    # (the candidate has already hung up), so it's off the live audio path.
    # Without this push, /report/{session_id} could only ever see "running"
    # (set by server.py's /api/connect at call start) or the final
    # ready/empty state — "scoring" would never be a state a shared link
    # could actually show.
    await _push_report(
        session.session_id,
        {"status": "scoring", "role": session.role, "language": session.language},
    )
    try:
        evaluations = await asyncio.wait_for(evaluator.drain(), timeout=30)
    except asyncio.TimeoutError:
        log.warning("Scoring did not finish in time; writing partial report.")
        evaluations = evaluator.results

    # Clarification requests the regex gate in TranscriptTap.note_transcription
    # missed still got scored above (no extra round trip — same call), but
    # they were never real answers. Drop them before they touch the report,
    # so a "can you repeat that" never costs the candidate points.
    clarifications = sum(1 for e in evaluations if e.is_clarification)
    if clarifications:
        log.info(
            "Excluding %d clarification exchange(s) from session %s's report.",
            clarifications, session.session_id,
        )
        evaluations = [e for e in evaluations if not e.is_clarification]

    # A turn whose scoring call itself failed (e.g. the same Groq 429 that
    # can hit the interview LLM) carries no real judgement — TurnEvaluation
    # .average defaults to 0.0 when scores is empty, which without this
    # filter becomes a literal 0/5 baked into the candidate's overall score
    # for a question that was never actually graded. Drop it, same as a
    # clarification: "never scored" must never render as "scored zero".
    errored = sum(1 for e in evaluations if e.error)
    if errored:
        log.info(
            "Excluding %d turn(s) that failed to score (system error, not "
            "a judged answer) from session %s's report.",
            errored, session.session_id,
        )
        evaluations = [e for e in evaluations if not e.error]

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


# ── Pipecat Cloud entry point ────────────────────────────────────────────
# The platform imports this module and awaits bot(runner_args). Daily is the
# transport there; the webrtc branch keeps `pipecat run bot.py` usable locally.


async def bot(runner_args) -> None:
    from pipecat.runner.utils import create_transport
    from pipecat.transports.daily.transport import DailyParams

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    session_id = (
        getattr(runner_args, "session_id", None)
        or getattr(runner_args, "room_url", "").rsplit("/", 1)[-1]
        or uuid.uuid4().hex[:12]
    )

    # Topic + level chosen client-side before the call starts, carried here
    # via runner_args.body — server.py's /api/connect forwards whatever the
    # candidate submitted as the Pipecat /start payload's "body". Falls back
    # to the env-var defaults in Settings when the client sent nothing.
    body = getattr(runner_args, "body", None)
    body = body if isinstance(body, dict) else {}
    call_settings = settings.for_session(
        role=body.get("role"),
        seniority=body.get("seniority"),
    )

    await run_session(transport, session_id, call_settings)