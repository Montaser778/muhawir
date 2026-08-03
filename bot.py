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

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    EndFrame,
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
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

from config import settings
from prompts import interviewer_system_prompt, opening_instruction
from report import Session, build_report, save
from rubric import Evaluator
from store import store

log = logging.getLogger(__name__)


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
    """

    def __init__(self, session: Session, evaluator: Evaluator):
        super().__init__()
        self._session = session
        self._evaluator = evaluator
        self._question_buffer: list[str] = []
        self._answer_buffer: list[str] = []
        self._user_stopped_at: float | None = None

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


def build_pipeline(transport: BaseTransport, tap: TranscriptTap):
    stt = GroqSTTService(
        api_key=settings.groq_api_key,
        model=settings.stt_model,
        language=settings.stt_language(),
    )

    llm = GroqLLMService(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
    )

    tts_kwargs = {}
    if settings.tts_model:
        tts_kwargs["model"] = settings.tts_model

    tts = CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        voice_id=settings.tts_voice_id,
        params=CartesiaTTSService.InputParams(language=settings.tts_language()),
        **tts_kwargs,
    )

    context = LLMContext(
        messages=[{"role": "system", "content": interviewer_system_prompt(settings)}]
    )
    aggregators = LLMContextAggregatorPair(context)

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=settings.vad_confidence,
                start_secs=settings.vad_start_secs,
                stop_secs=settings.vad_stop_secs,
            )
        )
    )

    return (
        Pipeline(
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
        ),
        context,
    )


async def run_session(transport: BaseTransport, session_id: str) -> None:
    """One interview call, transport-agnostic.

    Takes a ready-made transport so the same body serves both SmallWebRTC
    locally and Daily on Pipecat Cloud.
    """
    settings.validate()
    log.info(
        "KEYCHECK groq_len=%d groq_ascii=%s cartesia_len=%d cartesia_ascii=%s voice=%r",
        len(settings.groq_api_key),
        settings.groq_api_key.isascii(),
        len(settings.cartesia_api_key),
        settings.cartesia_api_key.isascii(),
        settings.tts_voice_id,
    )

    session = Session(
        session_id=session_id,
        role=settings.role,
        language=settings.language,
    )
    store.create(session_id, settings.role, settings.language)
    evaluator = Evaluator(settings.groq_api_key, settings.scorer_model)
    tap = TranscriptTap(session, evaluator)

    pipeline, context = build_pipeline(transport, tap)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        # Interruption support is what makes the agent feel alive. Without
        # it the candidate has to wait politely for a machine to finish.
        idle_timeout_secs=180,
    )

    @transport.event_handler("on_first_participant_joined")
    async def _on_connected(_transport, _participant):
        log.info("Candidate connected — session %s", session.session_id)
        context.add_message({"role": "user", "content": opening_instruction(settings)})
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

    await run_session(transport, session_id)