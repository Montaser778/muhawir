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

Entry point is `bot(runner_args)`, which is what both the local development
runner and Pipecat Cloud call — once per session. The transport is built
from runner_args rather than constructed here, so the same file runs against
Daily (Pipecat Cloud, managed TURN) and SmallWebRTC (local) with no edits.

Verified against pipecat-ai 1.6.0.
"""

import asyncio
import logging
import time
import uuid

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EndFrame,
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from interview.config import settings
from interview.prompts import interviewer_system_prompt, opening_instruction
from interview.report import Session, build_report, save
from interview.rubric import Evaluator
from interview.store import store

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


def _transport_params():
    """Per-transport config. Daily is what Pipecat Cloud hands us; webrtc is
    the local path. Both want the same thing: audio in, audio out."""
    params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    # Daily's params class only exists when the [daily] extra is installed,
    # which is true in the deployed image but not necessarily on a dev box.
    try:
        from pipecat.transports.daily.transport import DailyParams

        params["daily"] = lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    except ImportError:
        log.debug("Daily transport not installed; webrtc only.")

    return params


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point. The runner calls this once per session.

    Pipecat Cloud and the local dev runner both land here; the only
    difference is which transport create_transport() hands back.
    """
    transport = await create_transport(runner_args, _transport_params())
    session_id = getattr(runner_args, "session_id", None) or uuid.uuid4().hex[:12]
    await run_interview(transport, session_id)


async def run_interview(transport: BaseTransport, session_id: str) -> None:
    """One interview call, from first question to written report."""
    settings.validate()

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

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
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


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()