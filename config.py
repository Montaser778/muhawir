"""Runtime configuration, driven entirely by environment variables."""

import os
from dataclasses import dataclass, field, replace

from pipecat.transcriptions.language import Language


def _optional_float(name: str) -> float | None:
    """Return None when the variable is unset, so a default can be chosen later."""
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


@dataclass
class Settings:
    # --- Credentials -------------------------------------------------
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    cartesia_api_key: str = field(default_factory=lambda: os.getenv("CARTESIA_API_KEY", ""))

    # --- Models ------------------------------------------------------
    # Conversation model: keep it small and fast. Latency beats eloquence
    # in the voice loop; the heavy reasoning happens off-loop in the scorer.
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    # Scoring model runs out-of-band, so it can be slower and stronger.
    scorer_model: str = field(
        default_factory=lambda: os.getenv("SCORER_MODEL", "llama-3.3-70b-versatile")
    )
    stt_model: str = field(
        default_factory=lambda: os.getenv("STT_MODEL", "whisper-large-v3-turbo")
    )
    tts_voice_id: str = field(
        default_factory=lambda: os.getenv("TTS_VOICE_ID", "")
    )
    tts_model: str = field(default_factory=lambda: os.getenv("TTS_MODEL", ""))

    # --- Interview ---------------------------------------------------
    language: str = "en"
    role: str = field(default_factory=lambda: os.getenv("INTERVIEW_ROLE", "Machine Learning Engineer"))
    seniority: str = field(default_factory=lambda: os.getenv("INTERVIEW_LEVEL", "mid"))
    max_questions: int = field(default_factory=lambda: int(os.getenv("MAX_QUESTIONS", "6")))

    # --- Turn taking -------------------------------------------------
    # stop_secs is the single most important number in this whole project.
    # Too low: the agent interrupts a candidate who is still thinking.
    # Too high: every reply feels sluggish. Tune it, don't guess it.
    vad_stop_secs: float | None = field(
        default_factory=lambda: _optional_float("VAD_STOP_SECS")
    )
    vad_start_secs: float = field(default_factory=lambda: float(os.getenv("VAD_START_SECS", "0.2")))
    vad_confidence: float = field(default_factory=lambda: float(os.getenv("VAD_CONFIDENCE", "0.7")))

    DEFAULT_STOP_SECS = 0.6

    def __post_init__(self) -> None:
        # An explicit VAD_STOP_SECS always wins; otherwise fall back to the default.
        if self.vad_stop_secs is None:
            self.vad_stop_secs = self.DEFAULT_STOP_SECS

    def stt_language(self) -> Language:
        return Language.EN

    def tts_language(self) -> Language:
        """Cartesia needs the language explicitly; it will not infer it."""
        return Language.EN

    def for_session(self, *, role: str | None = None, seniority: str | None = None) -> "Settings":
        """A per-call copy with topic/level overrides.

        `settings` is a module-level singleton read by every session. On
        Pipecat Cloud that is safe (one container per call), but mutating
        it in place would race across concurrent calls on any process that
        ever handles more than one session at once. Returning a copy via
        `dataclasses.replace` costs nothing and closes that door for good.
        """
        return replace(
            self,
            role=(role or "").strip() or self.role,
            seniority=(seniority or "").strip() or self.seniority,
        )

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("GROQ_API_KEY", self.groq_api_key),
                ("CARTESIA_API_KEY", self.cartesia_api_key),
                ("TTS_VOICE_ID", self.tts_voice_id),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill it in."
            )


settings = Settings()
