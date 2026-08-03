"""Session transcript tracking and the final report.

The report is the artifact the candidate keeps. It is also the thing that
makes the product feel finished rather than experimental.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .rubric import DIMENSIONS, TurnEvaluation


@dataclass
class Session:
    """Tracks the question/answer pairing across the call.

    The pairing matters: the evaluator needs to know which question an
    answer belongs to, and in a voice call those arrive as separate,
    interleaved events.
    """

    session_id: str
    role: str
    language: str
    started_at: float = field(default_factory=time.time)
    turns: list[tuple[str, str]] = field(default_factory=list)
    _last_question: str | None = None
    latencies_ms: list[float] = field(default_factory=list)

    def record_question(self, text: str) -> None:
        text = text.strip()
        if text:
            self._last_question = text

    def record_answer(self, text: str) -> tuple[str, str] | None:
        """Pair an answer with the question that preceded it."""
        text = text.strip()
        if not text or not self._last_question:
            return None
        pair = (self._last_question, text)
        self.turns.append(pair)
        self._last_question = None
        return pair

    def record_latency(self, ms: float) -> None:
        self.latencies_ms.append(ms)

    @property
    def p50_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return round(ordered[len(ordered) // 2], 1)


def build_report(session: Session, evaluations: list[TurnEvaluation]) -> dict:
    per_dimension: dict[str, list[int]] = {k: [] for k in DIMENSIONS}
    for ev in evaluations:
        for name, payload in ev.scores.items():
            if name in per_dimension and isinstance(payload.get("score"), int):
                per_dimension[name].append(payload["score"])

    averages = {
        name: round(sum(values) / len(values), 2)
        for name, values in per_dimension.items()
        if values
    }
    overall = round(sum(averages.values()) / len(averages), 2) if averages else 0.0

    return {
        "session_id": session.session_id,
        "role": session.role,
        "language": session.language,
        "duration_seconds": round(time.time() - session.started_at, 1),
        "questions_answered": len(session.turns),
        "overall_score": overall,
        "dimension_averages": averages,
        "weakest_dimension": min(averages, key=averages.get) if averages else None,
        "strongest_dimension": max(averages, key=averages.get) if averages else None,
        "p50_response_latency_ms": session.p50_latency,
        "turns": [ev.to_dict() for ev in evaluations],
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Interview report — {report['role']}",
        "",
        f"Session `{report['session_id']}` · {report['questions_answered']} questions "
        f"· {report['duration_seconds']}s · median response latency "
        f"{report['p50_response_latency_ms']} ms",
        "",
        f"**Overall: {report['overall_score']} / 5**",
        "",
        "## Scores by dimension",
        "",
        "| Dimension | Average |",
        "| --- | --- |",
    ]
    for name, value in report["dimension_averages"].items():
        lines.append(f"| {name.replace('_', ' ').title()} | {value} |")

    if report["weakest_dimension"]:
        lines += ["", f"Focus area: **{report['weakest_dimension'].replace('_', ' ')}**", ""]

    lines += ["## Question by question", ""]
    for index, turn in enumerate(report["turns"], start=1):
        lines += [
            f"### {index}. {turn['question']}",
            "",
            f"> {turn['answer']}",
            "",
            f"- Score: **{turn['average']} / 5**",
            f"- Strength: {turn['strength']}",
            f"- Improve: {turn['improvement']}",
            "",
        ]
    return "\n".join(lines)


def save(report: dict, directory: str = "reports") -> tuple[Path, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report['session_id']}.json"
    md_path = out / f"{report['session_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
