"""Session transcript tracking and the final report.

The report is the artifact the candidate keeps. It is also the thing that
makes the product feel finished rather than experimental.
"""

import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rubric import DIMENSIONS, TurnEvaluation


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
        # Derived from evaluations, not len(session.turns): the caller
        # (bot._finalise) filters out clarification-flagged turns and any
        # that never finished scoring before evaluations reaches here, so
        # this is the count of turns actually reflected in the scores below.
        "questions_answered": len(evaluations),
        "overall_score": overall,
        "dimension_averages": averages,
        "weakest_dimension": min(averages, key=averages.get) if averages else None,
        "strongest_dimension": max(averages, key=averages.get) if averages else None,
        "p50_response_latency_ms": session.p50_latency,
        "turns": [ev.to_dict() for ev in evaluations],
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Interview report â€” {report['role']}",
        "",
        f"Session `{report['session_id']}` آ· {report['questions_answered']} questions "
        f"آ· {report['duration_seconds']}s آ· median response latency "
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


def render_html(report: dict) -> str:
    """Standalone, shareable report page — served at /report/{session_id}
    (server.py). Same report dict build_report() already produces; this
    only changes presentation, never scoring.

    Print/PDF is the browser's native Print dialog against the @media
    print block below — no PDF library, nothing new to keep small.
    """
    esc = html.escape

    dim_rows = "\n".join(
        f'''<div class="dim{" weak" if name == report.get("weakest_dimension") else ""}">
          <div class="dim-row"><span class="dim-name">{esc(name.replace("_", " ").title())}</span>
          <span class="dim-score">{value}</span></div>
          <div class="track"><div class="fill" style="width:{max(0, min(100, (value / 5) * 100))}%"></div></div>
        </div>'''
        for name, value in report.get("dimension_averages", {}).items()
    )

    turn_blocks = "\n".join(
        f'''<div class="turn">
          <p class="turn-q">{index}. {esc(turn.get("question", ""))}</p>
          <p class="turn-a">{esc(turn.get("answer", ""))}</p>
          <p class="turn-note"><b>Strength</b> {esc(turn.get("strength", ""))}</p>
          <p class="turn-note"><b>Improve</b> {esc(turn.get("improvement", ""))}</p>
        </div>'''
        for index, turn in enumerate(report.get("turns", []), start=1)
    )

    focus = ""
    if report.get("weakest_dimension"):
        focus = (
            f'<p class="focus">Focus area: '
            f'<b>{esc(report["weakest_dimension"].replace("_", " "))}</b></p>'
        )

    role = esc(report.get("role", "Interview"))
    session_id = esc(report.get("session_id", ""))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Interview report — {role}</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet" />
<style>
  :root {{
    --bg: #0a0b0d; --panel: #131519; --line: #23272e;
    --amber: #ffb627; --cyan: #4dd8e6; --text: #e6e8eb; --muted: #6b7280;
    --display: "Space Grotesk", system-ui, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: var(--display); display: flex; justify-content: center;
    padding: 24px;
  }}
  main {{
    width: min(720px, 100%); background: var(--panel);
    border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
  }}
  .head {{ padding: 36px 32px 20px; border-bottom: 1px solid var(--line); }}
  .eyebrow {{
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--amber); margin: 0 0 12px;
  }}
  h1 {{ font-size: clamp(24px, 6vw, 34px); margin: 0 0 10px; line-height: 1.1; }}
  .meta {{ font-family: var(--mono); font-size: 12px; color: var(--muted); margin: 0; }}
  .score {{ display: flex; align-items: baseline; gap: 8px; margin: 18px 0 0; }}
  .score-value {{ font-family: var(--mono); font-size: 44px; font-weight: 700; color: var(--amber); }}
  .score-max {{ font-family: var(--mono); font-size: 14px; color: var(--muted); }}
  .focus {{ font-family: var(--mono); font-size: 13px; color: var(--muted); margin: 10px 0 0; }}
  .focus b {{ color: var(--amber); }}
  .body {{ padding: 24px 32px; }}
  .dim {{ margin-bottom: 14px; }}
  .dim-row {{ display: flex; justify-content: space-between; font-family: var(--mono); font-size: 12px; margin-bottom: 6px; }}
  .dim-score {{ color: var(--muted); }}
  .track {{ height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; }}
  .fill {{ height: 100%; background: var(--cyan); border-radius: 2px; }}
  .dim.weak .fill {{ background: var(--amber); }}
  .turn {{ border-top: 1px solid var(--line); padding: 18px 0; }}
  .turn-q {{ font-size: 15px; font-weight: 500; margin: 0 0 8px; }}
  .turn-a {{ margin: 0 0 10px; padding-left: 12px; border-left: 2px solid var(--line); color: var(--muted); font-size: 14px; line-height: 1.6; }}
  .turn-note {{ margin: 0 0 4px; font-size: 13px; }}
  .turn-note b {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }}
  .actions {{ padding: 20px 32px 28px; border-top: 1px solid var(--line); display: flex; gap: 12px; flex-wrap: wrap; }}
  .actions a, .actions button {{
    font-family: var(--mono); font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--amber); text-decoration: none; border: 1px solid var(--amber); border-radius: 8px;
    padding: 10px 18px; background: transparent; cursor: pointer;
  }}
  @media (max-width: 480px) {{
    .head, .body, .actions {{ padding-left: 18px; padding-right: 18px; }}
  }}
  @media print {{
    body {{ background: #fff; color: #111; padding: 0; }}
    main {{ width: 100%; border: none; border-radius: 0; }}
    .actions {{ display: none; }}
    .eyebrow, .score-value {{ color: #111; }}
    .track {{ background: #ddd; }}
    .fill {{ background: #333; }}
    .turn-a {{ border-left-color: #ccc; color: #333; }}
  }}
</style>
</head>
<body>
<main>
  <div class="head">
    <p class="eyebrow">Interview report</p>
    <h1>{role}</h1>
    <p class="meta">Session {session_id} · {report.get("questions_answered", 0)} questions · {report.get("duration_seconds", 0)}s</p>
    <div class="score">
      <span class="score-value">{report.get("overall_score", "—")}</span>
      <span class="score-max">/ 5</span>
    </div>
    {focus}
  </div>
  <div class="body">
    {dim_rows}
    {turn_blocks}
  </div>
  <div class="actions">
    <button onclick="window.print()">Print / Save as PDF</button>
    <a href="/api/report/{session_id}/download">Download markdown</a>
  </div>
</main>
</body>
</html>"""


def save(report: dict, directory: str = "reports") -> tuple[Path, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report['session_id']}.json"
    md_path = out / f"{report['session_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
