# Pre-deploy checklist

Run this — actually run it, not just read it — before every deploy to Fly
or Pipecat Cloud. It exists because a UI bug once cost a full day: the
front-end store never learned a session existed until the call was already
over, so "still scoring" and "this link is wrong" looked identical. Add a
row here every time a new behavior is built, in the same session that
builds it — a checklist that lags the code is worse than none, because it
lies.

Each row names the files it depends on. If you touched one of those files,
that row is not optional.

## 1. Topic + level reach the interviewer prompt

**Depends on:** `static/index.html` (form), `server.py` (`/api/connect`),
`bot.py` (`bot()`), `config.py` (`Settings.for_session`), `prompts.py`
(`interviewer_system_prompt`)

- [ ] Start a call with a custom topic typed into the topic field.
- [ ] Confirm the agent's first real question (after the opening brief) is
      about that topic, not the env-var default.

```bash
grep -n "topicInput\|role: topic" static/index.html   # form + wiring present
grep -n "for_session" bot.py config.py                # per-call override wired through
```

## 2. Opening brief plays as TTS, not LLM

**Depends on:** `prompts.py` (`opening_brief`), `bot.py` (`run_session`'s
`_on_connected`)

- [ ] Start a call. The first audio is the fixed brief text verbatim — not
      a paraphrase, not missing.
- [ ] Check the agent logs: `Generating TTS [Before we begin...` should
      appear with **no** preceding `GroqLLMService#0: Generating chat`
      call for that specific text (it must not have gone through the LLM).

```bash
grep -n "TTSSpeakFrame(opening_brief" bot.py
```

## 3. "Adapts to answers" means difficulty, not topic — and says so

**Depends on:** `prompts.py`

```bash
grep -n "not topic" prompts.py   # must appear in both interviewer_system_prompt and opening_brief
```

## 4. Clarification requests are never scored as answers

**Depends on:** `rubric.py` (`looks_like_clarification`, `is_clarification`
flag), `bot.py` (`TranscriptTap.note_transcription`, `_finalise` filter),
`report.py` (`build_report`)

- [ ] During a call, ask "can you repeat that?" — confirm the bot repeats
      the question and the exchange does not appear as a scored turn in
      the final report.

```bash
python -c "
from rubric import looks_like_clarification as f
assert f('Can you repeat the question?')
assert not f('I used PyTorch for this.')
print('regex gate OK')
"
grep -n "is_clarification" bot.py rubric.py report.py   # flag must flow through all three
```

## 5. Report reaches the front-end and displays at session end

**Depends on:** `bot.py` (`_finalise`, `_push_report`), `server.py`
(`ingest_report`, `get_report`, `report_page`), `static/index.html`
(`pollForReport`, `renderReport`)

This item previously read "complete a full call, eyeball the SPA" plus a
curl check that only asserted `/api/health` was up — it never once called
the actual functions that produce a real push, so it passed while a real
session failed to show a report. Fixed by calling bot.py's real
`_finalise()` (which calls the real `_push_report()`, twice, exactly as a
live call does) against a `server.py` this script starts itself:

```bash
python scripts/check_report_pipeline.py
```

Must print `PASS` and exit 0. If it doesn't, the reported failure is real
— this script exercises the identical code path a live call uses, not a
simulation of it.

**Known pitfall this will NOT catch**: if `REPORT_SINK_URL` in your shell
environment points at production (`muhawir.fly.dev`) while you're running
`bot.py` locally for manual testing, `bot.py` will happily push a real
report — to production, not to the local `server.py` you're watching. The
push can fully succeed and you will still see nothing, because you were
watching the wrong process. Before any manual local test:

```bash
echo $REPORT_SINK_URL   # must be http://localhost:<port your local server.py is on>
```

If it says `fly.dev`, unset or override it for that shell before running
`bot.py` locally.

## 6. `questions_answered` counts evaluations, not raw turns

**Depends on:** `report.py` (`build_report`)

```bash
grep -n '"questions_answered"' report.py   # must read len(evaluations), not len(session.turns)
```

## 7. The six `/report/{session_id}` states render distinctly

**Depends on:** `server.py` (`report_page`, `_report_status_page`)

None of these may show the same message. "Still scoring" and "wrong link"
looking identical is the exact bug this file exists to prevent from
recurring.

```bash
PORT=8080 python server.py &
SERVER_PID=$!
sleep 2

TOKEN=<REPORT_INGEST_TOKEN from .env>

# unknown
curl -s -o /dev/null -w "unknown:  %{http_code}\n" http://localhost:8080/report/does-not-exist

# scoring/running
curl -s -X POST http://localhost:8080/api/report/chk-scoring \
  -H "X-Ingest-Token: $TOKEN" -d '{"status":"scoring","role":"X","language":"en"}' > /dev/null
curl -s -o /dev/null -w "scoring:  %{http_code}\n" http://localhost:8080/report/chk-scoring
curl -s http://localhost:8080/report/chk-scoring | grep -q "refresh" && echo "  has auto-refresh: OK"

# empty
curl -s -X POST http://localhost:8080/api/report/chk-empty \
  -H "X-Ingest-Token: $TOKEN" -d '{"status":"empty","role":"X","language":"en"}' > /dev/null
curl -s -o /dev/null -w "empty:    %{http_code}\n" http://localhost:8080/report/chk-empty

# error
curl -s -X POST http://localhost:8080/api/report/chk-error \
  -H "X-Ingest-Token: $TOKEN" -d '{"status":"error","role":"X","language":"en","error":"test"}' > /dev/null
curl -s -o /dev/null -w "error:    %{http_code}\n" http://localhost:8080/report/chk-error

kill $SERVER_PID
```

Expected: `unknown: 404`, `scoring: 200` + has auto-refresh, `empty: 200`,
`error: 500`. Any two of these returning the same body is a fail.

## 8. Health check alerts on real failure, stays quiet otherwise

**Depends on:** `scripts/check_agent_health.py`,
`.github/workflows/agent-health-check.yml`

- [ ] After any change to this script, manually trigger the workflow
      (`workflow_dispatch` in the Actions tab) and read its log — it
      should either say "Healthy" or show the consecutive-failure count,
      never crash.
- [ ] Confirm `FAILURE_THRESHOLD` is still >= 2 — a threshold of 1 means a
      normal cold-start blip (the service scales to zero between calls)
      will alert every time.

```bash
grep -n "FAILURE_THRESHOLD = " scripts/check_agent_health.py
```

## Before adding a new row

If you build a new behavior, add its row here in the same sitting — not
"later." A behavior with no checklist row is a behavior nobody will
remember to re-verify the next time something nearby changes.
