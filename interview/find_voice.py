"""Find Cartesia voices that actually support a given language.

A voice_id that sounds great in one language can mangle another, because the
voice — not just the model — carries the language support.

    python scripts/find_voice.py en
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CARTESIA_API_KEY", "")
VERSION = "2026-03-01"


def fetch_voices() -> list[dict]:
    """Page through the voice catalogue."""
    voices: list[dict] = []
    params = {"limit": 100}
    while True:
        response = requests.get(
            "https://api.cartesia.ai/voices",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Cartesia-Version": VERSION,
            },
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("data", payload if isinstance(payload, list) else [])
        voices.extend(page)
        if not payload.get("has_more") or not page:
            break
        params["starting_after"] = page[-1]["id"]
    return voices


def supports(voice: dict, code: str) -> bool:
    """A voice supports a language if it is native to it or lists its locale."""
    if voice.get("language", "").startswith(code):
        return True
    return any(
        loc.get("locale", "").startswith(code) for loc in voice.get("locales", []) or []
    )


def main() -> int:
    if not API_KEY:
        print("CARTESIA_API_KEY is not set. Add it to .env first.")
        return 1

    code = (sys.argv[1] if len(sys.argv) > 1 else "en").lower()

    try:
        voices = fetch_voices()
    except requests.RequestException as exc:
        print(f"Could not reach Cartesia: {exc}")
        return 1

    matches = [v for v in voices if supports(v, code)]

    print(f"\n{len(matches)} of {len(voices)} voices support '{code}'\n")
    if not matches:
        print("No match. Options:")
        print("  1. Clone a voice from ~10s of audio in that language.")
        print("  2. Use a different TTS provider for this language.")
        return 0

    for voice in matches:
        native = "native" if voice.get("language", "").startswith(code) else "localised"
        print(f"{voice['id']}  {voice.get('name', '?'):<24} {native:<10} {voice.get('gender', '')}")
        if voice.get("description"):
            print(f"{'':38}{voice['description'][:70]}")

    print("\nCopy an id into TTS_VOICE_ID in your .env file.")
    print("Listen to it on https://play.cartesia.ai/voices before committing to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
