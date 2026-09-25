#!/usr/bin/env python
"""
Ask the local llama.cpp server a question, speak the answer to a .wav file.

Two independent steps, chained — the LLM and the TTS engine never talk to each
other:

  1. POST the prompt to llama-server's OpenAI-compatible /v1/chat/completions
  2. Pull choices[0].message.content out of the JSON
  3. Strip markdown so it isn't read aloud as punctuation
  4. Synthesize to a wav

Not streaming, not real time. It waits for the full answer, then synthesizes.

Usage
-----
  python ask_and_speak.py "Explain why the sky is blue, in three sentences."
  python ask_and_speak.py "..." --engine kokoro --speaker af_heart
  python ask_and_speak.py "..." --out answer.wav --keep-text

Requires llama-server already running (default: http://127.0.0.1:5174).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
OUTPUT_DIR = HERE / "output"


def default_out_path(stem: str = "speech") -> Path:
    """<tts>/output/<stem>_YYYYMMDD-HHMMSS.wav — never overwrites."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    p = OUTPUT_DIR / f"{stem}_{ts}.wav"
    n = 2
    while p.exists():          # same-second collision
        p = OUTPUT_DIR / f"{stem}_{ts}_{n}.wav"
        n += 1
    return p

DEFAULTS = {
    "piper": {
        "model": MODELS / "piper/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "speaker": None,
    },
    "kokoro": {
        "model": MODELS / "kokoro/model/kokoro.onnx",
        "voices": MODELS / "kokoro/voice/voices-v1.0.bin",
        "speaker": "af_heart",
    },
    "qwen": {
        "model": MODELS / "qwen3-tts-0.6b-customvoice",
        "speaker": "ryan",
    },
}


def die(msg: str):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# 1-2. talk to llama-server
# --------------------------------------------------------------------------- #

def ask_llm(url: str, prompt: str, system: str | None, max_tokens: int,
            temperature: float, timeout: float) -> str:
    payload = {"messages": [], "max_tokens": max_tokens, "temperature": temperature}
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except urllib.error.URLError as e:
        die(f"could not reach llama-server at {url}\n       {e}\n"
            f"       is it running? start it with your usual llama-server command")
    except TimeoutError:
        die(f"llama-server timed out after {timeout}s — try a smaller --max-tokens")

    msg = body["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    # If thinking is enabled server-side, reasoning lands here — never speak it.
    if not text and msg.get("reasoning_content"):
        die("model returned only reasoning_content and no answer; "
            "check --chat-template-kwargs '{\\\"enable_thinking\\\":false}'")
    if not text:
        die("model returned empty content")
    return text


# --------------------------------------------------------------------------- #
# 3. make markdown speakable
# --------------------------------------------------------------------------- #

def clean_for_speech(text: str) -> str:
    """Strip markup a TTS engine would otherwise pronounce."""
    # drop <think> blocks entirely, if any slipped through
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    # fenced code blocks -> a short spoken placeholder, not the code itself
    text = re.sub(r"```[\w+-]*\n.*?```", " (code omitted) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)          # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> label
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"(\*\*|__|\*|_)", "", text)         # emphasis
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.M)  # table rows
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)   # blockquotes
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 4. synthesize
# --------------------------------------------------------------------------- #

def add_voice_args(ap: argparse.ArgumentParser) -> None:
    """Shared voice/delivery flags, so every script offers the same controls."""
    ap.add_argument("--engine", default="piper", choices=sorted(DEFAULTS))
    ap.add_argument("--model", help="override TTS model path")
    ap.add_argument("--voices", help="kokoro voices .bin override")
    ap.add_argument("--speaker", help="voice name")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="talking speed; 1.0 normal, 1.5 faster, 0.75 slower "
                         "(kokoro + piper; qwen has no speed control)")
    ap.add_argument("--noise", type=float, default=None,
                    help="piper only: expressiveness/variability, default 0.667. "
                         "Lower = flatter and steadier, higher = more varied")
    ap.add_argument("--lang", default="en-us", help="kokoro language code")
    ap.add_argument("--qwen-language", default="english")


def synth(engine: str, text: str, args) -> tuple[np.ndarray, int]:
    d = DEFAULTS[engine]
    model = Path(args.model) if args.model else d["model"]
    if not model.exists():
        die(f"model not found: {model}")
    speaker = args.speaker or d.get("speaker")

    speed = getattr(args, "speed", 1.0) or 1.0
    noise = getattr(args, "noise", None)

    def unsupported(what: str):
        print(f"note: --{what} has no effect with the {engine} engine; ignoring",
              file=sys.stderr)

    if engine == "piper":
        from piper import PiperVoice, SynthesisConfig
        # piper's length_scale is DURATION, so it's inverted: higher = slower.
        cfg = SynthesisConfig(length_scale=1.0 / speed)
        if noise is not None:
            cfg.noise_scale = noise
        v = PiperVoice.load(str(model))
        chunks = list(v.synthesize(text, syn_config=cfg))
        if not chunks:
            die("piper produced no audio")
        return (np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32),
                chunks[0].sample_rate)

    if engine == "kokoro":
        from kokoro_onnx import Kokoro
        voices = Path(args.voices) if args.voices else d["voices"]
        if not voices.exists():
            die(f"voices file not found: {voices}")
        if noise is not None:
            unsupported("noise")
        audio, sr = Kokoro(str(model), str(voices)).create(
            text, voice=speaker, speed=speed, lang=args.lang)
        return np.asarray(audio, dtype=np.float32), sr

    if engine == "qwen":
        import torch
        from qwen_tts import Qwen3TTSModel
        if noise is not None:
            unsupported("noise")
        if speed != 1.0:
            unsupported("speed")
        m = Qwen3TTSModel.from_pretrained(
            str(model), device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
        kw = {}
        if getattr(args, "instruct", None):
            kw["instruct"] = args.instruct
        wavs, sr = m.generate_custom_voice(
            text=text, speaker=speaker, language=args.qwen_language, **kw)
        return np.asarray(wavs[0], dtype=np.float32), sr

    die(f"unknown engine {engine}")


# --------------------------------------------------------------------------- #
# 4b. load once, synthesize many — for callers that synthesize in pieces
# --------------------------------------------------------------------------- #

class Voice:
    """A TTS engine loaded once, ready to synthesize repeatedly.

    synth() above reloads the model on every call, which is fine for one shot
    and useless when a streaming caller invokes it per sentence. Same engines,
    same flags, same defaults — the load just happens up front.
    """

    def __init__(self, engine: str, say):
        self.engine = engine
        self._say = say
        self.sample_rate: int | None = None

    def say(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize one piece of text. None if the engine produced nothing."""
        got = self._say(text)
        if got is None:
            return None
        audio, sr = got
        self.sample_rate = sr
        return audio, sr


def load_voice(engine: str, args) -> Voice:
    """Resolve paths, load the model, return a Voice. Piper and kokoro only."""
    d = DEFAULTS[engine]
    model = Path(args.model) if args.model else d["model"]
    if not model.exists():
        die(f"model not found: {model}")
    speaker = args.speaker or d.get("speaker")
    speed = getattr(args, "speed", 1.0) or 1.0
    noise = getattr(args, "noise", None)

    if engine == "piper":
        from piper import PiperVoice, SynthesisConfig
        # piper's length_scale is DURATION, so it's inverted: higher = slower.
        cfg = SynthesisConfig(length_scale=1.0 / speed)
        if noise is not None:
            cfg.noise_scale = noise
        v = PiperVoice.load(str(model))

        def say(text: str):
            chunks = list(v.synthesize(text, syn_config=cfg))
            if not chunks:
                return None
            audio = np.concatenate([c.audio_float_array for c in chunks])
            return audio.astype(np.float32), chunks[0].sample_rate

        return Voice(engine, say)

    if engine == "kokoro":
        from kokoro_onnx import Kokoro
        voices = Path(args.voices) if args.voices else d["voices"]
        if not voices.exists():
            die(f"voices file not found: {voices}")
        if noise is not None:
            print(f"note: --noise has no effect with the {engine} engine; ignoring",
                  file=sys.stderr)
        k = Kokoro(str(model), str(voices))

        def say(text: str):
            audio, sr = k.create(text, voice=speaker, speed=speed, lang=args.lang)
            return np.asarray(audio, dtype=np.float32), sr

        return Voice(engine, say)

    die(f"{engine} cannot synthesize incrementally; use synth() for a single pass")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    add_voice_args(ap)
    ap.add_argument("--llm-url", default="http://127.0.0.1:5174/v1/chat/completions")
    ap.add_argument("--system", help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", help="explicit output path (default: output/answer_<timestamp>.wav)")
    ap.add_argument("--keep-text", action="store_true",
                    help="also write the answer as .txt next to the wav")
    args = ap.parse_args()

    t0 = time.perf_counter()
    raw = ask_llm(args.llm_url, args.prompt, args.system,
                  args.max_tokens, args.temperature, args.timeout)
    t_llm = time.perf_counter() - t0

    spoken = clean_for_speech(raw)
    if not spoken:
        die("nothing left to speak after cleaning markdown")

    t0 = time.perf_counter()
    audio, sr = synth(args.engine, spoken, args)
    t_tts = time.perf_counter() - t0

    out = Path(args.out) if args.out else default_out_path('answer')
    sf.write(str(out), audio, sr)
    if args.keep_text:
        out.with_suffix(".txt").write_text(raw, encoding="utf-8")

    dur = len(audio) / sr
    print()
    print("--- answer ---")
    print(raw[:600] + ("..." if len(raw) > 600 else ""))
    print("--------------")
    print(f"llm time   : {t_llm:.1f} s  ({len(raw)} chars)")
    print(f"tts time   : {t_tts:.1f} s  ({args.engine})")
    print(f"audio      : {dur:.1f} s -> {out.resolve()}")
    if args.keep_text:
        print(f"text       : {out.with_suffix('.txt').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
