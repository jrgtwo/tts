#!/usr/bin/env python
"""
Compare local TTS engines on the same text and report timing.

Every engine takes a LOCAL model path. Nothing is downloaded — if the weights
aren't on disk, this exits with an error telling you what it wanted.

Usage
-----
  python tts_test.py --engine kokoro --model <kokoro.onnx> --voices <voices.bin> --text "Hello."
  python tts_test.py --engine piper  --model <voice.onnx>                        --text "Hello."
  python tts_test.py --engine qwen   --model <model_dir> --speaker Ryan          --text "Hello."

  python tts_test.py --engine kokoro --model ... --voices ... --list-voices

The number that matters is RTF (real-time factor): synthesis seconds per second
of audio produced. RTF < 1.0 means faster than real time.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Local text to speech synthesis, running entirely on the CPU."
)


def die(msg: str) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def need(path: str | None, what: str) -> Path:
    if not path:
        die(f"--{what} is required for this engine")
    p = Path(path)
    if not p.exists():
        die(f"{what} not found: {p}\n       weights are sourced manually — nothing is downloaded")
    return p


# --------------------------------------------------------------------------- #
# engines: each returns (audio_float32_mono, sample_rate, load_seconds)
# --------------------------------------------------------------------------- #

def run_kokoro(args) -> tuple[np.ndarray, int, float]:
    from kokoro_onnx import Kokoro

    model = need(args.model, "model")
    voices = need(args.voices, "voices")

    t0 = time.perf_counter()
    k = Kokoro(str(model), str(voices))
    load = time.perf_counter() - t0

    if args.list_voices:
        print("voices:", ", ".join(k.get_voices()))
        raise SystemExit(0)

    voice = args.speaker or "af_heart"
    t0 = time.perf_counter()
    audio, sr = k.create(args.text, voice=voice, speed=args.speed, lang=args.lang)
    args._synth = time.perf_counter() - t0
    return np.asarray(audio, dtype=np.float32), sr, load


def run_piper(args) -> tuple[np.ndarray, int, float]:
    from piper import PiperVoice, SynthesisConfig

    model = need(args.model, "model")

    t0 = time.perf_counter()
    voice = PiperVoice.load(str(model), config_path=args.config)
    load = time.perf_counter() - t0

    syn = SynthesisConfig(length_scale=1.0 / args.speed if args.speed else 1.0)

    t0 = time.perf_counter()
    chunks = list(voice.synthesize(args.text, syn_config=syn))
    args._synth = time.perf_counter() - t0

    if not chunks:
        die("piper produced no audio chunks")
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return audio, chunks[0].sample_rate, load


def run_qwen(args) -> tuple[np.ndarray, int, float]:
    import torch
    from qwen_tts import Qwen3TTSModel

    model_dir = need(args.model, "model")

    t0 = time.perf_counter()
    m = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    load = time.perf_counter() - t0

    if args.list_voices:
        print("speakers :", m.get_supported_speakers())
        print("languages:", m.get_supported_languages())
        raise SystemExit(0)

    kw = dict(text=args.text, language=args.qwen_language)
    if args.instruct:
        kw["instruct"] = args.instruct

    t0 = time.perf_counter()
    wavs, sr = m.generate_custom_voice(speaker=args.speaker or "ryan", **kw)
    args._synth = time.perf_counter() - t0
    return np.asarray(wavs[0], dtype=np.float32), sr, load


ENGINES = {"kokoro": run_kokoro, "piper": run_piper, "qwen": run_qwen}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--model", help="local path to the model file or directory")
    ap.add_argument("--voices", help="kokoro: path to voices .bin")
    ap.add_argument("--config", help="piper: path to voice .onnx.json (defaults beside the model)")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--speaker", help="voice name (kokoro: af_heart, qwen: Ryan, ...)")
    ap.add_argument("--lang", default="en-us", help="kokoro language code")
    ap.add_argument("--qwen-language", default="english", help="qwen language name")
    ap.add_argument("--instruct", help="qwen: style/emotion instruction")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", help="output wav (default: <engine>_out.wav)")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()
    args._synth = 0.0

    audio, sr, load = ENGINES[args.engine](args)

    dur = len(audio) / sr if sr else 0.0
    synth = args._synth
    rtf = synth / dur if dur else float("nan")

    out = Path(args.out or f"{args.engine}_out.wav")
    sf.write(str(out), audio, sr)

    print(f"engine        : {args.engine}")
    print(f"model         : {args.model}")
    print(f"voice         : {args.speaker or '(default)'}")
    print(f"sample rate   : {sr} Hz")
    print(f"audio length  : {dur:.2f} s")
    print(f"load time     : {load:.2f} s")
    print(f"synth time    : {synth:.2f} s")
    print(f"RTF           : {rtf:.3f}   ({'faster' if rtf < 1 else 'SLOWER'} than real time)")
    print(f"wrote         : {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
