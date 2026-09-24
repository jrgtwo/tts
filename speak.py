#!/usr/bin/env python
"""
Read text and speak it to a .wav file.

Paste an LLM response into a file (or just copy it) and turn it into audio.
Markdown is stripped first, so asterisks and code fences aren't read aloud.

Usage
-----
  python speak.py response.txt                 # from a file
  python speak.py --clipboard                  # straight from the clipboard
  echo "hello there" | python speak.py         # from stdin

  python speak.py response.txt --engine kokoro --speaker af_heart
  python speak.py response.txt --out talk.wav --raw
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from ask_and_speak import (DEFAULTS, add_voice_args, clean_for_speech,
                           default_out_path, die, synth)


def read_clipboard() -> str:
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        die(f"could not read clipboard: {e}")
    if r.returncode != 0:
        die(f"clipboard read failed: {r.stderr.strip()}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="text file to read (omit for stdin)")
    ap.add_argument("--clipboard", action="store_true", help="read from the clipboard instead")
    add_voice_args(ap)
    ap.add_argument("--instruct", help="qwen only: style, e.g. 'sound excited'")
    ap.add_argument("--out", help="explicit output path (default: output/<name>_<timestamp>.wav)")
    ap.add_argument("--raw", action="store_true",
                    help="do NOT strip markdown — speak the text exactly as given")
    args = ap.parse_args()

    if args.clipboard:
        text = read_clipboard()
        default_out = default_out_path("clipboard")
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            die(f"file not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        default_out = default_out_path(p.stem)
    else:
        if sys.stdin.isatty():
            die("no input — pass a file, use --clipboard, or pipe text in")
        text = sys.stdin.read()
        default_out = default_out_path("stdin")

    if not text.strip():
        die("input was empty")

    spoken = text if args.raw else clean_for_speech(text)
    if not spoken.strip():
        die("nothing left to speak after stripping markdown (try --raw)")

    audio, sr = synth(args.engine, spoken, args)

    out = Path(args.out) if args.out else default_out
    sf.write(str(out), audio, sr)

    print(f"engine : {args.engine}")
    print(f"chars  : {len(spoken)}")
    print(f"audio  : {len(audio)/sr:.1f} s")
    print(f"wrote  : {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
