#!/usr/bin/env python
"""
Ask the local llama.cpp server a question and speak the answer as it arrives.

Same two steps as ask_and_speak.py, but overlapped — the answer is streamed,
buffered into sentences, and each sentence is synthesized and played while the
model is still writing the next one:

  SSE deltas --> sentence chunker --> TTS worker --> playback
                  (flush on . ! ?)      (queue)       (queue)

TTS needs a whole sentence to get prosody right, so nothing is spoken until a
sentence closes. First audio lands a second or two in; after that the model
generates text faster than speech consumes it, so the queue stays fed.

Piper and kokoro stream. Qwen is too slow to keep up, so --engine qwen falls
back to the batch path: text still streams to the terminal, audio comes at the
end.

The prompt can come from an argument, a file, the clipboard, or stdin.

Usage
-----
  python stream_and_speak.py "Explain why the sky is blue."
  python stream_and_speak.py --prompt-file prompt.txt
  python stream_and_speak.py --clipboard
  type prompt.txt | python stream_and_speak.py

  python stream_and_speak.py --clipboard --engine kokoro --speaker af_heart
  python stream_and_speak.py "Write a bedtime story." --no-save --speed 1.15
  python stream_and_speak.py --prompt-file p.txt --llm-model qwen27b  # router

Requires llama-server running (default http://127.0.0.1:5174, or $LLM_URL)
and sounddevice for playback: pip install sounddevice
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from ask_and_speak import (add_voice_args, clean_for_speech, default_out_path,
                           die, load_voice, synth)
from speak import read_clipboard      # speak.py owns the powershell call

DEFAULT_LLM_URL = os.environ.get(
    "LLM_URL", "http://127.0.0.1:5174/v1/chat/completions")

STREAMING_ENGINES = ("piper", "kokoro")

# A sentence ends at .!?… plus any closing quote/bracket, then whitespace.
# Requiring the trailing space is what keeps "3.5" and "v1.2" from splitting.
SENTENCE_END = re.compile(r'[.!?…]["\')\]]*\s')

ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
          "e.g.", "i.e.", "etc.", "fig.", "no.", "approx.", "al."}


# --------------------------------------------------------------------------- #
# sentence chunking
# --------------------------------------------------------------------------- #

def _abbrev_before(text: str, end: int) -> bool:
    """True if the period at `end` belongs to an abbreviation, not a sentence."""
    words = text[:end].rstrip().split()
    return bool(words) and words[-1].lower() in ABBREV


def _split(text: str, max_chars: int, final: bool) -> tuple[list[str], str]:
    """Pull complete sentences off the front. Returns (sentences, leftover)."""
    out: list[str] = []
    buf = text
    while True:
        m = SENTENCE_END.search(buf)
        if not m:
            break
        end = m.end()
        if _abbrev_before(buf, end):
            m2 = SENTENCE_END.search(buf, end)
            if not m2:
                break              # wait for the real sentence end to arrive
            end = m2.end()
        out.append(buf[:end])
        buf = buf[end:]

    # A run-on with no punctuation would stall the audio, so cut it at a space.
    while len(buf) > max_chars:
        cut = buf.rfind(" ", 0, max_chars)
        if cut <= 0:
            break
        out.append(buf[:cut])
        buf = buf[cut:]

    if final and buf.strip():
        out.append(buf)
        buf = ""
    return out, buf


class Chunker:
    """Accumulate streamed text, hand back complete speakable sentences.

    Markdown can't be stripped per-delta: a ``` fence opens in one chunk and
    closes hundreds of tokens later, and ** can arrive split in half. So fences
    are tracked here, and everything else goes through clean_for_speech() once
    the sentence around it is whole.
    """

    def __init__(self, max_chars: int = 300):
        self.buf = ""
        self.in_fence = False
        self.max_chars = max_chars

    def feed(self, text: str) -> list[str]:
        self.buf += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            if self.in_fence:
                i = self.buf.find("```")
                if i < 0:
                    # Code we will never speak. Keep the tail in case the
                    # closing ``` is arriving split across deltas.
                    self.buf = self.buf[-2:]
                    break
                self.buf = self.buf[i + 3:]
                self.in_fence = False
                out.append("(code omitted)")
                continue

            i = self.buf.find("```")
            if i >= 0:
                head, self.buf = self.buf[:i], self.buf[i + 3:]
                self.in_fence = True
                done, _ = _split(head, self.max_chars, final=True)
                out.extend(done)
                continue

            done, self.buf = _split(self.buf, self.max_chars, final)
            out.extend(done)
            break

        if final and self.in_fence:
            out.append("(code omitted)")     # stream ended inside a fence
            self.in_fence = False

        return [s for s in (clean_for_speech(x) for x in out) if s.strip()]


# --------------------------------------------------------------------------- #
# where the prompt comes from
# --------------------------------------------------------------------------- #

def resolve_prompt(args) -> str:
    """Prompt text from an argument, a file, the clipboard, or stdin."""
    given = [bool(args.prompt), bool(args.prompt_file), args.clipboard]
    if sum(given) > 1:
        die("give the prompt one way only: an argument, --prompt-file, "
            "or --clipboard")

    if args.clipboard:
        text = read_clipboard()
    elif args.prompt_file:
        p = Path(args.prompt_file)
        if not p.exists():
            die(f"prompt file not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
    elif args.prompt:
        text = args.prompt
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        die("no prompt — pass one as an argument, use --prompt-file or "
            "--clipboard, or pipe text in")

    if not text.strip():
        die("prompt was empty")
    return text


# --------------------------------------------------------------------------- #
# talk to llama-server, streaming
# --------------------------------------------------------------------------- #

def stream_deltas(url: str, payload: dict, headers: dict, timeout: float,
                  stats: dict):
    """Yield content deltas from an SSE chat-completions response."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        die(f"llama-server returned HTTP {e.code}\n       {body}")
    except urllib.error.URLError as e:
        die(f"could not reach llama-server at {url}\n       {e}\n"
            f"       is it running? start it with your usual llama-server command")
    except TimeoutError:
        die(f"llama-server timed out after {timeout}s")

    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            if obj.get("timings"):
                stats["timings"] = obj["timings"]

            ch = (obj.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                stats["finish_reason"] = ch["finish_reason"]

            delta = ch.get("delta") or {}
            # If thinking is on server-side, reasoning lands here. Never speak it.
            if delta.get("reasoning_content"):
                stats["reasoning_chars"] = (stats.get("reasoning_chars", 0)
                                            + len(delta["reasoning_content"]))
            piece = delta.get("content")
            if piece:
                yield piece


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #

def start_workers(voice, prebuffer: int, play: bool, audio_out: list,
                  errors: list) -> tuple[queue.Queue, list[threading.Thread]]:
    """Wire up synth -> playback. Returns the text queue and the threads."""
    text_q: queue.Queue = queue.Queue()
    audio_q: queue.Queue = queue.Queue()

    def synth_worker():
        try:
            while True:
                s = text_q.get()
                if s is None:
                    break
                got = voice.say(s)
                if got is None:
                    continue
                audio, sr = got
                audio_out.append(audio)
                audio_q.put((audio, sr))
        except Exception as e:
            errors.append(f"tts failed: {e}")
        finally:
            audio_q.put(None)

    def play_worker():
        if not play:
            while audio_q.get() is not None:
                pass
            return
        try:
            import sounddevice as sd
        except Exception as e:
            errors.append(f"playback unavailable ({e}); "
                          f"pip install sounddevice, or pass --no-play")
            while audio_q.get() is not None:
                pass
            return

        stream = None
        pending: list[tuple[np.ndarray, int]] = []
        try:
            while True:
                item = audio_q.get()
                if item is None:
                    break
                audio, sr = item
                if stream is None:
                    pending.append((audio, sr))
                    if len(pending) < prebuffer:
                        continue
                    stream = sd.OutputStream(samplerate=sr, channels=1,
                                             dtype="float32")
                    stream.start()
                    for a, _ in pending:
                        stream.write(a)
                    pending.clear()
                else:
                    stream.write(audio)

            if stream is None and pending:      # fewer chunks than --prebuffer
                sr = pending[0][1]
                stream = sd.OutputStream(samplerate=sr, channels=1,
                                         dtype="float32")
                stream.start()
                for a, _ in pending:
                    stream.write(a)
        except Exception as e:
            errors.append(f"playback failed: {e}")
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

    threads = [threading.Thread(target=synth_worker, daemon=True),
               threading.Thread(target=play_worker, daemon=True)]
    for t in threads:
        t.start()
    return text_q, threads


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?",
                    help="the prompt text (omit to use --prompt-file, "
                         "--clipboard, or stdin)")
    ap.add_argument("--prompt-file", help="read the prompt from a text file")
    ap.add_argument("--clipboard", action="store_true",
                    help="read the prompt from the clipboard")
    add_voice_args(ap)
    ap.add_argument("--instruct", help="qwen only: style, e.g. 'sound excited'")
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL,
                    help="chat-completions endpoint (default: $LLM_URL or :5174)")
    ap.add_argument("--llm-model",
                    help="model name to request; needed in router mode")
    ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"),
                    help="sent as a bearer token if the server needs one")
    ap.add_argument("--system", help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=-1,
                    help="-1 for no limit (default)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", help="explicit output path "
                                  "(default: output/stream_<timestamp>.wav)")
    ap.add_argument("--keep-text", action="store_true",
                    help="also write the answer as .txt next to the wav")
    ap.add_argument("--max-chunk-chars", type=int, default=300,
                    help="force a flush if no sentence end appears (default: 300)")
    ap.add_argument("--prebuffer", type=int, default=1,
                    help="sentences to synthesize before playback starts "
                         "(default: 1; raise to 2 if audio stutters)")
    ap.add_argument("--no-play", action="store_true", help="write the wav only")
    ap.add_argument("--no-save", action="store_true", help="play only, no wav")
    args = ap.parse_args()

    prompt = resolve_prompt(args)

    streaming = args.engine in STREAMING_ENGINES
    if not streaming:
        print(f"note: {args.engine} is too slow to stream; "
              f"text will stream, audio comes at the end", file=sys.stderr)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload: dict = {
        "messages": [],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if args.llm_model:
        payload["model"] = args.llm_model
    if args.system:
        payload["messages"].append({"role": "system", "content": args.system})
    payload["messages"].append({"role": "user", "content": prompt})

    # Load before the request goes out, so model load isn't part of the
    # time-to-first-audio.
    voice = load_voice(args.engine, args) if streaming else None

    audio_out: list[np.ndarray] = []
    errors: list[str] = []
    text_q, threads = (start_workers(voice, max(1, args.prebuffer),
                                     not args.no_play, audio_out, errors)
                       if streaming else (None, []))

    chunker = Chunker(max_chars=args.max_chunk_chars)
    stats: dict = {}
    raw_parts: list[str] = []
    t0 = time.perf_counter()
    t_first_audio = None

    print("--- answer ---")
    for piece in stream_deltas(args.llm_url, payload, headers, args.timeout, stats):
        raw_parts.append(piece)
        print(piece, end="", flush=True)
        if not streaming:
            continue
        for sentence in chunker.feed(piece):
            if t_first_audio is None:
                t_first_audio = time.perf_counter() - t0
            text_q.put(sentence)

    if streaming:
        for sentence in chunker.flush():
            if t_first_audio is None:
                t_first_audio = time.perf_counter() - t0
            text_q.put(sentence)
        text_q.put(None)

    raw = "".join(raw_parts)
    print()
    print("--------------")

    if not raw.strip():
        if stats.get("reasoning_chars"):
            die("model returned only reasoning and no answer; check "
                "--chat-template-kwargs '{\\\"enable_thinking\\\":false}'")
        die("model returned empty content")

    if streaming:
        for t in threads:
            t.join()
    else:
        spoken = clean_for_speech(raw)
        if not spoken.strip():
            die("nothing left to speak after stripping markdown")
        audio, sr = synth(args.engine, spoken, args)
        audio_out.append(audio)
        if not args.no_play:
            try:
                import sounddevice as sd
                sd.play(audio, sr)
                sd.wait()
            except Exception as e:
                errors.append(f"playback unavailable ({e}); pass --no-play")

    for e in errors:
        print(f"warning: {e}", file=sys.stderr)

    if not audio_out:
        die("no audio was produced")

    full = np.concatenate(audio_out)
    sr = voice.sample_rate if streaming else sr
    total = time.perf_counter() - t0

    out = None
    if not args.no_save:
        out = Path(args.out) if args.out else default_out_path("stream")
        sf.write(str(out), full, sr)
        if args.keep_text:
            out.with_suffix(".txt").write_text(raw, encoding="utf-8")

    print(f"engine        : {args.engine}")
    print(f"chars         : {len(raw)}")
    print(f"audio         : {len(full)/sr:.1f} s")
    if t_first_audio is not None:
        print(f"first audio   : {t_first_audio:.1f} s")
    print(f"wall clock    : {total:.1f} s")
    if stats.get("finish_reason"):
        print(f"finish_reason : {stats['finish_reason']}")
    t = stats.get("timings") or {}
    if t.get("predicted_n"):
        print(f"tokens        : {t['predicted_n']} @ "
              f"{t.get('predicted_per_second', 0):.2f} t/s")
    if out:
        print(f"wrote         : {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
