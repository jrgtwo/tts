# tts

Local text-to-speech on the CPU. Three engines behind one set of flags, plus a
thin bridge that asks a local llama.cpp server a question and speaks the answer.

Everything runs offline. No model is ever downloaded at runtime — if the weights
aren't on disk, the scripts exit with an error naming the path they wanted.

## Scripts

| Script | What it does |
| --- | --- |
| `speak.py` | Read text from a file, the clipboard, or stdin → `.wav` |
| `ask_and_speak.py` | Prompt a running llama-server, speak the answer → `.wav` |
| `tts_test.py` | Benchmark one engine on fixed text and report RTF |

`ask_and_speak.py` is also the shared library: `DEFAULTS`, `synth()`,
`clean_for_speech()`, and the common voice flags live there, and `speak.py`
imports them.

## Engines

| Engine | Weights | Speaker | Speed | Expressiveness |
| --- | --- | --- | --- | --- |
| `piper` (default) | `en_US-amy-medium.onnx` | single voice | `--speed` | `--noise` |
| `kokoro` | `kokoro.onnx` + `voices-v1.0.bin` | `--speaker af_heart` | `--speed` | — |
| `qwen` | `qwen3-tts-0.6b-customvoice/` | `--speaker ryan` | — | `--instruct` |

Piper and Kokoro are ONNX and fast. Qwen is a 0.6B torch model on CPU — much
slower to load and synthesize, but it takes a natural-language style
instruction (`--instruct "sound excited"`) and covers 10 languages.

Flags that an engine doesn't support print a note to stderr and are ignored
rather than failing.

## Setup

Python 3.13, virtualenv in `.venv/`:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate elsewhere
pip install piper-tts kokoro-onnx qwen-tts soundfile numpy torch
```

### Model weights

Weights are sourced manually and are **not** in this repo. Put them here:

```
models/
├── piper/en/en_US/amy/medium/en_US-amy-medium.onnx   (+ .onnx.json beside it)
├── kokoro/
│   ├── model/kokoro.onnx
│   └── voice/voices-v1.0.bin
└── qwen3-tts-0.6b-customvoice/                       (full HF snapshot)
```

- **piper** — `rhasspy/piper-voices` on Hugging Face, `en/en_US/amy/medium/`.
  The `.onnx.json` config must sit next to the `.onnx`.
- **kokoro** — the `kokoro-onnx` project's release assets (the model and
  `voices-v1.0.bin`). Paths above are what `DEFAULTS` expects; override with
  `--model` / `--voices` if yours differ.
- **qwen** — `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`:
  ```bash
  hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
      --local-dir models/qwen3-tts-0.6b-customvoice
  ```

Override any default path with `--model`, `--voices`, or `--speaker`.

## Usage

### Speak some text

```bash
python speak.py response.txt                      # from a file
python speak.py --clipboard                       # straight from the clipboard
echo "hello there" | python speak.py              # from stdin

python speak.py response.txt --engine kokoro --speaker af_heart
python speak.py response.txt --out talk.wav --raw
```

Markdown is stripped before synthesis, so headings, emphasis, links, and table
pipes aren't read aloud as punctuation; fenced code blocks become a spoken
"(code omitted)". Pass `--raw` to speak the text exactly as given.

Output defaults to `output/<name>_<timestamp>.wav` and never overwrites.

### Ask and speak

Requires llama-server already running with an OpenAI-compatible endpoint
(default `http://127.0.0.1:5174/v1/chat/completions`):

```bash
python ask_and_speak.py "Explain why the sky is blue, in three sentences."
python ask_and_speak.py "..." --engine kokoro --speaker af_heart
python ask_and_speak.py "..." --out answer.wav --keep-text
```

Not streaming — it waits for the full answer, then synthesizes. It prints LLM
time, TTS time, and audio length. If the model returns only `reasoning_content`
and no answer, it stops rather than speaking the reasoning; disable thinking
server-side with `--chat-template-kwargs '{"enable_thinking":false}'`.

Useful flags: `--llm-url`, `--system`, `--max-tokens` (400), `--temperature`
(0.7), `--timeout` (600s), `--keep-text`.

### Benchmark

```bash
python tts_test.py --engine piper  --model models/piper/en/en_US/amy/medium/en_US-amy-medium.onnx
python tts_test.py --engine kokoro --model models/kokoro/model/kokoro.onnx \
                   --voices models/kokoro/voice/voices-v1.0.bin --list-voices
python tts_test.py --engine qwen   --model models/qwen3-tts-0.6b-customvoice --speaker ryan
```

The number that matters is **RTF** (real-time factor): synthesis seconds per
second of audio. RTF < 1.0 is faster than real time.

## Repo layout

```
speak.py            entry point: text → wav
ask_and_speak.py    entry point: prompt → LLM → wav; also the shared library
tts_test.py         engine benchmark
models/             weights (not committed — ~2.5 GB)
output/             generated wavs (not committed)
```

`models/`, `output/`, `.venv/`, and stray `*.wav` are gitignored — model
weights are far over GitHub's 100 MB file limit and re-downloadable.
