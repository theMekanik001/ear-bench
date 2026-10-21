# Ear Bench

A three-eared transcription bench. One audio file goes in; three independent
speech engines transcribe it, a fourth layer tags sound events, and the word
ears vote. When independent engines agree, the transcript is trustworthy;
when they disagree, the report says so instead of guessing quietly.

Built for stream-archive work: setlists, transcripts, and caption repair
where a single engine's hallucination is worse than no answer.

## The ears

| Ear | Engine | Job |
|-----|--------|-----|
| 1 | whisper.cpp (`ggml-base.en`) | Fast first pass, clean punctuation |
| 2 | SenseVoice (sherpa-onnx, int8) | Words + emotion + event tags |
| 3 | faster-whisper (CTranslate2, CPU int8) | Model ladder (base/small/...), second opinion |
| 4 | sherpa-onnx Zipformer audio tagger | Sound events (music, speech, noise...) |

`ears-all.ps1` runs all four, writes a report with every transcript, and
scores pairwise token agreement between the word ears plus an average
consensus verdict (EARS AGREE at >= 85%).

## Layout on the bench machine (Legion, Windows)

Everything lives under `C:\Users\robse\pcloud-stage\`:

- `ears-all.ps1` - the wrapper (this repo is its source of truth)
- `whisper\Release\whisper-cli.exe` + `whisper\ggml-base.en.bin`
- `sense\` - sherpa-onnx v1.13.8 win-x64 binaries, SenseVoice int8 model,
  Zipformer audio-tagging model
- `fwbench\venv\` - Python 3.12 venv with faster-whisper installed
- `fwbench\fw_transcribe.py` - the faster-whisper runner

## Setup

1. `winget install -e --id Python.Python.3.12`
2. `python -m venv fwbench\venv`
3. `fwbench\venv\Scripts\python.exe -m pip install -r requirements.txt`
4. Place the whisper.cpp + sherpa-onnx binaries/models per the layout above.
5. Run: `powershell -File ears-all.ps1 -InFile <audio> [-FwModel small]`

## Design notes

- **No disposable installs.** The bench is maintained, not improvised:
  the venv lives in `fwbench\`, and this repo is the source of truth for
  both scripts.
- **PyAV-proofing.** `fw_transcribe.py` decodes WAV input with the Python
  stdlib + numpy and hands the model a float array. faster-whisper's own
  decoder passes kwargs to `av.open()` that changed across PyAV releases
  (av 19 broke it); the bench pipeline always converts to 16 kHz mono WAV
  first, so the main path never touches PyAV.
- **Shell discipline.** PowerShell scripts here are ASCII-only with a
  UTF-8 BOM (PowerShell 5.1 misreads BOM-less UTF-8). The git copy of
  `ears-all.ps1` is BOM-less; add the BOM when deploying to the bench.
- **Inline remote commands bite.** Ship scripts as files and run them;
  cmd.exe eats `<`, `>`, and `$_` out of inline SSH command strings.

## Benchmark (2026-10-21, Legion: Ryzen 5 5600H, CPU only)

180 s of real stream speech (gaming stream, voice over game audio):

| Engine | Model | Total time | Notes |
|--------|-------|-----------|-------|
| whisper.cpp | base.en | 11.9 s | wall clock incl. load |
| faster-whisper | base int8 | 10.0 s | 1.7 s load + 8.3 s transcribe |
| faster-whisper | small int8 | 21.6 s | 2.6 s load + 19.0 s transcribe |

Consensus on the same file (ears-all v2): Whisper/SenseVoice 94%,
Whisper/FasterWhisper 98%, SenseVoice/FasterWhisper 94% - average 95%,
EARS AGREE. faster-whisper base is the fastest ear at equal accuracy;
small trades speed for the fewest word errors. Default bench model: small.
