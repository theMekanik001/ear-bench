"""fw_transcribe.py - faster-whisper transcription for the ear bench.
Usage: python fw_transcribe.py <audio-file> [model] [compute_type]
The bench pipeline always hands this a 16 kHz mono WAV (ears-all.ps1
converts first). WAV input is decoded with the stdlib + numpy and fed to
the model as an array, so transcription never depends on PyAV's
decode_audio (whose av.open kwargs change between PyAV releases).
Non-WAV input falls back to faster-whisper's own decoder.
Prints timings + transcript; writes <audio-file>.fw.txt beside the input."""
import sys, time, wave
import numpy as np
from faster_whisper import WhisperModel

def load_wav_array(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        ch = w.getnchannels()
        sw = w.getsampwidth()
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a

def main():
    path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "small"
    compute = sys.argv[3] if len(sys.argv) > 3 else "int8"
    t0 = time.time()
    model = WhisperModel(model_name, device="cpu", compute_type=compute)
    t_load = time.time() - t0
    src = load_wav_array(path) if path.lower().endswith(".wav") else path
    t1 = time.time()
    segments, info = model.transcribe(src, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    t_tr = time.time() - t1
    with open(path + ".fw.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"MODEL {model_name} compute={compute} lang={info.language} "
          f"audio={info.duration:.1f}s load={t_load:.1f}s transcribe={t_tr:.1f}s")
    print(text)

if __name__ == "__main__":
    main()
