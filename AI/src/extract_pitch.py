# Pitch 특징 추출 함수
import numpy as np
import librosa
import json
from pathlib import Path

def pitch_features(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    if f0 is None:
        return None

    f0v = f0[voiced_flag]
    if len(f0v) < 10:
        return None

    feat = {
        "f0_min": float(np.min(f0v)),
        "f0_max": float(np.max(f0v)),
        "f0_median": float(np.median(f0v)),
        "range_semitone": float(12 * np.log2(np.max(f0v) / np.min(f0v))),
        "std_semitone": float(np.std(12 * np.log2(f0v / 440.0))),
        "voiced_ratio": float(np.mean(voiced_flag))
    }
    return feat

def save_pitch(path_wav, out_dir="data/embeddings/pitch"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pitch_features(path_wav)
    if feat is None:
        print("[Pitch] failed:", path_wav)
        return

    out_path = out_dir / (Path(path_wav).stem + ".json")
    with open(out_path, "w") as f:
        json.dump(feat, f, indent=2)

    print(f"[Pitch] saved → {out_path}")
    print(feat)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_pitch.py path/to/audio.wav")
        sys.exit(1)

    save_pitch(sys.argv[1])
