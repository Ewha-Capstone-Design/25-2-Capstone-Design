# 유저의 녹음 전처리 
# 30초 녹음 -> MR 제거 -> data/user/user_record_N.m4a
import subprocess
import shutil
from pathlib import Path
import time

# 녹음용
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "data" / "_tmp"
USER_DIR = ROOT / "data" / "user"

TMP_DIR.mkdir(parents=True, exist_ok=True)
USER_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
RECORD_SECONDS = 30


def get_next_user_filename() -> Path:
    existing = USER_DIR.glob("user_record_*.m4a")
    indices = []
    for f in existing:
        try:
            indices.append(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    next_idx = max(indices) + 1 if indices else 0
    return USER_DIR / f"user_record_{next_idx}.m4a"


def record_30s_to_wav(out_wav: Path):
    """마이크로 30초 녹음해서 wav로 저장 (16k mono)"""
    print(f"🎙️ 녹음 시작: {RECORD_SECONDS}초 (샘플링 {SAMPLE_RATE}Hz, mono)")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("✅ 녹음 종료")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, SAMPLE_RATE)
    print("→ 원본 녹음 저장:", out_wav)


def remove_mr_with_demucs(input_audio: Path, out_m4a: Path):
    """Demucs로 MR 제거 후 vocals.wav를 m4a로 변환해 저장"""
    demucs_out = TMP_DIR / "demucs_out"
    shutil.rmtree(demucs_out, ignore_errors=True)

    print("🧪 MR 제거(Demucs) 실행 중...")
    subprocess.run([
        "demucs",
        "-n", "htdemucs",
        "-o", str(demucs_out),
        str(input_audio)
    ], check=True)

    model_dir = demucs_out / "htdemucs" / input_audio.stem
    vocals_wav = model_dir / "vocals.wav"

    if not vocals_wav.exists():
        raise RuntimeError("Demucs 결과 vocals.wav를 찾을 수 없음")

    print("🎧 vocals.wav → m4a 변환 중...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(vocals_wav),
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_m4a)
    ], check=True)

    print("✅ MR 제거 완료")
    print("→ 보컬 저장:", out_m4a)

    # 임시 파일 정리(원하면 주석 처리)
    shutil.rmtree(demucs_out, ignore_errors=True)


def main():
    # 1) 30초 녹음 → tmp wav
    ts = int(time.time())
    raw_wav = TMP_DIR / f"user_raw_{ts}.wav"
    record_30s_to_wav(raw_wav)

    # 2) MR 제거 → data/user/user_record_N.m4a
    out_audio = get_next_user_filename()
    remove_mr_with_demucs(raw_wav, out_audio)

    print("\n🚀 이제 아래 명령으로 추천 실행하면 됨:")
    print(f"python src/recommend_user.py {out_audio}")


if __name__ == "__main__":
    main()
