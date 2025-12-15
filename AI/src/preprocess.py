# 음성 전처리 
# user, songs_vocal -> wav, segment 6개
import os
import subprocess
from pathlib import Path
from glob import glob

ROOT = Path(__file__).resolve().parents[1]

DATA_USER = ROOT / "data" / "user"
DATA_SONGS = ROOT / "data" / "songs_vocal"

WAV_USER_DIR = ROOT / "data" / "_wav" / "user"
WAV_SONG_DIR = ROOT / "data" / "_wav" / "songs"
SEG_DIR = ROOT / "data" / "_segments"

def run(cmd: list[str]):
    subprocess.run(cmd, check=True)

def to_wav_16k_mono(in_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(out_path)
    ])

def segment_into_6(in_wav: Path, out_folder: Path):
    """
    6개로 '균등 분할': 전체 길이를 6등분해서 seg0~seg5 생성
    (데모에는 이게 제일 빠르고 충분)
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    # 길이(초) 구하기
    # ffprobe로 duration 추출
    result = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(in_wav)
    ])
    duration = float(result.decode().strip())
    seg_len = duration / 6.0

    for i in range(6):
        start = i * seg_len
        out_path = out_folder / f"seg{i}.wav"
        run([
            "ffmpeg", "-y",
            "-i", str(in_wav),
            "-ss", f"{start}",
            "-t", f"{seg_len}",
            "-ac", "1", "-ar", "16000",
            str(out_path)
        ])

def main():
    # 1) user m4a -> wav
    user_files = glob(str(DATA_USER / "*"))
    if not user_files:
        print("data/user/에 user.m4a 넣어줘.")
        return
    user_in = Path(user_files[0])
    user_wav = WAV_USER_DIR / "user.wav"
    print("[USER] convert:", user_in, "->", user_wav)
    to_wav_16k_mono(user_in, user_wav)

    # 2) songs mp3 -> wav + segment 6개
    song_files = glob(str(DATA_SONGS / "*.mp3"))
    if not song_files:
        print("data/songs_vocal/에 mp3를 넣어줘.")
        return

    for p in song_files:
        in_mp3 = Path(p)
        song_id = in_mp3.stem
        out_wav = WAV_SONG_DIR / f"{song_id}.wav"
        print("[SONG] convert:", in_mp3, "->", out_wav)
        to_wav_16k_mono(in_mp3, out_wav)

        out_seg_folder = SEG_DIR / song_id
        print("[SONG] segment 6:", out_wav, "->", out_seg_folder)
        segment_into_6(out_wav, out_seg_folder)

    print("✅ preprocess done")

if __name__ == "__main__":
    main()
