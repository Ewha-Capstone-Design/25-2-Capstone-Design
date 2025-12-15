# 유저 음성 -> 특징 추출 -> 곡 특징 로드 -> 점수 계산 -> 결과 출력
from pathlib import Path
from glob import glob
import json
import subprocess
import torch

from extract_ecapa import ecapa_embed
from extract_pitch import pitch_features
from score import segment_score

ROOT = Path(__file__).resolve().parents[1]

FEATURE_E = ROOT / "data" / "_features" / "ecapa"
FEATURE_P = ROOT / "data" / "_features" / "pitch"

USER_WAV_DIR = ROOT / "data" / "_wav" / "user"

def convert_to_user_wav(input_audio: Path) -> Path:
    """m4a/mp3/wav 뭐가 오든 user.wav(16k mono)로 변환해서 반환"""
    USER_WAV_DIR.mkdir(parents=True, exist_ok=True)
    out_wav = USER_WAV_DIR / "user.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_audio),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(out_wav)
    ], check=True)

    return out_wav

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/recommend_user.py path/to/user_audio.(m4a|wav|mp3)")
        return

    user_in = Path(sys.argv[1])
    if not user_in.exists():
        print("User file not found:", user_in)
        return

    # 1) 유저만 변환
    user_wav = convert_to_user_wav(user_in)

    # 2) 유저 특징
    u_emb = ecapa_embed(str(user_wav))
    u_pitch = pitch_features(str(user_wav))
    if not u_pitch:
        print("유저 pitch 추출 실패(무음/잡음/반주 영향). 다른 샘플로 시도.")
        return

    # 3) 저장된 곡 feature 로드해서 점수 계산
    pt_files = glob(str(FEATURE_E / "*.pt"))
    if not pt_files:
        print("곡 feature가 없습니다. 먼저 build_features를 1회 실행하세요.")
        return

    by_song = {}
    for pt in pt_files:
        pt = Path(pt)
        key = pt.stem  # songid__seg0
        if "__" not in key:
            continue

        song_id, seg_name = key.split("__", 1)
        pitch_path = FEATURE_P / f"{key}.json"
        if not pitch_path.exists():
            continue

        s_emb = torch.load(pt)
        with open(pitch_path, "r") as f:
            s_pitch = json.load(f)

        score, se, sp = segment_score(u_emb, u_pitch, s_emb, s_pitch)
        by_song.setdefault(song_id, []).append((score, se, sp))

    alpha = 0.6
    results = []
    for song_id, seg_scores in by_song.items():
        if not seg_scores:
            continue
        seg_only = [x[0] for x in seg_scores]
        song_score = alpha * max(seg_only) + (1 - alpha) * (sum(seg_only) / len(seg_only))
        best = max(seg_scores, key=lambda x: x[0])
        results.append((song_score, song_id, best))

    results.sort(reverse=True, key=lambda x: x[0])

    print("\n=== TOP MATCH ===")
    if results:
        s, sid, best = results[0]
        print(f"1) {sid}  song_score={s:.3f}  best_seg(total={best[0]:.3f}, ecapa={best[1]:.3f}, pitch={best[2]:.3f})")

    print("\n=== TOP 10 ===")
    for i, (s, sid, best) in enumerate(results[:10], 1):
        print(f"{i:2d}. {sid}  song_score={s:.3f}  best(total={best[0]:.3f}, ecapa={best[1]:.3f}, pitch={best[2]:.3f})")

if __name__ == "__main__":
    main()
