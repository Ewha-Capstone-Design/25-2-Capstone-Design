# 유저 특징 로드 -> 곡 특징 로드 -> 점수 계산 -> 결과 출력
# max → “이 곡에 잘 맞는 구간이 존재하는가”
# mean → “곡 전체가 전반적으로 무리 없는가”
from pathlib import Path
from glob import glob
import json
import torch

from extract_ecapa import ecapa_embed
from extract_pitch import pitch_features
from score import segment_score 

ROOT = Path(__file__).resolve().parents[1]
USER_WAV = ROOT / "data" / "_wav" / "user" / "user.wav"
OUT_E = ROOT / "data" / "_features" / "ecapa"
OUT_P = ROOT / "data" / "_features" / "pitch"

def main():
    if not USER_WAV.exists():
        print("유저 wav 없음. 먼저 preprocess.py 실행해.")
        return

    # 유저 특징(온라인)
    u_emb = ecapa_embed(str(USER_WAV))
    u_pitch = pitch_features(str(USER_WAV))
    if not u_pitch:
        print("유저 pitch 추출 실패(무음/잡음/반주). 다른 녹음으로 시도해.")
        return

    # 곡별로 세그먼트 6개 점수 계산
    pt_files = glob(str(OUT_E / "*.pt"))
    if not pt_files:
        print("곡 feature 없음. 먼저 build_features.py 실행해.")
        return

    # song_id -> [(segScore, ecapaScore, pitchScore), ...]
    by_song = {}

    for pt in pt_files:
        pt = Path(pt)
        key = pt.stem  # songid__seg0
        if "__" not in key:
            continue
        song_id, seg_name = key.split("__", 1)

        pitch_path = OUT_P / f"{key}.json"
        if not pitch_path.exists():
            continue

        s_emb = torch.load(pt)
        with open(pitch_path, "r") as f:
            s_pitch = json.load(f)

        score, se, sp = segment_score(u_emb, u_pitch, s_emb, s_pitch)
        by_song.setdefault(song_id, []).append((score, se, sp))

    # 곡 점수 = α*max + (1-α)*mean
    alpha = 0.6
    results = []
    for song_id, seg_scores in by_song.items():
        if len(seg_scores) == 0:
            continue
        seg_only = [x[0] for x in seg_scores]
        song_score = alpha * max(seg_only) + (1 - alpha) * (sum(seg_only) / len(seg_only))
        best = max(seg_scores, key=lambda x: x[0])
        results.append((song_score, song_id, best, len(seg_scores)))

    results.sort(reverse=True, key=lambda x: x[0])

    print("\n=== TOP MATCH ===")
    if results:
        song_score, song_id, best, nseg = results[0]
        print(f"1) {song_id}  song_score={song_score:.3f}  best_seg(total={best[0]:.3f}, ecapa={best[1]:.3f}, pitch={best[2]:.3f}) segs={nseg}")

    print("\n=== TOP 10 ===")
    for i, (song_score, song_id, best, nseg) in enumerate(results[:10], 1):
        print(f"{i:2d}. {song_id}  song_score={song_score:.3f}  best(total={best[0]:.3f}, ecapa={best[1]:.3f}, pitch={best[2]:.3f})")

if __name__ == "__main__":
    main()
