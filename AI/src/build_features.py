# 세그먼트 -> ECAPA, Pitch 특징 추출 -> 저장
from pathlib import Path
from glob import glob
import json
import torch

from extract_ecapa import ecapa_embed
from extract_pitch import pitch_features

ROOT = Path(__file__).resolve().parents[1]
SEG_DIR = ROOT / "data" / "_segments"
OUT_E = ROOT / "data" / "_features" / "ecapa"
OUT_P = ROOT / "data" / "_features" / "pitch"

def main():
    OUT_E.mkdir(parents=True, exist_ok=True)
    OUT_P.mkdir(parents=True, exist_ok=True)

    seg_paths = glob(str(SEG_DIR / "*" / "seg*.wav"))
    if not seg_paths:
        print("세그먼트가 없어. 먼저 python src/preprocess.py 실행해.")
        return

    for sp in seg_paths:
        sp = Path(sp)
        song_id = sp.parent.name
        seg_name = sp.stem  # seg0
        key = f"{song_id}__{seg_name}"

        # ECAPA
        e = ecapa_embed(str(sp))
        torch.save(e, OUT_E / f"{key}.pt")

        # Pitch
        p = pitch_features(str(sp))
        if not p or not p.get("ok", True):
            # pitch_features 구현이 ok를 안 쓰면 이 줄은 무시되어도 됨
            pass
        with open(OUT_P / f"{key}.json", "w") as f:
            json.dump(p, f, indent=2)

        print("saved:", key)

    print("✅ build_features done")

if __name__ == "__main__":
    main()
