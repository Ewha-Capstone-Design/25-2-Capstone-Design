from pathlib import Path
import numpy as np
import librosa
from typing import Optional, Dict

def _percentile_rank(x, ref_vals):
    # ref 분포 대비 대략적인 위치(0~100)
    return float(np.mean(ref_vals <= x) * 100.0)

def extract_timbre_features(audio_path: str, sr: int = 16000):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # 무음 제거(가벼운 방식)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < sr * 1.0:
        return None  # 너무 짧으면 설명 불가

    # 에너지
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # 밝기/고역 비중(스펙트럼 중심)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))

    # 고역 비중(rolloff)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_mean = float(np.mean(rolloff))

    # 음색 질감(MFCC 평균)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1).astype(float)

    return {
        "rms": rms,
        "centroid_mean": centroid_mean,
        "rolloff_mean": rolloff_mean,
        "mfcc_mean": mfcc_mean.tolist(),
        "duration_sec": len(y)/sr,
    }

def describe_timbre(feat: Dict, ref: Optional[Dict] = None) -> str:
    """
    ref: 선택. 기준 분포(곡 데이터/유저 데이터)로 퍼센타일을 계산해 더 자연스러운 문장 생성 가능.
         MVP에서는 None으로 둬도 동작.
    """
    rms = feat["rms"]
    c = feat["centroid_mean"]
    r = feat["rolloff_mean"]

    # 기준값(임시). 가능하면 너희 songs_vocal 전체에서 분포를 만들어 ref로 넣는 게 가장 좋음.
    # ref 없을 때는 경험적 범위로 대략 분류
    # (이 범위는 환경에 따라 변하니 "대략"이라는 표현을 쓰는 게 안전)
    brightness = "중간"
    if c > 2500:
        brightness = "밝은 편"
    elif c < 1800:
        brightness = "어두운 편"

    high_end = "중간"
    if r > 6000:
        high_end = "고역 성분이 많은 편"
    elif r < 4500:
        high_end = "고역 성분이 적은 편"

    energy = "중간"
    if rms > 0.06:
        energy = "에너지가 강한 편"
    elif rms < 0.03:
        energy = "부드럽고 약한 편"

    # 아주 간단한 질감 추정(과장 금지: ‘경향’으로 표현)
    # MFCC[1]~[3]가 크면 거칠다/선명하다를 암시할 수 있으나 절대 단정 X
    mf = np.array(feat["mfcc_mean"])
    texture_score = float(np.mean(np.abs(mf[1:4])))
    texture = "중간 정도의 질감"
    if texture_score > 40:
        texture = "선명하고 또렷한 질감 경향"
    elif texture_score < 25:
        texture = "부드럽고 둥근 질감 경향"

    return (
        "음색 지표(값 기준): "
        f"밝기(스펙트럼 중심: {c:.0f} Hz) → {brightness}, "
        f"고역 성분(스펙트럼 롤오프: {r:.0f} Hz) → {high_end}, "
        f"에너지(RMS: {rms:.3f}) → {energy}. "
        f"질감(MFCC 2~4 평균 절댓값: {texture_score:.1f}) → {texture}."
    )

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/describe_timbre.py path/to/audio.(m4a|wav|mp3)")
        return

    audio = sys.argv[1]
    feat = extract_timbre_features(audio)
    if feat is None:
        print("오디오가 너무 짧거나 무음이 많아 음색 설명을 만들기 어렵습니다.")
        return

    text = describe_timbre(feat)
    print(text)

if __name__ == "__main__":
    main()
