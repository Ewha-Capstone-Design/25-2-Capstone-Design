# 🎤 AI 기반 보컬 추천 파이프라인

기존 노래 음원에서 보컬 부분을 추출하고, ECAPA/피치 특성 비교를 통해 유저 음성과 잘 어울리는 곡을 추천하는 연구용 파이프라인입니다. 이 README에서는 내부 AI 파이프라인의 구성과 실행 워크플로우를 소개합니다.

## 📦 폴더 구조 요약 (AI)
```
AI/
├── data/
│   ├── songs_vocal/    # 원곡 mp3
│   ├── user/           # 유저 m4a + 녹음 저장
│   ├── _wav/           # preprocess 결과(곡/유저 16k mono)
│   ├── _segments/      # 곡을 seg0~seg5로 자른 wav
│   ├── _features/      # ECAPA(.pt) + Pitch(.json) 특징 저장
│   └── _tmp/           # 중간 파일(녹음 등)
├── requirements.txt
└── src/
    ├── preprocess*.py
    ├── extract_*.py
    ├── build_features.py
    ├── recommend*.py
    ├── score.py
    ├── describe_timbre.py
    ├── run_personalization.py
    └── index/
```

## 🧰 필수 도구
- `python3 -m pip install -r requirements.txt`
- 시스템 종속: `ffmpeg`, `demucs`, 마이크 접근 권한
- `demucs`는 CLI로 설치되어 있어야 합니다.

## ⚙️ 기본 워크플로우
1. `python src/preprocess.py` (songs/user → 16k wav + segments)
2. `python src/build_features.py` (segments → ECAPA(.pt) + Pitch(.json))
3. `python src/recommend.py` 또는 `python src/recommend_user.py` (ECAPA+Pitch 유사도 기반 추천)

## 🧑‍🎤 유저 녹음 기반 플로우
1. `python src/preprocess_user.py` (30초 녹음 → Demucs MR 제거 → data/user/user_record_N.m4a)
2. `python src/recommend_user.py data/user/user_record_N.m4a`

## 🧠 개인화/FAISS
- `src/index/ann.py`: ECAPA 벡터로 FAISS IndexFlatIP 인덱스를 빌드하고 cosine 유사도로 검색
- `src/run_personalization.py`: FAISS → `get_full_score()`로 Pitch 포함 점수 → `UserProfile` 피드백 적용 후 Round 2 재검색

## 💡 부가 정보
- `src/describe_timbre.py`: RMS/centroid/rolloff/MFCC 기반 간단한 음색 설명을 생성
- `src/score.py`: `sim_ecapa()`/`sim_pitch()`를 `w_e=0.35`, `w_p=0.65`로 조합하는 `segment_score()`
- 중요 파라미터: `score.segment_score()`의 `w_e`, `w_p`, `run_personalization.py`의 `alpha`, `UserProfile`의 `alpha`, `beta`

## 🧪 실행 예시
```bash
cd AI
python src/preprocess.py
python src/build_features.py
python src/preprocess_user.py
python src/recommend_user.py data/user/user_record_*.m4a
python src/run_personalization.py
```

## 🗃️ 팁
- `data/_features/{ecapa,pitch}`는 필요에 따라 삭제 후 재생성 가능합니다.
- `describe_timbre.py`로 extract한 지표를 확인해 튜닝에 활용하세요.
