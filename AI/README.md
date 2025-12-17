# 🎤 AI 기반 보컬 추천 파이프라인

기존 노래 음원에서 보컬 부분을 추출하고, ECAPA/피치 특성 비교를 통해 유저 음성과 잘 어울리는 곡을 추천하는 연구용 파이프라인입니다.

## 폴더 구조 요약 (AI)

```
AI/
├── data/
│   ├── songs_vocal/    # 원곡 mp3
│   ├── user/           # 유저 m4a + 기록 저장
│   ├── _wav/           # preprocess 결과(곡/유저 16k mono)
│   ├── _segments/      # 곡을 seg0~seg5로 자른 wav
│   ├── _features/      # ECAPA(.pt) + Pitch(.json) 특징 저장
│   └── _tmp/           # 중간 파일(녹음 등)
├── requirements.txt    # pip 라이브러리 의존
└── src/                # 핵심 실행 스크립트
    ├── preprocess*.py  # 음원 → 16k wav → 세그먼트/추천용
    ├── extract_*.py    # ECAPA / Pitch 피쳐 추출
    ├── build_features.py
    ├── recommend*.py   # 추천/사용자 추천 로직
    ├── score.py        # ECAPA+Pitch 조합 점수
    ├── describe_timbre.py
    ├── run_personalization.py  # FAISS + feedback 예시
    └── index/          # FAISS index + user profile
```

## 🧰 필수 도구

- `python3 -m pip install -r requirements.txt`
- 시스템 종속: `ffmpeg`, `demucs`, 마이크 접근(로컬 녹음인 경우), `sounddevice` 관련 드라이버
- `demucs`는 `pip install demucs` 후 `demucs` 명령이 PATH에 있어야 작동합니다.

## ⚙️ 기본 워크플로우

1. **원곡/유저 전처리** (`python src/preprocess.py`)
   - `data/songs_vocal/*.mp3` → `data/_wav/songs/*.wav` (16k mono) + 6개 세그먼트(`seg0~seg5`)
   - `data/user/*.m4a` → `data/_wav/user/user.wav`
2. **특징 추출** (`python src/build_features.py`)
   - 각 세그먼트에 ECAPA 벡터(.pt)와 Pitch 통계(.json) 저장 → `data/_features/{ecapa,pitch}`
3. **추천 실행** (`python src/recommend.py`)
   - `score.segment_score()`이 ECAPA(`sim_ecapa`) + Pitch(`sim_pitch`)를 `w_e=0.35`, `w_p=0.65` 비율로 조합
   - `α=0.6 × max + 0.4 × mean`으로 각 곡 점수 집계 → Top 10 출력

## 🧑‍🎤 유저 녹음 기반 플로우

1. **녹음 + MR 제거** (`python src/preprocess_user.py`)
   - 마이크로 30초 녹음 → `data/_tmp/user_raw_<ts>.wav` → Demucs로 MR 제거 → `data/user/user_record_N.m4a`
2. **추천** (`python src/recommend_user.py data/user/user_record_N.m4a`)
   - 기존 워크플로우와 동일한 방식의 ECAPA+Pitch 비교
   - `convert_to_user_wav()`가 입력을 16k mono로 맞춰주므로 다양한 오디오 포맷 허용

##  주요 스크립트/함수 역할

- `src/preprocess.py`: mp3/m4a → 16k mono wav로 변환하고, `songs_vocal` 음원은 6개 세그먼트(`seg0~seg5`)로 잘라서 `data/_segments`에 저장합니다.
- `src/build_features.py`: 각 세그먼트에 `extract_ecapa.ecapa_embed`와 `extract_pitch.pitch_features`를 실행하여 ECAPA `.pt`와 Pitch `.json` 파일을 `data/_features/{ecapa,pitch}`에 기록합니다.
- `src/recommend.py`: `data/_wav/user/user.wav`과 저장된 곡 세그먼트에 대해 `score.segment_score()`를 적용하고, `α × max + (1-α) × mean`으로 곡 점수를 계산하여 Top 10 추천을 출력합니다.
- `src/preprocess_user.py`: 로컬 마이크로 30초 녹음 → 임시 WAV(`data/_tmp/...`) → `demucs` MR 제거 → `data/user/user_record_N.m4a`로 출력합니다.
- `src/recommend_user.py`: 유저 녹음 결과 또는 기존 m4a를 `convert_to_user_wav()`로 16k mono로 맞춘 뒤 추천 파이프라인을 실행합니다.
- `src/score.py`: `sim_ecapa()`와 `sim_pitch()` 점수를 `w_e=0.35`, `w_p=0.65`로 섞어 세그먼트 유사도를 구하는 `segment_score()`를 제공합니다.
- `src/describe_timbre.py`: RMS/centroid/rolloff/MFCC 특징을 추출해 “밝기/고역/에너지/질감”으로 요약하고, 그 숫자와 경향을 함께 출력합니다.
- `src/run_personalization.py`: FAISS `VocalIndex`로 후보를 추려 `get_full_score()`로 Pitch까지 반영한 최종 점수를 계산하고, `UserProfile`로 피드백을 반영한 Round 2 개인화 추천까지 시뮬레이션합니다.
- `src/index/ann.py`: FAISS `IndexFlatIP`에 ECAPA 벡터를 넣고 cosine(inner-product) 유사도가 가장 높은 세그먼트를 빠르게 검색합니다.
- `src/index/profile.py`: 좋아요/싫어요 벡터 평균을 이용해 현재 유저 임베딩을 이동시켜 개인화 추천의 방향성을 바꿉니다.

##  고급: 개인화/FAISS

- `src/index/ann.py`: ECAPA 특징을 FAISS inner-product index에 올리고 similarity 서치
- `src/run_personalization.py`: 
  - FAISS로 후보군(Top 100) 추출 → `get_full_score()`로 Pitch까지 포함해 최종 점수 계산
  - `index/profile.py`의 `UserProfile`로 “좋아요/싫어요” 벡터 업데이트 → 임베딩 조정을 통해 Round 2 재검색

##  부가 정보

- **describe_timbre.py**: `librosa` 기반 RMS/centroid/rolloff/MFCC 특징을 추출해서 CPU에서 간단한 문장으로 음색 설명
- **score.segment_score()**에서 `w_e`, `w_p` 값을 조절하면 음색(ECAPA) vs 음역대(Pitch) 영향력 변경 가능
- **describe_timbre** 출력은 숫자와 경향을 같이 보여주도록 업데이트되어 어떤 지표를 기반으로 묘사했는지 확인 가능

##  실행 순서 예시

```bash
# 1. 원곡/유저 전처리 → 2. 특징 추출
python src/preprocess.py
python src/build_features.py

# 3. 유저 녹음/추천 (또는 기존 user.wav 사용)
python src/preprocess_user.py
python src/recommend_user.py data/user/user_record_*.m4a

# 개인화 실험
python src/run_personalization.py
```

## 팁

- `data/_features/{ecapa,pitch}`는 버전 관리 대상이지만 재생성 시 기존 `.pt`/`.json` 삭제해도 무방합니다.
- `score.segment_score()`나 `run_personalization.py` 안의 트레이드오프 파라미터(`w_e`, `w_p`, `alpha`, `UserProfile`의 `alpha/beta`)로 추천 성향을 튜닝하세요.
- 로그 수준을 높이고 싶으면 `describe_timbre.py`를 직접 호출해 `extract_timbre_features()` 반환값을 확인하거나 `print()`를 추가해 보세요.


