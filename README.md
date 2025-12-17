# Voice2Song — Team 32 올데이프로젝트

Voice2Song는 ML 이론은 이해했지만 실습/코딩 경험이 부족한 사용자를 위해 디자인된 개인화 음악 추천 플랫폼으로, 사용자가 MR/무반주 20~30초 보컬을 녹음하면 음색·음역·구간 특징을 분석해 “내가 불렀을 때 가장 잘 어울리는 곡”을 찾아줍니다. BuildX처럼 비전공자도 흐름을 따라올 수 있도록 UI/UX와 파이프라인이 결합된 구조를 지향하며, 이 README는 프로젝트의 목적과 구성을 알기 쉽게 정리해 둔 일반적인 안내서입니다.

## 프로젝트 개요

- **문제 정의:** 기존 추천 시스템은 청취 로그 중심으로, 사용자의 음색/음역/가창 경험을 반영하지 않습니다.
- **목표:** Demucs로 MR 제거 → ECAPA-TDNN 음색 임베딩 + Librosa PYIN pitch → ECAPA/Pitch 혼합 유사도 → Max/Mean aggregation → Rocchio로 피드백 반영한 재정렬을 통해 “가창 맞춤” 추천을 제공합니다.
- **타깃 사용자:** 노래방/회식에서 곡 선택이 어려운 일반 사용자, 자신의 음색에 맞는 곡을 찾고 싶은 취미 보컬리스트, ‘부르는 경험’을 확장하려는 음악 소비자 등.
- **핵심 효과:** 음색/음역을 벡터화하여 객관적인 추천을 제공하고, Like/Dislike 피드백과 실시간 로그를 활용한 개인화가 가능합니다.

## 팀 정보

- **팀번호:** 32  
- **팀명:** 올데이프로젝트  
- **팀원:** 김민주(2271071, Backend/DevOps), 이윤서(2271049, AI/Signal Processing), 유서연(2276202, Frontend/Data Viz)  
- **지도교수:** 심재형 교수님  
- **레포지토리:** https://github.com/Ewha-Capstone-Design/25-2-Capstone-Design

## Repository 구성

| 디렉터리 | 설명 |
| --- | --- |
| `AI/` | 음원 전처리, 세그먼트→ECAPA/Pitch 추출, 추천/개인화 파이프라인  |
| `frontend/` | React + TypeScript 기반 UI (블록형/웹 오디오 녹음) |
| `backend/` | FastAPI + Spring Boot API (모델 실행, USER profile 관리) |
| `docs/` | Ground rule/기획자료 |

## 기술 스택

- **Frontend:** React, TypeScript, Web Audio API, describe_timbre 템플릿
- **Backend:** Python 3.14 (FastAPI) + Spring Boot, PostgreSQL, Redis, Celery-style job handling
- **AI/ML:** PyTorch, SpeechBrain ECAPA-TDNN, Librosa PYIN, FAISS, score.py hybrid scoring
- **Audio tooling:** FFmpeg, Demucs, sounddevice, soundfile
- **Infra:** Docker, Docker Compose, AWS (EC2/Nginx), GPT API for feedback

## How to Build & Run

- **필수 패키지 설치:** AI 서브모듈로 이동하여 `python -m pip install -r AI/requirements.txt` 실행  
- **데이터 준비:** `AI/data/songs_vocal/`에 음원, `AI/data/user/`에 유저 녹음을 두거나 `AI/src/preprocess_user.py`로 30초 샘플을 녹음  
- **전처리 및 특징 추출:** 
  ```bash
  python AI/src/preprocess.py
  python AI/src/build_features.py
  ```
- **추천 실행:**  (유저의 목소리 기반 추천)
  ```bash
  python AI/src/recommend_user.py data/user/user_record_*.m4a
  ```
- **개인화 시연:**  (like, dislike 반영된 개인화 추천)
  ```bash
  python AI/src/run_personalization.py
  ```


## 핵심 기능

- **L-Stage(특징 추출):** Demucs로 MR 제거 후 ECAPA와 Librosa PYIN 기반 피치/통계 정보를 30초 세그먼트(총 6개)에 대해 저장  
- **A-Stage(유사도):** `score.segment_score()`에서 `sim_ecapa()`와 `sim_pitch()`를 w_e=0.35/w_p=0.65로 결합하고, Max(α=0.6)/Mean 으로 곡 점수를 집계  
- **B-Stage(피드백):** Rocchio 스타일로 `UserProfile` 벡터를 업데이트하고 FAISS에서 재검색하여 순위를 개인화  
- **Describe timbre:** `AI/src/describe_timbre.py`는 RMS·centroid·rolloff·MFCC 값을 추출해 사용자가 읽기 쉬운 설명과 함께 수치도 로깅

## Sample data & DB

- **학습용 데이터:** `AI/data/_segments/`에 저장된 세그먼트 
- **임베딩/피치:** `AI/data/_features/ecapa/*.pt`, `AI/data/_features/pitch/*.json`으로 기록

## Troubleshooting & Notes

- ECAPA 추론이 실패하면 `requirements.txt`에서 torchaudio/shape 버전을 확인하고 `python src/extract_ecapa.py <파일>`로 개별 점검하세요.  
- 밝기/음역 비교를 위해 `AI/src/describe_timbre.py`를 실행하면 도움이 됩니다.  
- `AI/src/score.py`에서 `w_e`, `w_p`, `alpha` 값을 조정하여 유사도 가중치를 튜닝하세요.  
- MR 제거가 제대로 되지 않으면 유저 오디오를 다시 녹음하거나 `AI/data/_tmp/`를 정리하세요.

## 참고

- Ground rules: https://github.com/Ewha-Capstone-Design/25-2-Capstone-Design/blob/main/docs/GroundRule.md  
