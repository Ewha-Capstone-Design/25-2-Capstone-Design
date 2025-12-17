# Voice2Song — Team 32 올데이프로젝트

Voice2Song는 사용자의 직접 녹음한 MR/무반주 20~30초 보컬을 분석해 “내 목소리와 가장 어울리는 노래”를 추천하는 개인화 음악 추천 서비스입니다. 이 README는 외부 방문자가 프로젝트의 목적, 구성, 주요 기능, 기술 스택을 빠르게 파악할 수 있도록 일반적인 형식으로 작성되었습니다.

## 주요 정보
- **팀번호:** 32
- **팀명:** 올데이프로젝트
- **팀원:** 2271071 김민주, 2271049 이윤서, 2276202 유서연
- **레포지토리:** https://github.com/Ewha-Capstone-Design/25-2-Capstone-Design

## 프로젝트 개요
- **문제:** 기존 음악 추천은 청취 기록 중심이라 실제 가창 특성(음색/음역)을 반영하지 못함
- **해결:** Demucs MR 제거 → ECAPA-TDNN 음색 + Librosa PYIN 피치 추출 → segment-wise hybrid similarity → Max/Mean 집계 → Rocchio 피드백으로 개인화 재정렬
- **타깃 사용자:** 노래방/회식에서 곡 선택이 어려운 일반인, 음색에 맞는 곡을 찾고 싶은 취미 보컬리스트, ‘부르는 경험’을 확장하고 싶은 음악 소비자
- **기대 효과:** 음색·음역을 벡터화하여 객관적 추천 제공, Like/Dislike feedback으로 지속 개선, 보컬 트레이닝/오디션/커뮤니티 확장 가능성

## Tech Stack
- **AI/ML:** PyTorch, SpeechBrain ECAPA-TDNN, Librosa PYIN, FAISS
- **Audio tooling:** FFmpeg, Demucs, sounddevice, soundfile
- **Frontend:** React, TypeScript, Web Audio API
- **Backend:** Python 3.14 (FastAPI), Spring Boot, PostgreSQL, Redis

## 폴더 구조
```
25-2-Capstone-Design/
├── AI/          # Voice2Song 파이프라인 (data/, src/, README.md)
├── backend/     # Spring Boot, REST API
├── frontend/     # React/TypeScript 애플리케이션
├── docs/        # Ground rules, 문서
└── README.md     # 이 문서
```

## 빠른 시작
1. `cd 25-2-Capstone-Design`
2. `python -m pip install -r AI/requirements.txt`
3. 준비된 음원/녹음 또는 `AI/src/preprocess_user.py`로 30초 녹음
4. `python AI/src/preprocess.py`
5. `python AI/src/build_features.py`
6. `python AI/src/recommend_user.py data/user/user_record_*.m4a`
7. 개인화 시연: `python AI/src/run_personalization.py`

> **참고:** `AI/README.md`에서 세부 파이프라인, 점수 계산, describe_timbre 보조 설명 등 추가 정보를 확인하세요.

## 핵심 기술 
- **L-Stage:** Demucs + ECAPA + Pitch로 음악의 Timbre & Range를 벡터화합니다.
- **A-Stage:** `score.segment_score()`에서 `sim_ecapa()`/`sim_pitch()`를 w_e=0.35/w_p=0.65로 조합하고, Max-Mean(α=0.6) aggregation으로 곡 점수를 도출합니다.
- **B-Stage:** Like/Dislike 피드백을 Rocchio 방식으로 반영하며 FAISS에서 다시 검색해 순위를 개인화합니다.

## 참고 링크
- Tech blog: https://l0915ys.tistory.com/32 (Whisper→ECAPA pivot 배경)
- Ground rule: https://github.com/Ewha-Capstone-Design/25-2-Capstone-Design/blob/main/docs/GroundRule.md
