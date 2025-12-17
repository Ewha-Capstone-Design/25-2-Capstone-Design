# src/run_personalization.py
import sys
import torch
import torchaudio
import ast
import soundfile  # 직접 로딩을 위해 추가

# ==========================================
# [최종 해결책] Python 3.14 & 호환성 패치 (유지)
# ==========================================

# 1. [SpeechBrain] 사라진 list_audio_backends 함수 복구
if not hasattr(torchaudio, "list_audio_backends"):
    def _shim_list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = _shim_list_audio_backends

# 2. [HyperPyYAML] Python 3.14에서 사라진 ast.Num 복구
try:
    ast.Num
except AttributeError:
    ast.Num = ast.Constant
    ast.Str = ast.Constant
    ast.NameConstant = ast.Constant

# 3. [Torchaudio] load 함수 완전 대체 (Direct Bypass)
def _manual_audio_load(filepath, *args, **kwargs):
    data, sr = soundfile.read(filepath)
    tensor = torch.tensor(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = _manual_audio_load
# ==========================================

from pathlib import Path
import json

# 기존 모듈 import
from extract_ecapa import ecapa_embed
from extract_pitch import pitch_features
from score import segment_score

# B가 만든 모듈 import
from index.ann import VocalIndex
from index.profile import UserProfile

ROOT = Path(__file__).resolve().parents[1]
FEATURE_E = ROOT / "data" / "_features" / "ecapa"
FEATURE_P = ROOT / "data" / "_features" / "pitch"
USER_WAV = ROOT / "data" / "_wav" / "user" / "user.wav"

def get_full_score(u_emb, u_pitch, candidate_keys):
    """
    ANN으로 추려진 후보(candidate_keys)들에 대해
    Pitch까지 포함한 정밀 스코어(Max/Mean) 계산
    """
    candidate_songs = set(k.split("__")[0] for k in candidate_keys)
    
    final_results = []
    alpha = 0.6  # Max weight

    for song_id in candidate_songs:
        seg_scores = []
        best_detail = None
        
        for i in range(6):
            key = f"{song_id}__seg{i}"
            pt_path = FEATURE_E / f"{key}.pt"
            js_path = FEATURE_P / f"{key}.json"

            if not pt_path.exists() or not js_path.exists():
                continue

            s_emb = torch.load(pt_path)
            with open(js_path, 'r') as f:
                s_pitch = json.load(f)

            # 점수 계산
            total, se, sp = segment_score(u_emb, u_pitch, s_emb, s_pitch)
            seg_scores.append((total, se, sp))
        
        if not seg_scores:
            continue

        scores_only = [x[0] for x in seg_scores]
        max_s = max(scores_only)
        mean_s = sum(scores_only) / len(scores_only)
        
        final_score = alpha * max_s + (1 - alpha) * mean_s
        
        best_seg_idx = scores_only.index(max_s)
        best_detail = seg_scores[best_seg_idx]

        final_results.append({
            "song_id": song_id,
            "score": final_score,
            "best_seg": best_detail
        })

    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results

def print_top_k(title, results, k=3):
    print(f"\n=== {title} (Top {k}) ===")
    for i, res in enumerate(results[:k], 1):
        bd = res['best_seg']
        print(f"{i}. {res['song_id']} | Score: {res['score']:.3f} "
              f"(Best Seg: Tot={bd[0]:.2f}, Ecapa={bd[1]:.2f}, Pitch={bd[2]:.2f})")
    return results[:k]

def main():
    if not USER_WAV.exists():
        print("User wav not found. Run preprocess.py first.")
        return

    # 1. Init: 유저 특징 추출
    print(">>> 1. Loading User Features...")
    u_base_emb = ecapa_embed(str(USER_WAV))
    u_pitch = pitch_features(str(USER_WAV))
    
    # 2. Build Index (ANN)
    print(">>> 2. Building FAISS Index...")
    v_index = VocalIndex(FEATURE_E)
    v_index.build()

    # 3. User Profile 생성
    user_profile = UserProfile(u_base_emb, alpha=0.4, beta=0.2)

    # --- Round 1: 초기 검색 ---
    print("\n>>> [Round 1] Searching with raw voice...")
    current_u_vec = user_profile.get_embedding()
    
    candidates = v_index.search(current_u_vec, top_k=100)
    candidate_keys = [c['key'] for c in candidates]
    
    r1_results = get_full_score(current_u_vec, u_pitch, candidate_keys)
    top_songs = print_top_k("Round 1 Recommendations", r1_results)

    if not top_songs:
        return

    # --- Interaction: 사용자 피드백 시뮬레이션 ---
    if len(top_songs) >= 3:
        # [수정됨] 1위가 아니라 3위를 '싫어요', 1위를 '좋아요' 등으로 예시 변경 가능
        # 에러가 났던 이유는 파일명에 대괄호 '[]'가 있어서 glob이 오작동했기 때문입니다.
        disliked_song = top_songs[0]['song_id']
        liked_song    = top_songs[2]['song_id']
        
        print(f"\n>>> [Feedback] User dislikes '{disliked_song}' but likes '{liked_song}'")

        def get_song_vec(sid):
            # [버그 수정] glob 대신 안전하게 파일 찾기
            # glob(f"{sid}...")는 sid에 대괄호[]가 있으면 에러가 납니다.
            target_prefix = f"{sid}__seg"
            found = None
            for p in FEATURE_E.glob("*.pt"):
                if p.name.startswith(target_prefix):
                    found = p
                    break # 하나만 찾으면 됨
            
            if found:
                return torch.load(found)
            else:
                print(f"Warning: Could not find feature file for {sid}")
                return current_u_vec # 에러 방지용으로 유저 벡터 리턴

        vec_like = [get_song_vec(liked_song)]
        vec_dislike = [get_song_vec(disliked_song)]

        user_profile.update(positives=vec_like, negatives=vec_dislike)

        # --- Round 2: 피드백 반영 후 재검색 ---
        print("\n>>> [Round 2] Re-searching with personalized profile...")
        new_u_vec = user_profile.get_embedding()
        
        candidates_v2 = v_index.search(new_u_vec, top_k=100)
        keys_v2 = [c['key'] for c in candidates_v2]
        r2_results = get_full_score(new_u_vec, u_pitch, keys_v2)
        
        print_top_k("Round 2 (Personalized) Recommendations", r2_results)
        
        r1_top = r1_results[0]['song_id']
        r2_top = r2_results[0]['song_id']
        print(f"\n[Summary] Top Recommendation changed: {r1_top} -> {r2_top}")

    else:
        print("Not enough results to simulate feedback.")

if __name__ == "__main__":
    main()