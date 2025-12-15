# 유사도 계산 함수
import math
import torch
import torch.nn.functional as F

def sim_ecapa(u: torch.Tensor, s: torch.Tensor) -> float:
    return float(F.cosine_similarity(u, s, dim=0).item())

def exp_sim(diff: float, tau: float) -> float:
    return math.exp(-abs(diff) / tau)

def sim_pitch(u: dict, s: dict) -> float:
    # tau 값은 데모용 기본값(튜닝 가능)
    tau_range = 6.0     # 반음 기준
    tau_med = 3.0       # 반음 기준(중앙 음역)
    tau_std = 1.5       # 반음 기준(안정성)

    # Hz -> semitone 변환 함수
    def hz_to_semi(hz: float) -> float:
        return 12.0 * math.log2(hz / 440.0)

    u_range = u["range_semitone"]
    s_range = s["range_semitone"]

    u_med = hz_to_semi(u["f0_median"])
    s_med = hz_to_semi(s["f0_median"])

    u_std = u["std_semitone"]
    s_std = s["std_semitone"]

    a = exp_sim(u_range - s_range, tau_range)
    b = exp_sim(u_med - s_med, tau_med)
    c = exp_sim(u_std - s_std, tau_std)

    # voiced_ratio는 품질 체크로 가볍게 반영
    q = min(u["voiced_ratio"], s["voiced_ratio"])
    return 0.4*a + 0.4*b + 0.2*c * q

def segment_score(u_emb, u_pitch, s_emb, s_pitch, w_e=0.75, w_p=0.25):
    se = sim_ecapa(u_emb, s_emb)
    sp = sim_pitch(u_pitch, s_pitch)
    return w_e * se + w_p * sp, se, sp
