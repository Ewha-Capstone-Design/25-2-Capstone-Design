# ECAPA 특징 추출 함수
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier
from pathlib import Path

# --- [긴급 패치] Torchaudio 2.9+ 호환성 문제 해결 ---
# speechbrain이 옛날 함수(list_audio_backends)를 찾을 때 에러가 나지 않도록 가짜 함수를 넣어줌
if not hasattr(torchaudio, "list_audio_backends"):
    def _shim_list_audio_backends():
        return ["soundfile"]  # 윈도우에서 기본으로 쓰는 백엔드
    torchaudio.list_audio_backends = _shim_list_audio_backends
# ------------------------------------------------

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)

def load_wav_16k_mono(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav

def ecapa_embed(path):
    wav = load_wav_16k_mono(path)
    with torch.no_grad():
        emb = classifier.encode_batch(wav).squeeze()
    emb = emb / emb.norm(p=2)
    return emb

def save_ecapa(path_wav, out_dir="data/embeddings/ecapa"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = ecapa_embed(path_wav)
    out_path = out_dir / (Path(path_wav).stem + ".pt")
    torch.save(emb, out_path)

    print(f"[ECAPA] saved → {out_path}")
    print(" shape:", emb.shape)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_ecapa.py path/to/audio.wav")
        sys.exit(1)

    save_ecapa(sys.argv[1])
