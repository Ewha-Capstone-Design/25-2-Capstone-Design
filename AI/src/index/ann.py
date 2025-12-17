# src/index/ann.py
import faiss
import torch
import numpy as np
from pathlib import Path
from glob import glob
import pickle

class VocalIndex:
    def __init__(self, feature_dir: Path, dimension=192):
        self.feature_dir = feature_dir
        self.dimension = dimension
        self.index = None
        # FAISS는 정수 ID만 반환하므로, int_id -> (song_id, seg_name) 매핑이 필요함
        self.meta_map = [] 

    def build(self):
        """ECAPA .pt 파일들을 로드하여 FAISS 인덱스 생성"""
        pt_files = glob(str(self.feature_dir / "*.pt"))
        if not pt_files:
            print("[ANN] No feature files found.")
            return

        vectors = []
        self.meta_map = []

        print(f"[ANN] Building index from {len(pt_files)} segments...")
        
        for pt in pt_files:
            pt = Path(pt)
            # key format: song_id__seg0
            key = pt.stem
            if "__" not in key:
                continue
            
            # Load vector
            emb = torch.load(pt).numpy()
            
            # Normalize for Cosine Similarity (L2 Norm)
            # Inner Product(IP) of normalized vectors == Cosine Similarity
            faiss.normalize_L2(emb.reshape(1, -1))
            
            vectors.append(emb)
            self.meta_map.append(key)

        # Convert to float32 matrix
        vectors = np.array(vectors).astype('float32')

        # Create Index (Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        
        print(f"[ANN] Index built. Total vectors: {self.index.ntotal}")

    def search(self, query_emb: torch.Tensor, top_k=50):
        """
        쿼리 벡터와 가장 유사한 세그먼트 top_k개를 반환
        Return: [{'key': 'song__seg', 'score': float}, ...]
        """
        if self.index is None:
            raise ValueError("Index not built yet.")

        # Query preprocessing
        q = query_emb.numpy().reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)

        distances, indices = self.index.search(q, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            key = self.meta_map[idx]
            results.append({"key": key, "sim": float(dist)})
            
        return results

    def save(self, path_prefix="data/index"):
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}_meta.pkl", "wb") as f:
            pickle.dump(self.meta_map, f)

    def load(self, path_prefix="data/index"):
        self.index = faiss.read_index(f"{path_prefix}.faiss")
        with open(f"{path_prefix}_meta.pkl", "rb") as f:
            self.meta_map = pickle.load(f)