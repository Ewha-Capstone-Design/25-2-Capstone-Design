# src/index/profile.py
import torch

class UserProfile:
    def __init__(self, initial_emb: torch.Tensor, alpha=0.5, beta=0.1):
        """
        initial_emb: 초기 유저 목소리 벡터 (preprocess 완료된 것)
        alpha: Positive 피드백 반영률 (크면 좋아하는 곡 쪽으로 급격히 이동)
        beta: Negative 피드백 반영률
        """
        self.current_emb = initial_emb.clone().detach()
        self.alpha = alpha
        self.beta = beta

    def update(self, positives: list[torch.Tensor], negatives: list[torch.Tensor]):
        """
        좋아요(positives) / 싫어요(negatives) 벡터 리스트를 받아
        현재 유저 임베딩을 이동시킴.
        """
        if not positives and not negatives:
            return

        # Move towards positive
        if positives:
            # 여러 개일 경우 평균 벡터 계산
            pos_mean = torch.stack(positives).mean(dim=0)
            self.current_emb = self.current_emb + (self.alpha * pos_mean)

        # Move away from negative
        if negatives:
            neg_mean = torch.stack(negatives).mean(dim=0)
            self.current_emb = self.current_emb - (self.beta * neg_mean)

        # Re-normalize (방향성 유지가 중요하므로 크기 1로 맞춤)
        self.current_emb = self.current_emb / self.current_emb.norm(p=2)
        
        print(f"[Profile] User vector updated via Feedback (+{len(positives)}, -{len(negatives)})")

    def get_embedding(self):
        return self.current_emb