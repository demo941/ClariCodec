import torch
from typing import Optional, Dict, Any

class CodebookStatsAccumulator:
    """
    Accumulate code usage statistics across many utterances / batches.

    Metrics:
      - coverage = (#codes used at least once) / K
      - entropy_bits = -sum_i p_i log2 p_i
      - perplexity = 2^{entropy_bits}  (effective number of used codes)
      - eff_utilization = perplexity / K
    """
    def __init__(self, codebook_size: int, num_quantizers: Optional[int] = None, quantizer_dim: Optional[int] = None):
        self.K = int(codebook_size)
        self.num_quantizers = num_quantizers
        self.quantizer_dim = quantizer_dim
        self.counts_overall = torch.zeros(self.K, dtype=torch.long)
        self.counts_per_q = None  # lazily initialized if we can infer quantizer dimension

    @torch.no_grad()
    def update(self, indices: torch.Tensor):
        """
        indices: torch.LongTensor, any shape.
                 Can be single-layer VQ indices or multi-layer RVQ indices.
                 Negative indices (e.g. -1 for dropout) are ignored.
        """
        if not torch.is_tensor(indices):
            raise TypeError(f"indices must be a torch.Tensor, got {type(indices)}")
        if indices.dtype != torch.long:
            indices = indices.long()

        # move to cpu for cheap accumulation
        idx = indices.detach().to("cpu")

        # ignore invalid indices (e.g., -1 from quantize dropout)
        valid = (idx >= 0) & (idx < self.K)
        idx_valid = idx[valid]
        if idx_valid.numel() == 0:
            return

        # overall counts
        self.counts_overall += torch.bincount(idx_valid.reshape(-1), minlength=self.K)

        qdim = self.quantizer_dim
        if qdim is None:
            return

        # initialize per-q buffers
        Q = idx.shape[qdim]
        if self.counts_per_q is None or len(self.counts_per_q) != Q:
            self.counts_per_q = [torch.zeros(self.K, dtype=torch.long) for _ in range(Q)]

        # move quantizer dim to front, then iterate
        idx_qfirst = idx.movedim(qdim, 0)  # [Q, ...]
        for q in range(Q):
            idx_q = idx_qfirst[q]
            v = (idx_q >= 0) & (idx_q < self.K)
            idx_q_valid = idx_q[v]
            if idx_q_valid.numel() == 0:
                continue
            self.counts_per_q[q] += torch.bincount(idx_q_valid.reshape(-1), minlength=self.K)

    def compute(self) -> Dict[str, Any]:
        out = {"overall": self._compute_from_counts(self.counts_overall, prefix="overall")}
        if self.counts_per_q is not None:
            out["per_quantizer"] = [
                self._compute_from_counts(c, prefix=f"q{qi}") for qi, c in enumerate(self.counts_per_q)
            ]
        return out

    def _compute_from_counts(self, counts: torch.Tensor, prefix: str) -> Dict[str, float]:
        counts = counts.to(torch.long)
        total = int(counts.sum().item())
        if total <= 0:
            return {
                "codebook_size": float(self.K),
                "total_tokens": 0.0,
                "active_tokens":0.0,
                "coverage": 0.0,
                "entropy_bits": 0.0,
                "perplexity": 0.0,
                "eff_utilization": 0.0,
            }

        active = int((counts > 0).sum().item())
        coverage = active / self.K

        p = counts[counts > 0].double() / float(total)
        entropy_bits = float(-(p * torch.log2(p)).sum().item())
        perplexity = float(2.0 ** entropy_bits)  # effective number of used codes
        eff_utilization = perplexity / self.K

        return {
            "codebook_size": float(self.K),
            "total_tokens": float(total),
            "active_tokens":float(active),
            "coverage": float(coverage),
            "entropy_bits": float(entropy_bits),
            "perplexity": float(perplexity),
            "eff_utilization": float(eff_utilization),
        }