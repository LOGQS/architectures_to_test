# tct.py ───────────────────────────────────────────────────────────────────


from __future__ import annotations
import torch, math
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Core layers
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Simpler/faster norm than LayerNorm (no per-feature mean subtraction)."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm

class SwiGLU(nn.Module):
    """SwiGLU: (a * SiLU(b))."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 2 * d_out, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return a * F.silu(b)

class ColumnCore(nn.Module):
    """
    Shared core executed by every column.
    Input: concat[h, x, m] → RMSNorm → SwiGLU → linear proj → residual add.
    """
    def __init__(self, d_state: int, d_input: int, d_msg: int, mult: int = 4):
        super().__init__()
        self.norm   = RMSNorm(d_state + d_input + d_msg)
        self.ff_g   = SwiGLU(d_state + d_input + d_msg, mult * d_state)
        self.ff_out = nn.Linear(mult * d_state, d_state, bias=False)
    def forward(self, h: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        z = self.norm(torch.cat([h, x, m], dim=-1))
        h_delta = self.ff_out(self.ff_g(z))
        return h + h_delta        # residual

# ─────────────────────────────────────────────────────────────────────────────
# Thousand-Columns Transformer
# ─────────────────────────────────────────────────────────────────────────────
class ThousandColumns(nn.Module):
    """
    Minimal flat TCT block.
    Args
    ----
    N          : number of columns
    d_state    : per-column hidden size
    d_input    : raw input dimension
    d_vote     : vote / message dimension
    active_pct : fraction of columns to update each step
    """
    def __init__(
        self,
        N: int = 4096,
        d_state: int = 64,
        d_input: int = 256,
        d_vote: int = 64,
        active_pct: float = 0.25,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.N        = N
        self.k_active = max(1, int(N * active_pct))
        self.d_state  = d_state
        self.d_vote   = d_vote

        # Shared sub-modules
        self.in_proj   = nn.Linear(d_input, d_state, bias=False)
        self.core      = ColumnCore(d_state, d_state, d_vote)
        self.gate_proj = nn.Linear(d_state, 1, bias=False)  # gating score
        self.vote_proj = nn.Linear(d_state, d_vote, bias=False)
        self.bus_attn  = nn.MultiheadAttention(
            embed_dim=d_vote,
            num_heads=4,
            batch_first=True,
            bias=False  # flashes faster without bias
        )
        # FiLM parameters for message conditioning
        self.film_scale = nn.Linear(d_vote, d_state, bias=False)
        self.film_shift = nn.Linear(d_vote, d_state, bias=False)

        # Read-out
        self.query = nn.Parameter(torch.randn(1, 1, d_state))
        self.out_head = nn.Sequential(
            RMSNorm(d_state),
            nn.Linear(d_state, d_input, bias=False)   # placeholder task head
        )

        # Persistent hidden state (B,N,d_state) - dtype will match model parameters
        self.register_buffer(
            "state",
            torch.zeros(1, N, d_state, device=device),
            persistent=True
        )

    # ------------------------------------------------------------------ #
    def reset_state(self, B: int):
        with torch.no_grad():
            self.state = torch.zeros(
                B, self.N, self.d_state,
                dtype=self.state.dtype,
                device=self.state.device
            )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,d_input) or (B,d_input)
        returns y: (B,T,d_input)  (change head for your task)
        """
        if x.dim() == 2:
            x = x[:, None, :]
        B, T, _ = x.shape
        if self.state.shape[0] != B:
            self.reset_state(B)

        h = self.state.clone()
        outputs = []
        for t in range(T):
            # Project & broadcast input
            x_t = self.in_proj(x[:, t])               # (B,d_state)
            x_t = x_t[:, None, :].repeat(1, self.N, 1)

            # ── Gating (differentiable top-k) ──────────────────────────
            gate_scores = self.gate_proj(h).squeeze(-1)      # (B,N)
            
            # Straight-through estimator for differentiable top-k
            topk_val, topk_idx = torch.topk(gate_scores, self.k_active, dim=-1, sorted=False)
            
            # Hard mask for forward pass
            mask_hard = torch.zeros_like(gate_scores, dtype=gate_scores.dtype)
            mask_hard.scatter_(dim=-1, index=topk_idx, value=1.0)
            
            # Soft mask for gradients
            mask_soft = torch.softmax(gate_scores, dim=-1)
            
            # Straight-through: hard forward, soft backward
            mask_scores = mask_hard + (mask_soft - mask_soft.detach())
            
            # Create both boolean and float versions
            mask_bool = torch.zeros_like(gate_scores, dtype=torch.bool)
            mask_bool.scatter_(dim=-1, index=topk_idx, value=True)
            mask = mask_bool                    # use this for where / padding
            mask_scores_exp = (mask_bool.float() + 
                              (mask_soft - mask_soft.detach())).unsqueeze(-1)
            mask_exp = mask_bool.unsqueeze(-1)  # keep boolean version for where

            # ── Local update (active columns only) ───────────────────
            if self.training and hasattr(self, '_optimize_active') and self._optimize_active:
                # Efficient: only process active columns
                active_indices = mask.nonzero(as_tuple=False)  # (num_active, 2) -> (batch, column)
                if active_indices.numel() > 0:
                    batch_idx, col_idx = active_indices[:, 0], active_indices[:, 1]
                    h_active = h[batch_idx, col_idx]  # (num_active, d_state)
                    x_active = x_t[batch_idx, col_idx]  # (num_active, d_state)
                    m_active = torch.zeros(len(batch_idx), self.d_vote, dtype=h.dtype, device=h.device)
                    
                    # Process only active columns
                    h_active_new = self.core(h_active[:, None], x_active[:, None], m_active[:, None]).squeeze(1)
                    
                    # Scatter back to full tensor
                    h = h.clone()
                    h[batch_idx, col_idx] = h_active_new
            else:
                # Original: process all columns (for compatibility)
                m_dummy = torch.zeros(B, self.N, self.d_vote, dtype=h.dtype, device=h.device)
                h_hat = self.core(h, x_t, m_dummy)
                # Use float mask for gradient flow
                h = h + mask_scores_exp * (h_hat - h)            # soft gating preserves gradients

            # ── Votes ────────────────────────────────────────────────
            v = self.vote_proj(h)                            # (B,N,d_vote)
            # Use float mask for gradient flow  
            v = v * mask_scores_exp                          # soft masking preserves gradients

            # ── Consensus bus (optimized MHA) ────────────────────────
            if self.training and hasattr(self, '_optimize_active') and self._optimize_active:
                # Efficient: only process active vectors
                active_indices = mask.nonzero(as_tuple=False)  # (num_active, 2)
                if active_indices.numel() > 0:
                    batch_idx, col_idx = active_indices[:, 0], active_indices[:, 1]
                    v_active = v[batch_idx, col_idx]  # (num_active, d_vote)
                    
                    # Group by batch for attention
                    m = torch.zeros_like(v)
                    for b in range(B):
                        batch_mask = batch_idx == b
                        if batch_mask.any():
                            v_b = v_active[batch_mask]  # (k_active_b, d_vote)
                            if v_b.shape[0] > 0:
                                # Self-attention on active vectors only
                                m_b, _ = self.bus_attn(v_b[None], v_b[None], v_b[None])  # (1, k_active_b, d_vote)
                                m[b, col_idx[batch_mask]] = m_b.squeeze(0)
                else:
                    m = torch.zeros_like(v)
            else:
                # Original: full attention with padding mask
                m, _ = self.bus_attn(v, v, v, key_padding_mask=~mask)    # (B,N,d_vote)

            # ── FiLM integration ─────────────────────────────────────
            scale = 1 + torch.tanh(self.film_scale(m))
            shift = self.film_shift(m)
            h = h * scale + shift

            # ── Read-out (query-attention) ───────────────────────────
            q = self.query.expand(B, -1, -1)                 # (B,1,d_state)
            attn_logits = torch.matmul(q, h.transpose(1, 2)) / math.sqrt(self.d_state)
            attn = attn_logits.softmax(-1)                   # (B,1,N)
            pooled = torch.matmul(attn, h).squeeze(1)        # (B,d_state)
            y_t = self.out_head(pooled)
            outputs.append(y_t)

        # Update persistent state (preserve buffer integrity)
        with torch.no_grad():
            self.state = h
        return torch.stack(outputs, dim=1)                   # (B,T,d_input)

# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(
        N=4096,
        d_state=64,
        d_input=256,
        d_vote=64,
        active_pct=0.2,
        device=dev
    ).to(dev).eval()

    dummy = torch.randn(2, 8, 256, device=dev)
    with torch.autocast(dev):
        out = model(dummy)
    print("output shape:", out.shape)  # expected (2,8,256)