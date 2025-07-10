# Design chosen for the “Thousand-Columns Transformer” (TCT-v1)

The table below summarises why each decision was made before the full, self-contained PyTorch code.

| Sub-system                    | Choice                                 | Rationale (⇧ efficiency & quality, ⇩ effort)                                                  |
| ----------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Input fan-out**       | **Broadcast-all (1A)**           | Zero routing overhead; enables Flash-attention kernel reuse.                                    |
| **Column core**         | **RMSNorm + SwiGLU micro-FFN**   | Faster and more stable than LayerNorm + GELU; matches 2024-25 LLM best-practice.                |
| **Activation policy**   | Top-k gating (learned score)           | Proven sparse compute win; deterministic → reproducible latencies.                             |
| **Consensus bus**       | Single Flash Multi-Head Attention      | For N ≤ 8 k full attention is still cheaper than custom LSH*and* gets hardware acceleration. |
| **Message integration** | FiLM (scale-shift) conditioning        | Parameter-efficient + keeps information separate from internal memory.                          |
| **Read-out**            | Task-query attention over columns      | Lets downstream head pick the relevant sub-population dynamically.                              |
| **Parameter sharing**   | One shared core; no per-column weights | O(1) parameter footprint; easiest to maintain.                                                  |
| **Non-linearities**     | SiLU (in SwiGLU) only                  | Favoured in modern transformer stacks, hardware-friendly.                                       |

---
