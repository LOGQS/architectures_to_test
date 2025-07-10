# Thousand-Columns Transformer (TCT) 🧩

> A proof-of-concept **PyTorch** implementation of a “thousand brains” style network:
> thousands of small, stateful columns that talk to each other, reach consensus, and remember what they learned.

---

## 1 Why bother? 🧠

* **Persistent memory** – each column keeps its own latent state `hᵢ` across timesteps.
* **Sparse compute** – only a top-k subset updates every tick; you pay O(k), not O(N).
* **Consensus bus** – multi-head attention lets active columns exchange short vote vectors and agree (or disagree).
* **Single shared core** – one micro-FFN (RMSNorm → SwiGLU) serves every column; parameter count is O(1).

Put together, you get a network that **scales capacity with memory, not weights**, and can in principle handle very long contexts or sensorimotor loops.

---

## 2 Current status 🔍

* **Works end-to-end** for BERT/GPT-style one-tick training.
* **All tests pass** (`python tct_test.py`) on CPU and GPU.
* **No training recipe yet.** You’ll need to wrap the block in a multi-tick loop and plug in your own loss.
* **Performance knobs**

  * `model._optimize_active = True` – skip inactive columns (saves time & memory).
  * `active_pct` – fraction of columns updated each tick (default 25 %).

Expect rough edges; this is a research toy, not production code.

---

## 3 Quick start ⚙️

```bash
git clone https://github.com/LOGQS/architectures_to_test.git
cd columnar_transformer
pip install torch     
python tct.py             # runs a tiny smoke-test
pytest tct_test.py        # full test suite
```

Minimal usage:

```python
from tct import ThousandColumns
import torch

model = ThousandColumns(N=4096, d_state=64, d_input=256).cuda().eval()
dummy = torch.randn(2, 8, 256, device='cuda')
with torch.no_grad():
    out = model(dummy)          # (2, 8, 256)
```

Need iterative refinement? Wrap the block:

```python
for _ in range(n_ticks):
    y = model(x)        # state is sticky; call reset_state() when you’re done
```

---

## 4 Design snapshot 📐

| Sub-system          | Choice                                  | Why                                        |
| ------------------- | --------------------------------------- | ------------------------------------------ |
| Column core         | **RMSNorm + SwiGLU**              | Stable, fast, matches 2024-25 LLM practice |
| Activation policy   | **Straight-through top-k gating** | Deterministic sparsity, gradients flow     |
| Consensus bus       | **Flash MHA on vote vectors**     | Hardware-accelerated ≤ 8 k columns        |
| Message integration | **FiLM scale-shift**              | Lightweight conditioning                   |
| Read-out            | **Query attention over columns**  | Downstream task chooses what it needs      |

A fuller design-space catalogue lives in `docs/`.

---

## 5 Limitations & gotchas ⚠️

* **O(N²)** attention if you leave `_optimize_active` off.
* Back-prop **spans all ticks** by default; use TBPTT or `reset_state()` if memory explodes.
* No ACT/halting, no quantisation path, no learned router yet implemented.
* Sensorimotor objectives are only sketched in prose; you’ll have to implement them.

---

## 6 Repo Structure 📁

```bash
columnar_transformer/
├── tct.py                     # Core implementation: ThousandColumns model
├── tct_test.py                # Full test suite (shapes, gradients, stability, etc.)
├── README.md                  # This file
│
├── docs/                      # Notes, design rationale, and references
│   ├── idea.md                         # Core concept & architectural motivation
│   ├── design_space.md                 # Design-space catalogue with some options
│   ├── chosen_implementation.md        # Why each component was picked
│   ├── sources.md                      # Related work & citations
│   ├── metadata.md                     # Version notes, creation method, etc.
│   ├── authors_extra_notes.md          # Unfiltered thoughts, caveats, criticisms
│   ├── tct_conceptual_aigenerated.html # Conceptual explanation (ai generated)
│   └── tct_visualization.html          # Architecture Visualization
```

---

## 7 Background reading 📚

* Hawkins, **“A Thousand Brains”** (2021) – biological inspiration.
* Goyal et al., **RIMs** (2019) – sparse stateful modules.
* Liu & Papamakarios, **TIM** (2021) – competitive attention.
* Fedus et al., **Switch Transformer** (2021) – MoE engineering tricks.

See *sources.md* for a longer bibliography.

---
