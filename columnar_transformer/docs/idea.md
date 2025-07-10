# “Thousand-Columns” Concept Document  
*(alias: TCT – Thousand-Columns Transformer)*  

---

## 1 Goal in One Sentence  
Build an **active, modular neural system** where **thousands of small stateful units** individually model their slice of the world, **exchange messages**, and converge on a **consensus belief** that drives perception, cognition and action.

---

## 2 Core Biological Inspiration  

| Cortex element | Computational analogue | Purpose |
|---------------|------------------------|---------|
| **Cortical column** (6-layer micro-circuit) | A *micro-module* with persistent latent vector **hᵢ** | Learns a complete sensorimotor model of some patch of the input space. |
| **Grid cells / path-integration (Layer-6)** | Internal location code updated by efference copy | Binds sensations to *locations* in an object-centred reference frame. |
| **Layer-5 motor output** | Action token / command vector | Generates movement; copy goes back to grid layer for state update. |
| **Layer-2/3 lateral fibres** | Message-passing / “consensus bus” | Shares object hypotheses; implements fast voting. |

---

## 3 Engineering Principles  

1. **Modularity** – 10³–10⁵ (rather than specific numbers, the principle of separated, modeling a part of the input by itself, communicating micro modules) identical micro-modules; parameters shared, *state* per module.  
2. **Persistent State** – Each module keeps **hᵢ** across time → enables path-integration & memory.  
3. **Sparse Activation** – Only a top-k subset updates each tick → O(k) compute, O(N) memory. (Since different modules are responsible for different inputs and to make it computationally efficient) 
4. **Two-Phase Cycle (per tick)**  
   1. *Local update*: module integrates current sensory slice + last consensus context.  
   2. *Communication*: modules emit low-dim votes; bus returns messages + global context.  
5. **Consensus over Competition** – Outcome is not chosen by a router but by *mutual reinforcement*; hypotheses consistent across modules amplify, others decay.  
6. **Active Learning Loop** – Model trained on sensorimotor prediction, not static corpora; action tokens change future inputs. (Writing this in more of a metaphorical sense since our main tests are going to be in NLP. Still, some stuff we write will be applicable to other possible modifications or use cases.) 
7. **Factorised Representations** – Regularisers (e.g. VICReg, FER-style evolutionary culling) keep columns disentangled and specialised.  

---

## 4 High-Level Objectives the Architecture Should Achieve  

| Objective | Why it matters |
|-----------|----------------|
| **Robust generalisation** | Distributed voting dampens single-unit failures and adversarial noise. |
| **Continual learning** (Not the "human kind", just preventing the classic catastrophic forgetting in the training process) | Localised updates let new columns specialise without catastrophic forgetting globally. |
| **Interpretability** (AI generated idea, I don't think it would help)| Column-level latents can be probed; each should map to a coherent concept or “object” slice. |
| **Scalability** | Adding columns should raise capacity linearly; params stay sub-linear via weight sharing. |
| **Sensorimotor grounding** | Predictions improve by actively sampling the environment; crucial for robotics, embodied AI. |

---

## 5 Conceptual Data-Flow Diagram  

```text
  sensory xₜ ─┐
              ▼
        ┌──────────────┐
        │ Columnᵢ      │  (∀ i ∈ 1…N)
hᵢₜ ───►│ 1. Local fwd │───► vote vᵢₜ
        │ 2. Gating    │
        └──────────────┘
              ▲                 consensus bus (attention / tree / LSH)
              │ mᵢₜ              ▼
              └───────────┬─────────┐
                          │ pooled gₜ
````

---

## 6 Final Takeaway

The project aims to **operationalise Jeff Hawkins’ Thousand-Brains theory** in a modern transformer framework: *thousands of stateful, talking micro-experts* that learn by acting, maintain their own reference-framed models, and resolve ambiguity through rapid lateral voting. The anticipated payoff is a system that **generalises like an ensemble, learns like an agent, and scales like a transformer.**
