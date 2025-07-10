# Design-Space Catalogue for the Thousand-Columns System

*(Some possible components & wiring options)*

---

## 0 Notation

* **N** – number of columns (10³ – 10⁵)
* **hᵢ** – persistent latent state of column *i*
* **xₜ** – external sensory input at time *t*
* **vᵢₜ** – vote / hypothesis vector emitted by column *i*
* **mᵢₜ** – message returned to column *i* from bus
* **gₜ** – global pooled context after consensus

---

## 1 Input Distribution

| Choice                              | Description                                                                         | Pros                                             | Cons                                                      |
| ----------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------- |
| **1A Broadcast-all**          | Every column receives the entire**xₜ** (fully shared view).                  | Simplest; no routing.                            | Redundant compute; may discourage specialisation.         |
| **1B Fixed receptive fields** | Hard-coded slice per column (e.g. token index, image patch).                        | Exploits locality; columns specialise naturally. | Rigid; poor with variable-length inputs.                  |
| **1C Learned router**         | Trainable mapping**xₜ → columns** (k-nearest prototypes).                   | Adaptive load-balancing.                         | Extra parameters; router becomes single point of failure. |
| **1D Hierarchical broadcast** | Low-level columns get raw data; higher-level columns get pooled outputs from lower. | Mirrors cortical hierarchy; natural abstraction. | Requires multiple column tiers & latency.                 |

---

## 2 Column Internal Architecture

1. **Tiny FFN cell** — 2-layer MLP + residual (cheap, stateless within tick).
2. **GRU/LSTM cell** — adds gate recurrence; better for sequence memory.
3. **Micro-transformer (SA+FFN)** — full attention *inside* column, dimensions 32-64.
4. **Hypernetwork-generated cell** — per-column weights produced by small hyper-net; param-efficient diversity.
5. **Sparse dendritic tree** — mimic cortical segregation: proximal (feed-forward) vs distal (context) branches.

---

## 3 State Storage & Precision

| Option                 | Memory / column    | Note                                   |
| ---------------------- | ------------------ | -------------------------------------- |
| **FP32**         | 4 × d_state B     | Max precision, training only.          |
| **FP16/BF16**    | 2 × d_state B     | Standard for mixed-precision training. |
| **INT8 quant.**  | 1 × d_state B     | Viable at inference with QAT.          |
| **Binary (+BN)** | 0.125 × d_state B | Extreme compression; research only.    |

---

## 4 Gating / Activation Policy

* **Top-k highest gate score** (deterministic)
* **Stochastic Bernoulli with learnable *pᵢ*** (as in RIMs)
* **Prediction-error spike** — update only if ‖hᵢ-predicted‖ > τ
* **Fixed active sets per timestep** (round-robin) for hard latency guarantees
* **Energy budget RL** — column pays “metabolic cost” in loss for being active

---

## 5 Vote Vector Generation

1. **Linear projection**: vᵢ = W·hᵢ
2. **Non-linear head**: MLP or cosine similarity to prototype bank
3. **Discrete codebook index** (vector quantisation)
4. **Hash of latent** (LSH) for cheap bucketed communication
5. **Graph message** (key, value) pair enabling key-value attention on bus

---

## 6 Consensus Bus Topologies

| ID                             | Mechanism                                     | Complexity   | Suitable*N*               |
| ------------------------------ | --------------------------------------------- | ------------ | --------------------------- |
| **6A Full MH-Attention** | v⟼softmax(QKᵀ)V                             | O(N²d)      | ≤8 k                       |
| **6B Ring All-Reduce**   | token passing, cumulative sum                 | O(Nd)        | 8 k – 64 k                 |
| **6C Radix-k Tree**      | hierarchical reduction                        | O(N logₖ N) | 64 k +                      |
| **6D LSH Attention**     | hash buckets then local attention             | O(N d log N) | any                         |
| **6E Router-Experts**    | Mixture-of-Experts bus; tokens sent to K hubs | O(N d)       | very large; trades fidelity |

Optional **multiple bus passes** (vote→message→re-vote) to sharpen consensus.

---

## 7 Message Integration (Broadcast Phase)

* **Additive residual**: hᵢ = hᵢ + W_m · mᵢ
* **Concatenation + MLP**
* **FiLM conditioning**: scale-and-shift on hᵢ’s channels
* **Gated attention**: column queries message pool with its own key
* **Hebbian update**: hᵢ ← (1-α)hᵢ + α mᵢ (biologically plausible)

---

## 8 Motor / Action Pathway

1. **Explicit action token** emitted by a subset of columns; fed to environment (SMT style).
2. **Latent Δlocation vector** decoded into actuator commands; copy returned as efference.
3. **Discrete action head** trained with RL (policy gradient / Q-learning).
4. **Hierarchical motor plan** — high-level columns output goals; low-level columns refine.
5. **No motor (passive mode)** for text-only tasks; efference copy simulated by random walk.

---

## 9 Read-Out Strategies

* **Mean / Max pool of hᵢ**
* **Attention over columns using task query**
* **Hierarchical aggregator** (cluster → super-cluster → head)
* **Voting ensemble** — each column predicts logits; majority vote or soft average
* **Mixture-of-Experts output layer** routed by global context gₜ

---

## 10 Training Objectives (Multi-task Mix)

| Category                          | Candidate losses                                        |
| --------------------------------- | ------------------------------------------------------- |
| **Sensorimotor prediction** | next-x, next-state, contrastive predictive coding (CPC) |
| **Task-specific**           | CE, RL reward, regression L2                            |
| **Consensus alignment**     | KL(vᵢ                                                  |
| **Disentanglement**         | VICReg, Barlow Twins on {hᵢ}                           |
| **Sparsity / energy**       | L₀, L₁ on gate, FLOPs regulariser                     |
| **Stability**               | EMA target networks, auxiliary distillation             |

---

## 11 Hierarchy & Grouping Variants

* **Flat**: single bus for all columns.
* **Two-stage**: local groups (cortical area) do micro-consensus → super-bus.
* **Dynamic clustering**: groups formed on-the-fly via k-means on hᵢ.
* **Tree of experts**: each level doubles receptive field, halves column count.

---

## 12 Parameter-Sharing Modes

| Mode                             | Params growth                           | Notes                                           |
| -------------------------------- | --------------------------------------- | ----------------------------------------------- |
| **Global shared**          | O(1)                                    | fastest compile, less expressivity              |
| **LoRA / IA³ per column** | O(N·rank)                              | tunable trade-off                               |
| **Hyper-Net generated**    | O(N) for embeddings; weights via f(emb) | amortises large weights                         |
| **Unique weights**         | O(N·d²)                               | only viable for ≤1 k columns or strong pruning |

---

## 13 Lifecycle Management

* **Static N** — fixed at init.
* **Prune-and-grow** — periodically delete low-utility columns, spawn new (NeuroEvolution).
* **Adaptive capacity** — grow when loss plateau detected; shrink when utilisation < θ.
* **Column dropout** — randomly deactivate some columns at train time for robustness.

---

## 14 Scheduling / Time Granularity

| Scheme                              | Micro-cycles per macro-timestep | Latency                     | Biological analogue        |
| ----------------------------------- | ------------------------------- | --------------------------- | -------------------------- |
| **Single-tick**               | 1                               | minimal                     | cortical gamma burst       |
| **Multi-tick recurrent**      | 2-4                             | higher                      | gamma-within-alpha nesting |
| **Asynchronous event-driven** | variable                        | unpredictable but efficient | spike-timing networks      |

---

## 15 Hardware & Parallelism

* **GPU / TPU** — batched kernels; bus via NCCL all-reduce.
* **FSDP sharding** — shard states across devices; shared weights replicated.
* **Neuromorphic** (Loihi 2) — map columns to cores; bus via multicast.
* **FPGA pipeline** — streaming implementation with ring bus.

---

## 16 Combinatorial Examples

1. **Minimal text TCT**

   * 1A Broadcast, tiny FFN (32d), INT8 state, Top-k=5 %, MH-Attention bus, additive message, mean-pooled readout.
2. **Sensorimotor robot agent**

   * 1B Receptive fields (vision patch), micro-transformer columns, FP16, prediction-error gating, LSH bus, FiLM integration, latent Δlocation motor head, RL + CPC losses.
3. **Large-scale hierarchical**

   * 1C Learned router, GRU columns with LoRA adapters, INT8 state, energy-budget RL gating, two-level tree bus, Hebbian integration, hierarchical aggregator readout, prune-and-grow lifecycle.

---

## 17 Open Combinations

* **Hyper-net + dynamic clustering + event-driven bus**
* **Ring all-reduce bus + spike gating on FER corruption metric**
* **Motor copy as direct phase shift in grid-PE**

---
