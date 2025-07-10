## Starting source: [Artem Kirsanov - A Fundamental Unit Of Intelligence](https://youtu.be/Dykkubb-Qus?si=d_N5EJQxTGc9U9Er)

## Later sources to scan (from the description of the above video):

### 📚 FURTHER READING & REFERENCES:

For those who want to dive deeper into the science:

1. Hawkins, J. (2021). A Thousand Brains: A New Theory of Intelligence. (Book co-authored with Richard Dawkins).
2. Clay, V. et al. (2024). The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence.
3. Hawkins, J. et al. (2019). A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex. Frontiers in Neural Circuits.
4. Hawkins, J. & Ahmad, S. (2016). Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex. Frontiers in Neural Circuits.
5. Harris, K.D. & Shepherd, G.M.G. (2015). The neocortical circuit: themes and variations. Nature Neuroscience.
6. Haueis, P. (2016). The life of the cortical column: opening the domain of functional architecture of the cortex (1955–1981). History and Philosophy of the Life Sciences.
7. Hawkins, J. (N.d.). A Theory of How Columns in the Neocortex Enable Learning the Structure of the World. Numenta Whitepaper.
8. Hawkins, J., Leadholm, N., Clay, V., 2025. Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain.
9. Leadholm, N., Clay, V., Knudstrup, S., Lee, H., Hawkins, J., 2025. Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference.

## Added representational principles (tried to) from the following source:

- [Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis](https://arxiv.org/pdf/2505.11581)

## Existing/Related/Prior Work:

| Year              | Title & link                                                                                                                                | Why it matters                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2019**    | **“Recurrent Independent Mechanisms (RIMs)”** – Goyal et al.                                                          | First neural architecture that splits a recurrent net into sparsely-communicating, state-holding mini-modules – the conceptual ancestor of per-column latents. |
| **2021**    | **“Transformers with Independent Mechanisms (TIM)”** – Liu & Papamakarios                                           | Ports the RIM idea into a transformer layer; mechanisms only interact through attention, giving a template for column-to-column messaging.                      |
| **2021**    | **“Relating Transformers to Models and Neural Representations of the Hippocampal Formation”** – Whittington et al.     | Shows that recurrent-PE Transformers spontaneously form grid- and place-cell codes, justifying a grid-cell-style location signal inside each column.            |
| **2021**    | **“Switch Transformer: Scaling to Trillion-Parameter Models with Sparse MoE”** – Fedus et al.                        | Demonstrates the engineering mechanics (routing, sharding) for thousands of experts; useful even though experts are stateless.                                  |
| **2023**    | **“SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention”** – Csordás et al.                        | Extends MoE sparsity to*attention heads*; inspiration for fine-grained, head-level column communication.                                                      |
| **2024**    | **“Mixture of a Million Experts (PEER)”** – He et al.                                                                 | Proves that ultra-tiny experts (10⁶) are trainable; key evidence that scaling to >10³ columns is feasible.                                                    |
| **2024**    | **“GridPE: Unifying Positional Encoding in Transformers with a Grid Code”** – Zhao et al.                             | Introduces continuous grid-cell embeddings as a positional code, matching the layer-6 “location” requirement of Thousand Brains.                              |
| **2024-11** | **Numenta’s *Thousand Brains Project* – open-source sensorimotor framework** (press release + GitHub)  | First end-to-end implementation of Hawkins’ theory (not transformer-based); invaluable reference for sensorimotor objectives and column APIs.                  |
| **2025**    | **“Cortico-Column Self-Attention”** – Ghosh et al. (pre-print)                                              | Maps transformer sublayers onto the six cortical layers; proposes a column-wise attention stack—direct architectural inspiration.                              |
| **2025**    | **“A Sensorimotor Vision Transformer”** – Gadzicki et al.                                                          | Adds an explicit saccade-like action token so vision transformers learn by*moving*—evidence that sensorimotor loops boost sample-efficiency.                 |
