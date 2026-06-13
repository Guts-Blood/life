# Recommendation Pool

Created: 2026-06-13

These are candidates, not commitments. Promote only a small number into `current-pool.md` during weekly updates.

## Priority Map

| Priority | Track | Why |
| --- | --- | --- |
| P0 | Training | You already selected Scaling Book; it compounds directly into model/data flywheel work. |
| P1 | Agent harness theory | Closest to your current differentiated edge and personal theory. |
| P2 | World models | Important, but should be entered through agent reliability and planning questions, not as an isolated rabbit hole. |

## Auto-Research / Harness Agent Theory

### A. Directly Useful Anchors

1. A control-theoretic framing for agentic systems
   - Link: https://arxiv.org/html/2603.10779v1
   - Why: Directly overlaps with your plant / controller / observer / feedback framing. Use it as a foil: what does it capture, and where is your theory sharper?

2. Stop Comparing LLM Agents Without Disclosing the Harness
   - Link: https://arxiv.org/abs/2605.23950
   - Why: Very close to your core point that agent performance is not separable from harness, tools, observation, memory, and evaluator design.

3. Survey on evaluation of LLM-based agents
   - Link: https://arxiv.org/abs/2503.16416
   - Why: Good map of evaluation dimensions. Read selectively for taxonomy, benchmark design, and failure attribution gaps.

4. BrowserGym / browser-agent evaluation ecosystem
   - Link: https://arxiv.org/abs/2412.05467
   - OpenReview: https://openreview.net/forum?id=5298fKGmv3
   - Why: Close to browser-use work. Use it to compare your benchmark design against public environment abstractions.

5. OSWorld
   - Link: https://arxiv.org/abs/2404.07972
   - Why: Desktop/computer-use benchmark with broad task environment. Useful for thinking about observation, action space, and evaluator robustness.

6. WebArena
   - Link: https://arxiv.org/abs/2307.13854
   - Why: Still a central web-agent benchmark reference. Useful as a baseline for what public browser benchmarks miss.

### B. Self-Improvement And Skill Distillation

1. Reflexion
   - Link: https://arxiv.org/abs/2303.11366
   - Why: Classic verbal feedback / self-reflection loop. Read critically: where does language feedback help, and where does it fail without structured observer signals?

2. Voyager
   - Link: https://arxiv.org/abs/2305.16291
   - Why: Skill library and lifelong learning loop. Useful for comparing with your auto-skill generation / upload / convergence pipeline.

3. Agent skill / playbook distillation papers and implementations
   - Use as a rolling slot rather than a fixed title.
   - Selection criterion: must expose how skills are represented, retrieved, evaluated, and pruned.

## Training / Scaling / Systems

### A. Current Backbone

1. How To Scale Your Model
   - Link: https://jax-ml.github.io/scaling-book/
   - Why: Your chosen daily backbone. Best read with notes that connect scaling mechanics to agent training and trajectory-data flywheels.

### B. Strong Supplements

1. Ultra-Scale Playbook
   - Link: https://huggingface.co/spaces/nanotron/ultrascale-playbook
   - Why: Practical distributed training systems view. Good companion when Scaling Book becomes abstract.

2. Stanford CS336: Language Modeling from Scratch
   - Link: https://stanford-cs336.github.io/spring2025/
   - Why: Good bridge between model internals, data, optimization, and training practice.

3. FlashAttention
   - Link: https://arxiv.org/abs/2205.14135
   - Why: Useful for understanding the hardware-aware side of training/inference efficiency.

4. ZeRO
   - Link: https://arxiv.org/abs/1910.02054
   - Why: Foundational for distributed optimizer/memory partitioning.

5. Megatron-LM
   - Link: https://arxiv.org/abs/1909.08053
   - Why: Foundational tensor/model parallelism reference.

Reading rule:

Do not read all systems papers linearly. Read them when Scaling Book raises a concrete question: memory, compute, communication, optimizer state, parallelism, data scaling, or inference economics.

## World Models

### A. Foundations

1. World Models
   - Link: https://arxiv.org/abs/1803.10122
   - Why: Classic compact formulation: learn compressed latent dynamics, then plan/control inside it.

2. DreamerV3
   - Link: https://arxiv.org/abs/2301.04104
   - Why: Strong modern example of learning behaviors from world models across domains.

3. A Path Towards Autonomous Machine Intelligence
   - Link: https://openreview.net/forum?id=BZ5a1r-kVsf
   - Why: JEPA-style framing; useful for thinking about predictive representations without reducing everything to next-token prediction.

### B. Generative / Interactive Environments

1. Genie
   - Link: https://arxiv.org/abs/2402.15391
   - Why: Generative interactive environment learning. Relevant if you think browser / GUI agents need better simulated interaction spaces.

2. Genie 2
   - Link: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
   - Why: High-level direction of large foundation world models. Treat as directional research signal, not a textbook.

### C. Entry Question

When you start this track, do not begin with "what are world models?" Begin with:

What kind of internal state would make a browser agent less myopic, less oscillatory, and better at recovering from partial failure?

That makes world-model reading serve your agent harness theory instead of becoming a separate academic island.

## Parking Lot

- General "AI agent framework" tutorials.
- Prompt collections.
- Papers whose main claim is benchmark improvement without revealing a mechanism.
- World-model hype pieces without environment, latent-state, rollout, planning, or evaluation detail.
