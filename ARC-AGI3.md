# ARC-AGI-3 — Research Dossier and Strategy

**Goal:** win (or place in) ARC Prize 2026, ARC-AGI-3 track. **Submission deadline: November 2, 2026.** $2M prize pool across tracks. Open-source mandate (MIT/Apache/CC0/GPLv3).

This document is the planning artifact for that effort. It pulls together what's known about ARC-AGI-3 as of April 2026, why TRM alone won't win, what a competitive architecture looks like, and the order of operations to get there. The bibliography at the end lists 20 papers whose ideas the proposed architecture borrows from; a download script is provided in `papers/download.sh`.

---

## 1. The Competition

### Timeline
| Date | Event |
|---|---|
| 2026-03-25 | Competition opens |
| 2026-06-30 | Milestone 1 (interim leaderboard cut) |
| 2026-09-30 | Milestone 2 |
| **2026-11-02** | **Submission deadline** |
| 2026-11-08 | Paper submission deadline |
| 2026-12-04 | Results announced |

### Submission constraints (the binding ones)
- **No internet at evaluation.** No calls to OpenAI/Anthropic/HF inference APIs from the submitted code. The agent must be fully self-contained.
- **Kaggle notebook submission.** All weights bundled. Runtime/memory caps apply (check competition page closer to submission).
- **Open-source mandate.** Winning entries must publish their code under a permissive license before being awarded the prize.
- **Eligible licenses:** CC0, MIT-0, Apache-2.0, GPLv3.

### What's at stake
- **$2M total** across ARC-AGI-3, ARC-AGI-2, and Paper Prize tracks.
- ARC-AGI-3 milestone bonuses on Jun 30 and Sep 30 — interim leaders get paid even before the final.

### Current state of the art
**Frontier AI agents score below 1% on ARC-AGI-3 (Mar 2026).** Humans solve 100% of the games by design. The 2025 ARC-AGI-3 *preview* (a smaller, simpler set of games) had:

| Rank | Agent | Score | Approach |
|---|---|---|---|
| 1 | StochasticGoose | **12.58%** (18 levels) | CNN + RL action-learning; predicts which actions cause frame changes |
| 2 | Blind Squirrel | 6.71% (13 levels) | ResNet18 + state-graph + value model over action pairs |
| 3–8 | various | 2–8% | Frame-graph analysis, rule-based, DSL+LLM, LLM video |

**Lesson from the preview:** efficient *exploration* outperforms reasoning quality. The 1st place agent's edge was a learned "this action will probably do something" predictor, not a smarter reasoning core.

---

## 2. The Game (technical specs)

### Observation
- Grid up to **64×64**.
- Cell values: **integers 0–15** (16 colors/states).
- Coordinate system: `(0,0)` top-left, `(x,y)` format.

### Action space (8 total)
| Action | Type | Notes |
|---|---|---|
| `RESET` | fixed | restart current level |
| `ACTION1..ACTION4` | fixed (discrete) | up / down / left / right (semantic, varies by game) |
| `ACTION5` | fixed | interact / select / rotate (game-specific) |
| `ACTION6` | **(x,y) coordinates 0–63** | the only continuous-ish action |
| `ACTION7` | fixed | undo |

`available_actions` in `FrameData` masks per-game which of the 7 are exposed.

### Episode budget
- **MAX_ACTIONS = 80** per game. Strict step limit. The agent that wastes steps loses, even with perfect per-step reasoning.

### Per-step data the agent receives
`FrameData`: `{ levels_completed, win_levels, state, frame, available_actions, game_id, guid, full_reset }`.

### Game persistence
Stateful — server tracks per-session via `AWSALB*` cookies. `arc-agi-3-agents` SDK handles this.

---

## 3. Why TRM alone loses

TRM is a static input→output model. It has:
- **No memory between calls.** Each forward pass is independent. ARC-AGI-3 needs "I've been in this state before, action X failed."
- **No exploration mechanism.** Pure supervised loss. No curiosity, no UCB, no visitation counts.
- **No planning / lookahead.** Recursion deepens *current* reasoning; it doesn't simulate `state + action → next_state`.
- **Fixed compute envelope.** Even with ACT halting, no way to "spend more steps thinking about a critical decision."

The 80-action budget rewards **efficient state-space search**, not single-step decision quality. TRM optimizes the wrong axis.

**Realistic ceiling for a behavior-cloned TRM:** 5–15% on training-distribution games, near-0% on hidden test. Mid-pack at best, far below SOTA needed for prize money.

---

## 4. Proposed architecture (full stack)

### Design philosophy
The winning ARC-AGI-3 agent will be a **loop**, not a single model. Five components, each doing one thing:

```
                     ┌────────────────────┐
   raw 64×64 grid → │  perception (CNN)  │ → state encoding
                     └────────────────────┘
                              │
                              ▼
                     ┌────────────────────┐
                     │  episodic memory   │ ← read: "seen this state? what happened?"
                     │ (state→outcome)   │ → write: log (s,a,r,s') after each step
                     └────────────────────┘
                              │
                              ▼
                     ┌────────────────────┐
                     │  reasoning core    │ ← TRM (or BDH) — "given this state +
                     │   (TRM / BDH)     │   what I've tried, score each action"
                     └────────────────────┘
                              │
                              ▼
                     ┌────────────────────┐
                     │  exploration head  │ ← RND / frame-change predictor —
                     │  (intrinsic reward)│   bias toward novel / effectful states
                     └────────────────────┘
                              │
                              ▼
                     ┌────────────────────┐
                     │   action selector  │ → emit GameAction; if ACTION6,
                     │  (type + coord)   │   also emit (x,y)
                     └────────────────────┘
                              │
                              ▼
                            ENV
```

### Component-by-component

#### (a) Perception
- **Choice:** small CNN, 3 conv layers, ~100K params. Outputs a fixed-length encoding regardless of grid size (pad/crop to 64×64).
- **Why not skip and feed grid straight to TRM:** TRM expects flat token sequences (seq_len=4096 for 64×64). Spatial inductive bias from a CNN front-end is essentially free and saves the reasoning core from re-learning convolution.
- **Alternative:** learnable patch embedding (ViT-style, 4×4 patches → 256 tokens of d=512). Roughly equivalent.

#### (b) Episodic memory
- **Choice (v1):** plain Python dict, key = `hash(state_grid_bytes)`, value = `[(action, outcome_delta, success_so_far)]`.
- **Why not Engram (DeepSeek):** Engram is a *static knowledge* lookup primitive (hashed n-grams → embeddings), designed for "Princess Diana is …" style retrieval. Episodic memory is conceptually different — it's a per-episode write-then-read log. Use Engram only later if the dict becomes a bottleneck (it won't, in 80-step episodes).
- **Critical insight:** the 2nd-place ARC-AGI-3 preview agent ("Blind Squirrel") explicitly built a "state graph from frames." This component alone is the single highest-ROI feature in the whole stack.

#### (c) Reasoning core
- **Default: TRM** (already in this repo, training pipeline exists, HF init weights for ARC-AGI-2 give a partial warm start).
- **Stretch: BDH** (Baby Dragon Hatchling). Hits 97.4% on Sudoku Extreme vs TRM's 84.8%. Brain-inspired, sparse + monosemantic. But brand new (Sept 2025), no ARC checkpoint, would require rebuilding the trainer.
- **Stretch ensemble: HRM + TRM + BDH**, picked per-game by validation accuracy. NVARC's 2025 win on ARC-AGI-2 (24%) used a TRM ensemble component — there's precedent.

#### (d) Exploration / intrinsic motivation
- **Frame-change predictor (StochasticGoose-style):** small head trained to predict P(state changes | action). At inference, multiply this into the action-selection logits → bias toward effectful actions. **Cheap, easy, and was the 1st-place 2025 preview entry.**
- **RND (Random Network Distillation):** secondary novelty bonus. A randomly-initialised target network and a learned predictor; intrinsic reward = predictor error on current state. High error = state never seen = explore here.
- **Together:** `score(action) = TRM_logit + α · frame_change_prob + β · RND_bonus(predicted_next_state)`.

#### (e) Action selector
- **Type head:** softmax over 8 actions.
- **Coord head (only used when type==ACTION6):** flat softmax over 64×64=4096 cells, OR factored as (row_softmax × col_softmax) for 64+64=128 logits.
- **Mask:** zero out `~available_actions` from the type head before sampling.

### Training data
ARC-AGI-3 has **no labelled training set**. You generate it.

1. **LLM bootstrap (legal — only Kaggle eval forbids internet).** Use Claude-Opus or GPT-4 offline, give it the game state, ask "what's the next action?", record the trajectory. ~50 successful trajectories per game seeds the supervised dataset.
2. **Random + heuristic for games the LLM can't crack.**
3. **DAgger loop.** Once the agent works, replay it against the env, append its trajectories (with corrections) to the dataset, retrain. Repeat.
4. **Self-play after convergence.** Use the trained agent to generate more data on its weakest games.

### Why the stack is more competitive than just TRM

| Component | Lifts the score by addressing |
|---|---|
| Episodic memory | "Don't repeat failed actions in the same state" — fixes wasted steps |
| Frame-change predictor | "Don't take no-op actions" — biggest score driver in 2025 preview |
| RND novelty | "Find unseen states first" — handles hidden-test generalisation |
| TRM/BDH reasoning | "Pick the right action when memory + heuristics are ambiguous" — what TRM is actually good at |

---

## 5. Implementation plan (phased, against the Nov 2 deadline)

| Phase | Weeks | Deliverable |
|---|---|---|
| **0 — Verify access** | wk 0 (3 days) | API key, `arc-agi-3-agents` SDK runs locally, random agent vs `ls20` smoke test. Game registry JSON. |
| **1 — Env + data plumbing** | wk 1–2 | Trajectory collector, dataset class, dual action head on TRM, ARC-3 config YAML, early-stopping callback in trainer. |
| **2 — LLM-bootstrapped BC baseline** | wk 3–4 | LLM (Claude offline) generates ~50 trajectories per game; train TRM behaviour-cloning baseline. Submit to leaderboard for Jun 30 milestone. **Target: 5–10%.** |
| **3 — Add memory + frame-change predictor** | wk 5–7 | State-graph episodic memory; train frame-change predictor head. Re-eval. **Target: 12–18% (matches StochasticGoose).** |
| **4 — RND + DAgger loop** | wk 8–10 | Add RND bonus. DAgger: replay agent → augment dataset → retrain. Re-eval after each round. **Target: 18–25%.** |
| **5 — Stretch: BDH / world model** | wk 11–14 | Swap reasoning core to BDH OR add DreamerV3-style learned world model for 1-step lookahead. **Target: 25–30%.** |
| **6 — Sep 30 milestone push** | wk 15–17 | Hyperparam sweep, ensemble (TRM + BDH + HRM), final dataset augmentation. |
| **7 — Submission packaging + buffer** | wk 18–20 | Strip wandb/HF/codecarbon imports, bake weights into Kaggle notebook, verify offline run. Submit to Nov 2. |

**Honest read:** mid-tier finish (top-10) is achievable. Winning outright is a stretch — the field will be working hard. But the prize structure pays milestones, so Phase 4 results would already be earning.

---

## 6. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| LLM-bootstrapped trajectories cost > $200 | Budget overrun | Cap at 30 trajectories/game; mix in random + heuristic |
| Hidden-test games structurally different from public | Scores on private set << public | RND + curiosity (good zero-shot exploration); never overfit per-game |
| Compute: 64×64×8 forward passes saturate VRAM on RTX 3070/5070 | OOM mid-training | seq_len=1024 with patch embedding instead of seq_len=4096; gradient checkpointing |
| Kaggle notebook size limits | Submission rejected | Quantize weights to int8 or fp16; prune ensemble to 1–2 models |
| ARChitects/NVARC enter ARC-AGI-3 with their TTT machinery | Outclassed | Same battlefield; their ARC-AGI-1/2 stack assumes static puzzles, doesn't trivially port to interactive games — their advantage is smaller here |
| "Frame-change predictor" trick gets known and copied | Lose differentiation | Treat it as table stakes; differentiation comes from BDH+memory combo |
| Internet-needed dependency leaks into submission | Auto-fail at scoring | CI step: run submission with `--network none` Docker before submit |

---

## 7. Component swap matrix (what to try if Phase N stalls)

| If stuck on… | Try swapping |
|---|---|
| Reasoning core (TRM) plateaus | → HRM (slightly bigger, brain-inspired) → BDH (newest, best reasoning numbers) |
| Memory (dict) doesn't generalise across episodes | → Differentiable Neural Computer / Memformer / Engram-style hash table |
| Exploration insufficient | → Replace RND with NGU (Never Give Up) or Go-Explore |
| Per-step ambiguity | → Add MuZero-style 1-step MCTS (3–5 rollouts via learned dynamics model) |
| Long-horizon planning | → DreamerV3 world model + actor-critic on imagined trajectories |
| Need offline RL on collected trajectories | → Decision Transformer (return-conditioned sequence modelling) |

---

## 8. Bibliography (20 papers — see `papers/download.sh` to grab PDFs)

### A. Reasoning cores (recursive, brain-inspired, attention)

1. **Jolicoeur-Martineau (2025) — *Less is More: Recursive Reasoning with Tiny Networks (TRM).*** arXiv:2510.04871. The architecture currently in this repo. ARC-AGI-1 ~45%, ARC-AGI-2 ~8% standalone; was a component of NVARC's 2025 1st-place ARC-AGI-2 ensemble. ARC Prize 2025 Paper Award winner.
2. **Wang et al. (2025) — *Hierarchical Reasoning Model (HRM).*** arXiv:2506.21734. TRM's predecessor. 27M params, 41% on ARC-AGI-1 with only 1000 training tasks. Two coupled recurrent modules: H (slow planner), L (fast worker). Worth using as an ensemble companion to TRM.
3. **Kosowski et al. / Pathway (2025) — *The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain (BDH).*** arXiv:2509.26507. 97.4% on Sudoku Extreme without CoT. Sparse, monosemantic, scale-free Hebbian neurons. Best per-step reasoning candidate to swap into the stack.
4. **Dehghani et al. (2018) — *Universal Transformer.*** arXiv:1807.03819. Recursive depth via shared weights with adaptive halting. Foundational ancestor of TRM/HRM.
5. **Graves (2016) — *Adaptive Computation Time (ACT) for Recurrent Neural Networks.*** arXiv:1603.08983. The halting head used by TRM. Read this to understand why ACT collapses (and how to detect it via `frac_at_max_steps`).

### B. Memory architectures

6. **DeepSeek-AI (2026) — *Engram: Conditional Memory via Scalable Lookup.*** arXiv:2601.07372. O(1) hashed n-gram lookup. Static knowledge, not episodic — but the lookup primitive is reusable for an Engram-style episodic table if the dict approach fails.
7. **Graves et al. (2016) — *Hybrid Computing Using a Neural Network with Dynamic External Memory (Differentiable Neural Computer).*** Nature 538, 471–476. Read/write external memory. Theoretically what you want for an 80-step episode; practically heavy.
8. **Dai et al. (2019) — *Transformer-XL.*** arXiv:1901.02860. Recurrence + relative positional encoding for long-range memory. Useful if you decide to feed the agent the whole episode trace as a sequence.

### C. Model-based RL & planning

9. **Schrittwieser et al. (2020) — *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).*** arXiv:1911.08265. Learned dynamics model + MCTS. The reference for how to do planning when you don't know the rules.
10. **Hafner et al. (2023) — *Mastering Diverse Domains through World Models (DreamerV3).*** arXiv:2301.04104. World model → imagined rollouts → actor-critic. First to collect Minecraft diamond from scratch. Strong candidate for Phase 5.
11. **Ha & Schmidhuber (2018) — *World Models.*** arXiv:1803.10122. The original world-model paper. Compact and worth reading even if you go DreamerV3 in implementation.
12. **Chen et al. (2021) — *Decision Transformer: Reinforcement Learning via Sequence Modeling.*** arXiv:2106.01345. Cast offline RL as conditional sequence modelling. Useful if you accumulate a large trajectory dataset and want a transformer-native fit.

### D. Exploration / sparse-reward RL

13. **Burda et al. (2018) — *Exploration by Random Network Distillation (RND).*** arXiv:1810.12894. The intrinsic-reward bonus proposed for Phase 4. SOTA on Montezuma's Revenge in 2018; trivially simple to add.
14. **Pathak et al. (2017) — *Curiosity-Driven Exploration by Self-Supervised Prediction (ICM).*** arXiv:1705.05363. Predicts inverse and forward dynamics; intrinsic reward = forward-model error. Classical curiosity baseline.
15. **Ecoffet et al. (2019) — *Go-Explore: A New Approach for Hard-Exploration Problems.*** arXiv:1901.10995. Explicitly remembers + revisits promising states. Conceptually closest to the episodic-memory approach proposed in §4(b).

### E. ARC-AGI canon (read these first if you've never touched ARC)

16. **Chollet (2019) — *On the Measure of Intelligence.*** arXiv:1911.01547. The paper that introduced ARC and defined the benchmark's design philosophy. Required reading.
17. **Akyürek et al. (2024) — *The Surprising Effectiveness of Test-Time Training for Abstract Reasoning.*** arXiv:2411.07279. TTT pushed an 8B LLM to 53% on ARC-AGI-1 (61.9% ensembled — matches average human). TTT is unlikely to apply to ARC-AGI-3 directly (interactive, no per-task train pairs), but the *augmentation strategies* transfer.
18. **ARC Prize Foundation (2026) — *ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence.*** arXiv:2603.24621 / arcprize.org/media/ARC_AGI_3_Technical_Report.pdf. Official benchmark paper. Definitive source on game design, evaluation protocol, and scoring formulas.
19. **ARC Prize Foundation (2024) — *ARC Prize 2024: Technical Report.*** arXiv:2412.04604. Documents the ARChitects' winning approach (TTT + augmentation + custom tokenizer, 53.5% private). Good preview of what the field's top engineers are doing.
20. **ARC Prize Foundation (2025) — *ARC Prize 2025: Technical Report.*** arXiv:2601.10904. Documents NVARC (24% on ARC-AGI-2) and the rise of "refinement loop" approaches. Most current overview of the competitive landscape.

---

## 9. How to use this document

1. **Today:** read papers in this order: `[16 → 18 → 20 → 19 → 17] → [1 → 2 → 3] → [9 → 10] → [13]`. ARC canon first, then your reasoning options, then planning, then exploration. ~40 hours of reading total.
2. **This week:** complete Phase 0 (Tasks #1–2 in the task list) — get the env running, enumerate games.
3. **Re-read this doc after Phase 2** (BC baseline). Re-prioritize phases based on actual scores vs targets.
4. **Update §6 risk register** every 4 weeks.

---

## 10. Open questions to resolve before Phase 1 starts

1. ARC Prize API key obtained? (`arcprize.org/api-keys`)
2. Compute budget: which machine runs the long DAgger / RND loops? (RTX 5070 vs cloud).
3. LLM-bootstrap budget for Phase 2: $50 / $100 / $200 ceiling?
4. Coursework split confirmed: Sudoku/Maze runs continue against May 1 deadline; ARC-AGI-3 runs target Nov 2 in parallel?

Once these are answered, Phase 0 starts. The remaining tasks (#1 through #11 in the task list) are the operational breakdown of Phases 0–7 above.
