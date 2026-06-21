# EXPERIMENTS — rebot_arm imitation learning log

Living document. Each test gets a dated section: setup → observations →
**what they suggest** → next-test hypotheses.

The point of this file is NOT to log code bugs (those live in commits) — it's
to capture what we learned about the *task* and *model behavior*, so the next
experiment is grounded in evidence rather than guesswork.

---

## Test #1 — DP from-scratch baseline (2026-06-17 → 2026-06-18)

### Setup (compressed)
- Dataset: 51 pick_tape episodes, 21,876 frames @ 20 Hz, C922 (third-person)
  + D435 (wrist), 7-DOF joint-position state & action
- Policy: LeRobot `DiffusionPolicy`, ResNet18 backbone, crop_shape=(84,84),
  n_obs_steps=2, horizon=16, n_action_steps=8, DDPM 100 steps,
  batch=16, num_workers=0, lr=1e-4, 50k step
- Two variants trained: `single` (intended third-only) and `dual` (third+wrist,
  separate RGB encoders)
- Trained locally on RTX 3500 Ada 12 GB; 8h per variant

### Tech debt fixed during the run
(For reference only — all already in code.)

1. `torchcodec` decoder leaks → patched to PyAV
2. `lerobot.train()` doesn't `init_logging()` → we now do
3. `make_policy()` over-writes `input_features` from dataset → we filter back
4. `eval_dp.py` import path / pyrealsense2 missing in dp_maniskill venv → fixed
5. Leader gripper is servo **0** not 7 — joint map was right, my diagnostic was wrong

### What actually mattered for the science

**A. The single-vs-dual ablation didn't happen.**
Both ckpts ended up with `input_features={state, third_person, wrist}` because
`make_policy()` silently rebuilds from dataset features. We discovered this
only post-hoc by inspecting the saved `config.json`. What we trained instead
was *shared* vs *separate* RGB encoder, both with two cameras. The loss
snapshots are still informative but they answer a different question than the
one we set out to ask. **Test #2 has to re-run the real 1-cam vs 2-cam
comparison.**

**B. Loss plateaued by 30k step — 50k was probably wasted budget.**
```
    step | shared encoder | separate encoder
   10000 |         0.0335 |           0.0309
   20000 |         0.0274 |           0.0251
   30000 |         0.0233 |           0.0227
   40000 |         0.0213 |           0.0225   ← shared/separate diverge
   50000 |         0.0214 |           0.0204   ← shared is flat
```
(80 batches × bs=16 on train data. Not held-out, so this is a fit-quality
proxy — Test #2 must measure honest val MSE.)

**C. Real-arm eval revealed three clean failure modes** (one mechanical,
one perceptual, one kinematic) which together drive the Test #2/#3 plan:

#### Failure 1 — Jerky, "step-and-wait" motion
What we saw: arm moves to a target, pauses, moves to next target, pauses.
Not what a human teleop trace looks like.

Hypothesis ladder (from most-likely-mechanical to most-likely-policy):
1. **Motor mode** (most likely): `Mode.POS_VEL` runs a firmware-side
   trapezoidal trajectory generator. When DP issues a new target before the
   previous segment completes, the firmware re-segments — visible as
   step-and-wait. Direct PD (`Mode.MIT`) would track continuously.
2. **Control loop rate vs vlim mismatch**: at rate=15 Hz, vlim=0.5 rad/s,
   the arm can travel at most 0.033 rad per tick. If DP outputs targets
   ~0.1 rad apart, the arm is constantly chasing — interpreted as jerky.
3. **DP replanning cadence**: horizon=16, n_action_steps=8 means a new
   plan every 0.4 s. The plans aren't smooth across boundaries — small
   discontinuity at the 8-step junction. Lower `n_action_steps` would
   replan more often but make boundary jumps shorter.

Mitigations applied to `eval_dp.py` (incomplete fix — addresses 2 & 3 only):
rate 15→10 Hz, vlim 0.5→0.3 rad/s, new EMA `--smooth 0.6`. Improved but
did not eliminate jerkiness → hypothesis #1 is doing most of the work.
**Test #3 = MIT mode.**

#### Failure 2 — Object localization off by 1-3 cm
What we saw: arm goes to a position *near* the tape rather than on it.
Repeatable across episodes. Not random.

Hypothesis ladder:
1. **Image resolution lost to crop**: `crop_shape=(84,84)` on 480×640 is a
   7.6× linear downsample. The tape edge against the silver-box rim is
   probably below this resolution. Fine localization needs more pixels
   where the gripper-object interface is. Highest priority to test.
2. **Demo dataset diversity**: j1 (base sweep) range is only 1.30 rad
   across all 51 episodes; j5 std is 0.12 rad. The tape was placed in too
   narrow a region during collection. The policy memorized that region —
   when the test placement is even 5 cm off, it interpolates poorly.
3. **Over-fit at 50k**: shared-encoder loss flat 0.0213→0.0214 from
   40k→50k while still on train data. Could indicate over-fit; need val
   curve to confirm.
4. **Wrist camera framing at close range**: D435 may be occluded by the
   gripper during the last 5 cm of approach. Not verified.

Test #2 attacks #1 (no crop) and #3 (val split + use earlier ckpts).
Test #4 (data re-collection) attacks #2.

#### Failure 3 — j4 collides with j2/j3 on home
What we saw: pressing R, j2/j3 swing back before j4 unfolds → collision.

Hypothesis ladder:
1. Kinematic interference: with j4 folded back, the j3 link sweeps
   through the same space that j4 occupies. Solving order matters.
2. We use sequential homing (not joint-space planning), so order is
   trivially configurable.

Fix ✅: `eval_dp.py --home-order` default `4,1,2,3,5,6,7`.

**D. Dataset analysis pointed at recall problems before training even started.**

We didn't catch this until *after* the run, but `eval_ckpt_loss.py`'s dataset
scan turned up two facts that should change how we collect demos:

| Per-joint range across all 51 episodes | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
|---|---|---|---|---|---|---|---|
| range (rad) | **1.30** | 2.12 | 1.90 | 1.81 | **1.14** | 1.24 | 2.85 |
| std (rad) | **0.19** | 0.68 | 0.45 | 0.31 | **0.12** | 0.28 | 0.85 |

- **j1 / j5 are barely used.** The tape was always near the same horizontal
  position and never required wrist yaw. Policy will not generalize to
  tape placed off-center or rotated — directly explains Failure 2.#2.

- **Episode length spans 313 → 932 frames (3× ratio).** Median 415, so
  half the episodes are < 21 s and half are > 21 s. The grasp event itself
  takes a few hundred ms; the rest is approach and recovery. Long episodes
  are mostly *post-grasp holding*, which adds zero learning signal but does
  shift action statistics. Trimming after the detected grasp event would
  tighten the distribution.

- **Zero episodes have `follower_tau[:, 6] > 0.5 Nm`** despite all 51 being
  successful grasps. The torque-based contact event isn't being recorded —
  it's only used by `record_demo.py`'s contact-aware lock rule, not written
  into the dataset. The policy thus has no "I have grasped" signal and must
  infer it from gripper position alone, which is degenerate at the grasp
  boundary (one position can mean either "closing on object" or "closed on
  air"). Need to relabel.

### Hypotheses going into Test #2 (the ones we'll actually test)

| H# | Hypothesis | Predicted outcome if true | Test |
|---|---|---|---|
| H1 | crop_shape=(84,84) is the main loss-of-precision driver | No-crop run reaches lower val MSE at 30k step; real-arm reaches the tape more often | Run with crop_shape=None |
| H2 | Most of the gain is in the encoder, not the diffusion head | resnet34 noticeably beats resnet18 at same step | Two backbones side by side |
| H3 | Shorter chunk + faster replan reduces jerkiness independently of motor mode | `horizon=8, n_action_steps=2` at same vlim is visibly smoother in real-arm eval | Run with new horizon, eval same hardware setup |
| H4 | One camera is enough — the wrist view adds noise more than info, because it's occluded near contact | True 1-cam run matches or beats 2-cam val MSE | True single-cam variant in Test #2 (the one we *meant* to run in Test #1) |
| H5 | Trimming episodes after grasp event tightens action distribution and reduces train loss | Same model, trimmed dataset → faster convergence | Compare relabeled-and-trimmed vs raw |
| H6 | Pretrained VLA generalizes from 51 demos better than from-scratch DP | SmolVLA / pi0fast val MSE lower than best DP at convergence | Fine-tune both on same data |

### Open questions (no clean way to test in Test #2, parked)
- Does the D435 wrist actually see the tape during the final approach, or
  does the gripper occlude it? Would need to log wrist frames at grasp ± 5
  frames and visually inspect.
- Action delta is recorded relative to *current pos*, but motor moves toward
  *commanded target* — is the recorded `action[t]` what the policy should
  predict, or should it be `action[t] - state[t]` (delta-relative)? Standard
  DP uses absolute targets, but for high-precision tasks delta-relative is
  sometimes better. Not testing now to avoid changing too many variables.

---

## Test #2 — DP improved + offline val + first VLA runs (PLAN as written; results below)

### Plan
6 runs in parallel on jiagpu8_yz (8× A40 48 GB), train+val split 41/10
episodes. Each tests one of the hypotheses above.

| GPU | Run | Tests | Image | Backbone | batch | h / n_act | Variant |
|---|---|---|---|---|---|---|---|
| 0 | DP-A | H1, H2 | full 480×640, no crop | resnet34 | 48 | 8 / 2 | dual-cam, grasp-trimmed |
| 1 | DP-B | H2 control | full 480×640, no crop | resnet18 | 64 | 8 / 2 | dual-cam, grasp-trimmed |
| 2 | DP-C | H4 (real 1-cam) | full 480×640, no crop | resnet34 | 48 | 8 / 2 | **single-cam, filter verified** |
| 3 | DP-D | H1 sanity, smaller image | resize 320×240, no crop | resnet34 | 96 | 8 / 2 | dual-cam, grasp-trimmed |
| 4 | SmolVLA | H6 | per model preset | smolvla pretrained | per preset | per preset | full fine-tune |
| 5 | pi0fast | H6 | per model preset | PaliGemma+flow LoRA | bf16 LoRA r=16 | per preset | LoRA fine-tune |

Held-out 10 episodes used by `eval_ckpt_loss.py --val-split` to measure
honest val MSE at each checkpoint. Top 2 runs go to real-arm eval.

### What we will NOT vary in Test #2 (to keep ablations clean)
- Recording fps (20 Hz)
- DDPM scheduler for DP runs (flow matching is exclusive to pi0fast)
- Absolute joint target action space (no delta-relative experiment)
- Motor mode = `pos_vel` (MIT is Test #3, separate)
- Gripper kp/kd (will tune in Test #3)

### Decision criteria for Test #3
If Test #2 still shows the step-and-wait pattern after H3's shorter chunks
are in place, MIT mode is the remaining lever and worth the extra
re-collection effort. If H3 alone smooths it out, MIT becomes nice-to-have.

---

## Side experiment — lerobot 0.5 upgrade feasibility (2026-06-20)

### Why
Tests 1/2 ran on lerobot 0.3.2 (pinned to py3.11). After hitting a CPU-bound
conversion bottleneck and discovering 0.5 ships a **StreamingVideoEncoder
that bypasses PNG intermediates AND accepts h264_nvenc**, we built a parallel
local venv to measure how much we'd have to rewrite to migrate.

### Setup
- `~/code/rebot_arm/.venv_le05` = py3.12.9 (uv-managed) + lerobot 0.5.1 +
  torch 2.10.0+cu128 + torchcodec 0.10 + transformers 4.57 + peft + av + h5py
- Smoke surface: train_dp.py, convert_to_lerobot.py (TODO), eval_ckpt_loss.py
  (TODO), train_vla.py (TODO), relabel_grasp.py (lerobot-free, OK)

### What works without code changes
- All policy classes import: `DiffusionPolicy`, `PI0Policy`,
  **`PI05Policy`** (new in 0.5 — pi0.5 is here), `SmolVLAPolicy`,
  `PI0FASTPolicy`
- `StreamingVideoEncoder` is present, signature
  `(fps, vcodec='libsvtav1', pix_fmt='yuv420p', g=2, crf=30, preset=None,
   queue_maxsize=30, encoder_threads=None)`
- `LeRobotDataset.create` now accepts `vcodec=`, `streaming_encoding=`,
  `encoder_queue_maxsize=`, `encoder_threads=` as first-class params →
  no more monkey-patching of `encode_video_frames`
- `init_logging`, `dataset_to_policy_features`, `PolicyFeature`,
  `FeatureType`, `NormalizationMode` all still at the same import paths

### What broke (and how we patched it)
| # | Breakage | Fix |
|---|---|---|
| 1 | `lerobot.scripts.train` no longer exists | now `lerobot.scripts.lerobot_train`. train_dp.py uses try/except to load whichever is available, so the script is 0.3-and-0.5 compatible. |
| 2 | Dataset format v2.1 → **v3.0**, with `BackwardCompatibilityError` on load | Official converter exists: `python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id <name> --root <path> --push-to-hub False`. Runs in seconds (no video re-encode; remuxes parquet metadata + renames layout). Backs up the old dataset to `<name>_old`. |
| 3 | `DiffusionConfig.validate_features` rejects `crop_shape >= image_shape` (was ≤ in 0.3) | When `--no-crop`, pass `(H-1, W-1)` instead of `(H, W)`. Functionally identical, satisfies the strict check. |
| 4 | PolicyFeature image shape must be **(C, H, W)**; 0.3 accepted (H, W, C) | train_dp.py now detects HWC (trailing dim == 3) and transposes to CHW. |
| 5 | `lerobot_dataset.decode_video_frames` no longer exists (was a re-export in 0.3) | Our PyAV monkey-patch already guards with `hasattr` and only sets it if present — no-op on 0.5. |

### Verified working on lerobot 0.5 after the patches
- `train_dp.py --steps 15 --batch-size 4 --no-crop --horizon 8` against the
  v3.0 dataset: 15 steps in 8 s, loss 1.14 → 1.12, checkpoint saved. **train
  loop runs cleanly.**

### Big open question we didn't yet test
**Will `DiffusionPolicy.from_pretrained` load a 0.3-trained checkpoint
under 0.5?** Almost certainly no — our existing 50k ckpts:
- store `input_features` in HWC order (0.5 expects CHW)
- have a `config.json` schema that may have shifted between versions
- were saved with `crop_shape=(84,84)` which 0.5 may interpret differently

**Implication:** migrating to 0.5 effectively orphans the existing Test #1
ckpts. We'd need to retrain. This is the dominant cost — not the API patches.

### Not yet smoke tested under 0.5
- `convert_to_lerobot.py` — should drop its monkey-patch and use the new
  `vcodec=` / `streaming_encoding=True` params on `LeRobotDataset.create`.
  When done, `--codec h264_nvenc` is expected to work without further
  changes.
- `eval_ckpt_loss.py` — depends on whether it loads a 0.5-trained ckpt
  (yes, no problem) or a 0.3-trained one (likely no — see above).
- `train_vla.py` — `PI05Policy` is new, `PI0FASTPolicy` config field names
  may have shifted (need to re-probe). `--use-lora` path also untested.

### Recommendation (going into Test #3 planning)
- **Stay on 0.3.2 for Test #2** — re-running with all the 0.5 patches gives
  us a parallel, cleaner stack later, but the Test #2 server runs are
  already launched/being launched under 0.3, and 0.5 won't load those ckpts.
- **Move to 0.5 for Test #4 (post-MIT-collection)** — when we re-collect
  demos for MIT-mode policies, we should also do that under 0.5 from the
  start. Benefits realised:
  - Streaming conversion + h264_nvenc → fast iteration on dataset variants
  - pi0.5 (better generalization than pi0fast) available
  - Cleaner code (no monkey-patches for video, logging, or input_features
    filter — 0.5 fixed all three at the source)
- **Keep `.venv_le05` around** as the migration sandbox. The patches in
  train_dp.py (try/except import, HWC↔CHW, crop_shape −1) are 0.3-safe,
  so the script stays a drop-in for either env.

---

## Test #2 — RESULTS (2026-06-20 → 2026-06-21)

### Outcome summary
| Run | Config | Train loss (50K) | Val MSE (10K) | Val MSE (50K) | Real-arm result |
|---|---|---|---|---|---|
| DP_A | dual / r34 / bs48 | 0.006 | 0.665 | 1.007 | (not tested — DP path dead, see below) |
| DP_B | dual / r18 / bs64 | **0.003** | 0.579 | 0.975 | (not tested) |
| DP_C | single / r34 / bs48 | 0.005 | 0.587 | 1.072 | **wanders near tape, cannot grasp** |
| DP_D | dual / r34 / bs32 | 0.009 | **0.481** | 1.043 | **stays at home, only twitches** |
| SmolVLA | full fine-tune | 0.012 | 0.043 → **0.011** ↓ | (same) | **reaches near tape, cannot grasp precisely** |
| pi0fast | LoRA fine-tune | — | — | — | **never started** — HF repo `lerobot/pi0fast-base` has no `policy_preprocessor.json`; lerobot 0.5 refuses to load |
| pi05 (added 2026-06-21) | (replaced pi0fast) | — | — | — | **never started** — HF `lerobot/pi05_base` references a processor step `relative_actions_processor` that our lerobot 0.5.1 registry does not have |

### Findings on DP (catastrophic, 4 runs all fail in real-arm)
**F1. Train loss is decorrelated from val MSE.**
DP_B has the lowest train loss (0.003) but second-worst val MSE at step 10K
(0.579). DP_D has the worst train loss (0.009) but **best** val MSE at 10K
(0.481). Val ranking ≠ train ranking. Train loss cannot be used as a model
selection signal here.

**F2. Val MSE itself is also decorrelated from real-arm performance.**
DP_D (best val) is the WORST in real-arm: only twitches at home, never reaches
out. DP_C (third-best val) at least approaches the object. Honest val MSE is
better than train loss but still does not predict deployment behavior.

**F3. Val loss rises monotonically from step 10K onward — severe overfitting.**
At step 10K: 0.48–0.67. At step 50K: 0.97–1.07. Both DP_B (r18) and DP_D
(smaller batch) overfit comparably. The 50K ckpt is much worse than 10K for
every DP run; 50K step is wasted budget on this dataset.

**F4. Real-arm failure modes (from eval on 10K and 20K ckpts).**
- Both DP_C and DP_D at step 10K: j1 drifts left, j2/j3 do not move, j4
  eventually drops and motors fault (red LED on follower) before any task
  progress. This was traced to a normalization bug — DP ckpts in lerobot 0.5
  carry `policy_preprocessor.json` + `policy_postprocessor.json` and the
  postproc is what un-normalizes the [-1, 1] MIN_MAX action back to physical
  joint angles. After fix (eval_dp.py now loads postproc for all policy
  types), DP can run without fault — but still cannot solve the task:
  - DP_C 10K: gripper approaches the rough region of the tape, oscillates in
    front of the object ~5 cm out, never commits to closure. No grasp →
    no place-down step ever attempted.
  - DP_D 10K: stays in the home neighbourhood, only small twitches, won't
    reach out.
  - DP_C 20K, DP_D 20K: action range becomes **smaller** than at 10K (more
    confinement to home), worse than 10K.
- Pattern (DP_D < DP_C across 10K and 20K) is consistent with the
  user-articulated hypothesis below.

**F5. Saliency analysis confirms DP doesn't use vision.**
`scripts/visualize_attention.py` measures ∂‖action‖/∂pixels. On a held-out
demo (episode 5), DP_D 10K's saliency on third_person image at the moment
the demo is at 75% (approaching the tape, where vision matters most) is
**third_max = 0.0000**. Average across the 5 sampled frames is < 0.001.
SmolVLA on the SAME frames hits 0.522 at the approach moment — 1000× higher
gradient w.r.t. image pixels. **DP makes its action prediction essentially
independent of the input image** — concrete evidence of the "action lookup
table" / memorization hypothesis from Lin et al. 2025 (arxiv 2505.05787).

**F6. The chunk + chunk-replan interaction creates a self-reinforcing
home-state deadlock (user-articulated, supported by literature).**
- DP horizon=8, n_action_steps=2 means: predict 8 steps ahead, execute 2,
  then re-plan from the new state. At eval start the arm is at home
  (s ≈ 0); training demo data near home is mostly the first frame of an
  episode where the leader-arm action is small. So the policy maps
  "s ≈ home" → "small magnitude chunk". Execute 2 small actions → state is
  still ≈ home → re-plan → still small actions → ad infinitum.
- This is concretely "DP overfit to *local* state–action mappings" and DP
  has no inductive bias (no goal token, no language input, no
  cross-task pretraining) to break out. Lin 2025 frames the same thing
  more strongly: DP IS a (visually-conditioned) nearest-neighbour lookup.
- Test #1 used `n_action_steps=8` so each plan committed for 0.8 s before
  re-evaluation. With our smaller `n_action_steps=2` (Test #2 H3) the
  policy gets to re-cement "stay" more often → DP_D is WORSE than what we
  saw in Test #1, in this respect. H3 backfired for DP. (Probably fine for
  VLA; left for separate test.)
- 20K ckpt has even smaller action range than 10K — overfitting
  progressively tightens the local trajectory band. Val MSE is up at 20K
  too. Both diagnostics agree that for THIS dataset DP_*  monotonically
  gets worse past ~10K step.

### Findings on SmolVLA (partial success, instructive failure)
**G1. SmolVLA val MSE monotonically DECREASES (no overfitting on val).**
0.043 (5K) → 0.028 → 0.017 → 0.014 → 0.013 → 0.011 (30K). Opposite of DP.
Pretrained VLA prior + language condition act as a regulariser; small data
(41 demos) is enough to fine-tune style without destroying the underlying
manipulation primitives.

**G2. SmolVLA can execute coarse approach but cannot grasp precisely.**
Real-arm with the corrected postproc: arm reaches forward, approaches the
tape area, hovers / wanders within ~3–8 cm of the object, never closes the
gripper on it. No grasp → no place-down phase tested.

**G3. SmolVLA saliency goes near-zero mid-episode (open-loop coast).**
Saliency at frame 247 (≈ 50% of episode): third_max = 0.006, near DP-D
levels. Saliency peaks at frame 371 (≈ 75%, approach phase): 0.522. So the
model "looks at" the image at the grasp moment but coasts open-loop through
the middle. At deploy time the START state distribution differs from train
(at home; in train demo, the leader had been driving for a while already),
so the early-mid coast carries the arm off-course before the grasp-phase
attention can fix it. By the time it "looks", the geometry is already
slightly wrong → off by a few cm → no grasp.

**G4. n_action_steps = 50 (default) compounds G3.**
At our 10 Hz control loop, chunk_size=50 means each plan commits to **5
seconds of open-loop action**. With the wide tape-position distribution the
model needs to compensate for, 5 s of open-loop is too long. Likely
Test #3 lever: drop SmolVLA n_action_steps to 10–20 to force more frequent
re-planning at the cost of some smoothness.

### Why DP fails / why SmolVLA partially works — literature support
The Test #2 failure modes line up exactly with three independent 2024–2025
papers. We did not seek them out before the run; we found them after the
fact while looking for confirmation of the hypothesis we reached from F1–F6.

**Paper 1 — Lin et al. 2025, "Demystifying Diffusion Policies: Action
Memorization and Simple Lookup Table Alternatives" (arXiv 2505.05787).**
Direct quote: *"diffusion policies essentially memorize an action lookup
table."* They demonstrate this by feeding fully-OOD images (cats and dogs)
to a trained DP and showing the action sequence the policy outputs is still
from the training set. They propose Action Lookup Table (ALT) which makes
the lookup explicit (contrastive image encoder + nearest-neighbour
retrieval); ALT matches DP on small data at 0.0034× inference time and
0.0085× memory. The whole "diffusion + UNet" superstructure in DP turns
out to be doing nearest-neighbour retrieval — our 0.0000 saliency reading
(F5) is a concrete realisation of this on our dataset.

**Paper 2 — Sentinel: "Unpacking Failure Modes of Generative Policies"
(arXiv 2410.04640, 2024).** Identifies two failure categories that map
cleanly onto our Test #2 observations:
- **Erratic** — temporal action inconsistency, oscillation, refusal to
  commit. Direct match to DP_D "stays at home, only twitches" (F4).
- **Task progression** — confidently and consistently takes actions that
  don't solve the task. Direct match to SmolVLA "wanders near tape, can't
  grasp" (G2). They propose a VLM-based runtime monitor; we are not
  building this now but Methodology section uses the same principle —
  saliency + video as the actual ground-truth signal.

**Paper 3 — "Action Chunking and Exploratory Data Collection Yield
Exponential Improvements" (arXiv 2507.09061, 2025).** Two prescriptions:
- **Longer action chunks** (commit further before re-planning) →
  exponentially smaller compounding error. Our Test #2 H3 went the OTHER
  direction (n_action_steps 8 → 2) on the hypothesis "more frequent
  re-plan = more reactive". The paper says we made it worse, and our DP_D
  worse-than-Test#1 result confirms.
- **Exploratory data collection** — train on demonstrations that
  deliberately leave the expert manifold so the model sees recovery
  actions. Matches our Test #3 plan to randomise object placement and
  intentionally let the leader-arm wander during collection.

**The DP vs SmolVLA architectural difference, table form.** This is the
user's own analysis from the post-Test-#2 discussion, recorded here as the
working model going into Test #3:

| Factor | DP | SmolVLA |
|---|---|---|
| Pretraining data | 0 (ImageNet on visual backbone is task-unrelated) | HuggingFace cross-task robot data + SmolVLM2 |
| Base motor primitives | Learned from scratch on 41 demos | Pretrained "reach / grasp / place" priors |
| Goal representation | None | Language token (`"pick_tape"`) as condition |
| Default chunk size | 8 (we set; library default 16) | 50 (default) |
| Action prediction conditioned on | f(image, state) | f(image, state, language) ← critical |

The two lines about chunk size and goal representation together explain
why SmolVLA can complete the reach motion from home while DP can't:
SmolVLA's language token tells it *"task: pick_tape"* regardless of what
the image shows; combined with a 5-second open-loop chunk that commits to
the whole reach phase, it never gets a chance to lock into a "stay" loop.
DP has no goal token + a 0.4-second commit window → maximally easy to
lock onto "stay" if the home-state action prior is small.

**Conclusion going into Test #3.** Small-dataset (≤ 100 demos), multi-
sub-action real-arm IL is not the regime DP is fit for. DP is fit for
single primitive (e.g. push, pour) + ≥ hundreds of demos + light OOD —
the original DP paper setup. For our setup, VLA + LoRA / full fine-tune
is the realistic path; an ALT-style nearest-neighbour baseline is a
cheap sanity check we could also run.

### Findings on the task itself (recorded for Test #3 to act on)
- The actual task is **TWO sub-actions chained**: (a) grasp the tape from
  the table, (b) place it inside the silver box. None of the 5 runs in
  Test #2 even reached step (b) because none succeeded at step (a). For
  Test #3 we should add a "phase tag" to demos (approach / grasp / lift /
  place / release) so the next analysis can localise where each model
  fails — currently we only know "doesn't grasp".

### Tech-debt items discovered during Test #2 (already fixed)
1. lerobot 0.5 splits normalization into a separate
   `policy_postprocessor.json` for ALL policy types (DP, SmolVLA, PI05,
   PI0Fast). Our `eval_dp.py` initially assumed DP folded normalization
   into the model; led to motors faulting on first real-arm run. Fixed:
   load both preproc + postproc for every policy type.
2. lerobot 0.5 made `select_action` apply `torch.no_grad` internally,
   breaking gradient-based saliency. `visualize_attention.py` uses
   `policy.forward(...)` with a zero action target as a fallback path.
3. The HF Hub model paths for VLA bases are inconsistently
   hyphen/underscore-named, and version-skewed between lerobot 0.5.0,
   0.5.1, and the unreleased dev branch the ckpts were uploaded from.
   pi0fast-base = no policy_preprocessor.json (uploaded pre-0.5);
   pi05_base = uses a processor step that didn't make it into 0.5.1.
   No clean fix without either (a) building lerobot from main, or (b)
   re-saving a ckpt with our installed lerobot.

---

## Methodology — standard post-run analysis (adopted after Test #2)

For every future training run we will produce, on the SAME held-out demo
episode (or on saved frames of the real-arm eval):

1. **Real-arm video** — recorded during `eval_dp.py`. Cover at minimum: SPACE→home→episode_start, full reach phase, grasp attempt, place attempt, release. Save the third_person + wrist streams.
2. **Saliency heatmap grid** — run `scripts/visualize_attention.py --grid 8`
   on (a) the trained ckpt and (b) for each saved eval segment, produce
   one PNG with 8 evenly-spaced frames showing RGB + saliency overlay.
   Print `third_max` / `wrist_max` per frame.
3. **Combined narrative** in the EXPERIMENTS.md entry for that test:
   - Where the model went visually wrong (frame numbers + heatmap shows
     the model was/wasn't looking at the object).
   - Where the model went kinematically wrong (joint trajectory peeled off
     from demo trajectory).
   - Decision on next action: data, model, or hyperparams.

Per the Demystifying / Sentinel papers, neither train loss nor val MSE
predicts deploy behaviour for DP-class policies. Saliency + video gives
the only honest signal; we will not skip it again.

---

## Test #3 — VLA-focused improvements + pi0.5 path

(supersedes the earlier "MIT motor mode" Test #3 plan — MIT is deprioritised
since DP is being retired; we keep the MIT branch as an optional
side-experiment for the next data collection round, if needed.)

### Two parallel work-streams

**A. SmolVLA tuning — try to fix the mid-episode open-loop / precision gap**

Levers to try (rank-ordered by expected ROI):

1. **Drop SmolVLA `n_action_steps` from 50 → 20 or 10.** Direct attack on G3
   / G4. More frequent re-planning means image saliency contributes more
   often, less coasting through OOD mid-states. Cost: more compute per
   second + slightly less smooth output. Likely worth `--n-action-steps 20`
   for a first try.
2. **Re-collect demos with object-position randomization.** 51 demos with
   j1 base range only 1.30 rad means SmolVLA only sees one narrow region.
   Goal: 100 demos with an 8×8 (or similar) grid of object placements
   covering the reachable workspace. Phase tags (approach/grasp/lift/
   place/release) added in `relabel_grasp.py`-style post-processing.
3. **Add an explicit grasp-event signal to the dataset** (we already wrote
   `relabel_grasp.py` for this — 46/51 of the Test #2 dataset got tagged).
   Either train SmolVLA with the grasp event as an auxiliary output (multi-
   task regularisation), or use it to trim post-grasp tails from training
   data.
4. **Try MIT motor mode for the new data collection,** for cleaner tracking
   dynamics. Optional. The motor mode is independent of the model — we can
   compare SmolVLA-trained-on-POS_VEL vs SmolVLA-trained-on-MIT on the
   same task.
5. **Sweep `--task-prompt`.** During SmolVLA fine-tuning the prompt was
   `"pick_tape"`. At eval we used the same. Try at deploy:
   `"pick up the black tape and place it in the silver box"` —
   longer/richer prompt may activate different VLM priors.

**B. Get pi0.5 working — the lerobot 0.5 ckpt loading problem**

Two paths:

1. **Self-host a pi0.5 ckpt that matches our installed lerobot 0.5.1.**
   Fine-tune the official `lerobot/pi05_base` weights via `train_vla.py`
   on the server, save the resulting ckpt with our installed lerobot
   version → the saved `policy_preprocessor.json` references only steps
   our registry has. But this requires the BASE load itself to succeed,
   which currently fails on `relative_actions_processor`. So:
2. **Patch the registry to alias `relative_actions_processor` → an
   existing step.** The list of available steps includes
   `absolute_actions_processor` and `delta_actions_processor`. From the
   semantic, "relative" is probably equivalent to "delta_actions_processor".
   Register an alias decorator under that name and re-try the load.

Order: try B.2 first (a one-line lerobot patch). If that lets the model
load + train, the resulting ckpt is automatically compatible. If pi0.5
turns out structurally incompatible with 0.5.1, we either build lerobot
from main (rebuild the whole `.venv_le05`) or revisit when 0.5.2 drops.

### What we will NOT do in Test #3
- No more DP. Test #2 closed that book. We may keep a single small DP
  run in future tests strictly as a baseline ablation, but no engineering
  effort goes into improving it.
- No more changes to the eval rig itself (rate/vlim/smooth/home-order
  defaults are settled). The variables in Test #3 are model + data.

### Expected outcome
A SmolVLA variant that completes the full task chain (approach → grasp →
lift → place → release) on ≥30% of trials in real-arm eval, OR a pi0.5
variant doing the same. Pure precision (closing inside 1 cm vs 5 cm) is
the secondary success criterion.
