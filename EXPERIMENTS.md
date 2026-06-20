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

## Test #2 — DP improved + offline val + first VLA runs (PLANNED)

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

## Test #3 — MIT motor mode + re-collection (CONDITIONAL on Test #2)

### Decision input
Whether the step-and-wait pattern persists at `horizon=8, n_action_steps=2`,
EMA smoothing, and `vlim=0.3 rad/s`. If yes → MIT.

### Implementation sketch
- `--motor-mode {pos_vel,mit}` flag added to `arm/controller.py`,
  `record_demo.py`, `teleop.py`, `eval_dp.py`
- Per-joint kp/kd added to `configs/joint_config.yaml`
- Starting guess: arm joints kp=30 kd=1.5, gripper kp=10 kd=0.5, tau_ff=0
- Validation = next teleop session: continuous motion = gains OK; oscillation
  = lower kp; sluggish = raise kp

### Cost
Demos collected under POS_VEL and demos collected under MIT have different
tracking dynamics, so they can't be pooled. If we go MIT, we need to
re-collect (likely a fresh ≥50-episode batch). This is the budget item that
makes Test #3 conditional.
