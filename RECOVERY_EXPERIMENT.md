# Bounded recovery experiments (distinct opt-in hypotheses)

`recovery_experiment.py` is separate from default training. It never touches
baseline STOP files, `train.py`, `insertion/env.py`, or `insertion/config.py`.
It does not implement correlated exploration or change the PPO distribution.
This is a legacy, opt-in experiment, not the active native training workflow.
The commands below reproduce its protocol; they are not instructions to resume it.

## Current evidence and automatic skip

The actual-motion diagnostic at
`runs/continuous_no_intent/recovery_motion_19610/summary.json` currently reports
`curriculum_gate.pilot_justified=false`: only 29 full one-second blocked-failure
windows, of which 27.6% have actual radius below 0.5 mm. The gate requires at
least 30 such failures **and** at least 50% below that radius. This does not
support a broad “commands are too small” hypothesis. Threshold-sensitivity
analysis is owned by the diagnostic, not this driver.
The driver reads that actual gate, verifies the frozen checkpoint identity and
full-noise condition, and independently enforces those same population thresholds.
Missing/false/incomplete/mismatched evidence yields `waiting_for_motion_gate`
in `status.json` and `summary.json`, with the diagnostic evidence and hash.
**No evaluation or training subprocess is started. There is no force/override flag.**
The driver does not rewrite or reinterpret diagnostic evidence to authorize training.
This remains the default `small-motion` hypothesis. Its existing output and failed
gate are preserved; a new hypothesis uses a different output directory.

## Prospective oscillatory-recovery pilot

The separately authorized `--hypothesis oscillatory-recovery` rationale is that
on-policy near-mouth approaches may improve **ineffective oscillatory search**,
not that actual movement is globally too small or exploration std needs enlarging.
Its evidence gate reads the same frozen-source/full-noise diagnostic and
`sensitivity.json`: >=60 mean-action timeout-tail windows, median actual radius
>=0.5 mm, >=10 actual direction reversals per two seconds, median depth gain
<=0.1 mm. Current evidence is 120 / 1.417 mm / 21 / 0.024 mm. This is exploratory
eligibility, **not causal proof** and not success of the original small-motion gate.
The original gate is embedded unchanged in the new report.

Hypothesis, source hash, diagnostic/sensitivity hashes and gate evidence are
immutable per output. Changed evidence on resumption requires a distinct output.
No overrides or post-action noise are provided.

Oscillatory-recovery smoke (one update per arm, no promotion):

```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -r ~/micromamba -n mj \
  python -u recovery_experiment.py --hypothesis oscillatory-recovery \
  --smoke --train-envs 4 --hours 2 \
  --output runs/recovery_oscillation_19610_smoke
```

The bounded pilot uses **300** updates per arm/seed
(the CLI's hard maximum remains 500), with original per-shape counts preserved:

```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -r ~/micromamba -n mj \
  python -u recovery_experiment.py --hypothesis oscillatory-recovery \
  --max-updates 300 --eval-every 100 --hours 18 --eval-envs 64 \
  --seeds 31415 27182 --eval-seeds 42 2026 \
  --final-seeds 104729 130363 \
  --output runs/recovery_oscillation_19610
```

Final test seeds 104729/130363 are disjoint from training and validation 42/2026;
no final-seed evaluation occurs before checkpoint selection. Selection uses
validation only; final tests run automatically for eligible selected candidates.
Checkpoints record validation/final seed sets, hypothesis and evidence gate.
The larger evaluation batch reduces sequential episode waves without changing
trials or training counts. Throughput and final-test completion within 18 hours
are not guaranteed: the persistent deadline takes priority and reports a bounded
stop/inconclusive result rather than waiving guards or silently extending time.

Secondary evaluation outcomes use physical trajectories captured after the
physics step **before automatic reset**, including terminal state: final-two-second
depth gain, radius/path, >120° direction reversals after dropping XY segments
shorter than 0.005 mm, and contact/nonprogress occupancy. Contact uses the norm of
true translational `qfrc_constraint` (>1 N), **not noisy actor load or sensor wrench**.
Secondary outcomes do not alter rewards or replace success/safety guards.
Terminal-inclusive tails differ from the diagnostic audit's terminal-excluded
timeout tails; results explicitly label this and count eligible full tails.

The experiment-only wrapper uses `nconmax=512,njmax=2048` for both arms and
evaluation after baseline allocation overflowed in an early smoke. It leaves core
defaults and solver settings unchanged. The driver rejects worker logs containing
broadphase overflow. Corrected one-update smoke results are resource checks, not
efficacy evidence; native EPA/cylinder–mesh contact warnings remain a limitation.

## Small-motion gate check and smoke

```bash
micromamba run -r ~/micromamba -n mj python recovery_experiment.py \
  --output runs/recovery_experiment_19610
```

This default `small-motion` command currently produces a gated skip, not training.
Only genuinely supporting diagnostic evidence permits its smoke, which must use
a **separate** output directory:

```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -r ~/micromamba -n mj \
  python -u recovery_experiment.py --smoke --train-envs 4 --hours 2 \
  --output runs/recovery_experiment_19610_smoke
```

Smoke evaluates four trials/shape/seed for ordinary plus all three recovery
challenges, then performs one PPO update and repeats evaluation for each arm.
It cannot promote a policy. Reduced smoke training counts are an explicit
resource check, not the full experiment's training layout.

## Matched protocol and decision rules

Defaults are one GPU, an 18-hour persistent deadline, training seeds
31415/27182, and validation seeds 42/2026. Each arm/seed starts from
the same frozen iteration-19610 checkpoint, preserving the saved runner
configuration, actor13/critic57 two-layer GRU128, Adam moments and learning rate,
normalization, reward, success, dynamics, observation noise, action limits and
per-shape training counts. Entropy is explicitly 0.001. Noise and physical
difficulty are frozen at 1; clearance is fixed to the saved target clearance
(0.1 mm). Success/clearance curricula and adaptive sampling are disabled.

At each 100-update budget (maximum 500 per arm/seed), control and recovery are
trained sequentially and evaluated at the **same** update budget. Validation
uses 128 trials/shape/seed, including heldout circle, under ordinary and all
three fixed recovery-start challenges with train-profile physical randomization.
Ordinary control and recovery both use the same full randomization.

Recovery training uses independent Bernoulli episode assignment: expected 75%
ordinary, 25% designated near-mouth approaches; realized counts are saved.
The wrapper first performs the unmodified ordinary reset, preserving its RNG
stream and physical sampling. Only designated positions change. Orientation,
grasp/calibration errors, noisy observations and all physical randomization
remain unchanged. The actual lowest peg point is placed 0.2–0.5 mm **above**
the hole plane, with radial offset initially 0.5–1 mm. Native CPU MuJoCo tests
verify collision-free geometry for every shape and nonzero grasp errors.
No scripted dynamics steps, injected actor truth, or GRU burn-in are used:
the policy itself creates contact through normal PPO episodes. Near-mouth
height distinguishes these starts from ordinary ±3 mm XY starts.

Promotion to 1–2 mm, then 2–3 mm offsets requires recovery-specific heldout
success lower confidence bounds >=80%, positive recovery improvements over
both matched control and frozen source, ordinary noninferiority within 2 pp,
and safety deterioration <=1 pp. These conditions apply to pooled and every
shape's statistics. Ordinary reset samples never change with stage.

Confident ordinary regression >2 pp or safety regression >1 pp stops that
seed's pair; control regression against frozen also stops the pair.
At >=300 updates, no observed promising recovery improvement is a futility
stop. Positive point estimates alone may warrant **evaluation**, not promotion.
The best eligible checkpoint is selected by a fixed stage-zero heldout
challenge score, not training success or merely the latest checkpoint.
Matched selected control, recovery and frozen checkpoints receive final
evaluations >=512 trials/shape/seed. Final promotion requires all guards for
every training seed, plus the gate still being satisfied.

## Artifacts, interruption, and interpretation

`status.json` is the machine-readable current state. `summary.json` is regenerated
from results and the **current actual** diagnostic gate. It distinguishes
waiting, running, bounded stop, failed and complete/inconclusive; logs include
worker exceptions, with no silent fallbacks. Evaluation JSON contains individual
episodes, per-shape success and safety Wilson intervals, termination counts,
force/moment median/p95/p99/max, and contact incidence (>1 N episode peak).
Raw success and safety differences use conservative unpaired Wilson interval
differences; identical seeds do not imply valid pairing.
`active_worker.json` records the child PID, log and training `progress.json` path;
each completed-update checkpoint refreshes that progress heartbeat.
Per-update progress also reports allocated/reserved/peak CUDA memory and free VRAM.
Completed updates release unused PyTorch CUDA cache to avoid retaining recurrent
PPO workspaces between rollouts. Capacity arrays scale with environment count;
small-count smoke does not establish full-count memory headroom or throughput.
Inspect progress telemetry rather than assuming the deadline will suffice.
Worker subprocess logs are unbuffered. Evaluation retains only the final 101
physical samples per environment, preserving terminal-inclusive tail metrics.

Checkpoints are atomically replaced only after finite model and optimizer
state checks. Every completed update can survive SIGTERM. Resume the **identical
command and output directory**: budgets, seeds, source hash and walltime deadline
are immutable. A process-owned output lock prevents duplicate supervisors.
Create `STOP` in this experiment's output directory to prevent new workers and
stop its current child; the supervisor checks once per second. The file is
never removed automatically, and must be removed explicitly before resumption.
Baseline runs' STOP files are neither read nor modified.
The supervisor also forwards interruption to its current child and allows 90 seconds
for a completed-update checkpoint before killing it. Thus maximum process
walltime may exceed the configured deadline by this grace period.
Abrupt mid-update failure resumes the previous complete update; fresh episodes
and zero actor/critic recurrent states start each worker boundary. Rollout
fragments are not replayed; checkpoint Adam, weights and normalization continue.
Matched arms have the same restart schedule at scheduled evaluation boundaries.
Unscheduled interruptions can perturb sampling; restart counts and update counts
are recorded, and strict bitwise replay is not claimed.

Saved checkpoints retain `infos.environment`, `infos.runner`, `infos.contract`,
progress and experiment metadata and are compatible with `evaluate.load_checkpoint`.
Ordinary `evaluate.py` intentionally evaluates ordinary resets; the experiment
driver is required to reproduce its recovery-start evaluation.

These are conservative **exploratory** intervals, not multiplicity-adjusted
confirmatory inference. With 128/512 trials, 2 pp noninferiority may remain
inconclusive; the driver must not waive uncertainty. A collision-free start
curriculum is not evidence of learned recovery from every contact state. If
insufficient, the report explicitly records a proper likelihood-consistent
correlated PPO distribution as **unexecuted/deferred**, not implemented by
ad-hoc post-action noise. There is no automatic deployment.
