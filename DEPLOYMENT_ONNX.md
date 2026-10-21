# ONNX deployment contract for the real-robot integration agent

## 0. Scope and readiness

This package is an inference interface, NOT a robot controller or a validated
real-robot safety system. The policy has only been trained/evaluated in simulation.
The exported checkpoint has not completed the noise curriculum. Check the exact
iteration, curriculum level, environment parameters and SHA256 in `contract.json`.
Do not interpret passing ONNX equivalence tests as successful sim-to-real validation.

The requested export is the backup immediately BEFORE the std reset to 0.15:
iteration 13574, curriculum level 7/15 (0.1 mm clearance, 20% noise).
The Gaussian std was approximately 0.048, but ONNX returns the deterministic mean:
no action sampling, no std input, no artificial exploration on the robot.
This file describes the interface for any export; `contract.json` is authoritative
for the particular file. Robot dimensions, sensor calibration, sign convention,
gravity compensation, controller bandwidth and contact compliance must be verified
by the integration team before enabling motion.

## 1. Files and dependencies

- `actor.onnx`: observation normalization + two-layer GRU + ELU MLP, deterministic.
- `contract.json`: input/action semantics, limits, source and training configuration.
- `validation.json`: CPU ONNX/PyTorch recurrent sequence equivalence and ONNX-only latency.
- `test_vectors.npz`: ordered test observations, PyTorch reference mean actions,
  reference hidden outputs and boolean reset flags. At a true flag zero the input
  hidden state; otherwise feed the preceding ONNX hidden output. Compare actions
  and states with rtol=1e-4, atol=1e-5. This tests inference, not the sensor adapter.

Inference needs Python, NumPy and ONNX Runtime, not MuJoCo, Torch or mjlab:

```bash
python -m pip install numpy onnxruntime
```

Repository export additionally needs `onnx`, `onnxscript`, `onnxruntime` and Torch.
All observations/states are finite `float32`, C-contiguous arrays. Fixed batch one:

| Name | Shape | Meaning |
|---|---|---|
| `obs` input | `[1,13]` | unnormalized SI-unit observation |
| `h_in` input | `[2,1,128]` | previous GRU state |
| `actions` output | `[1,6]` | deterministic, NOT yet clipped action |
| `h_out` output | `[2,1,128]` | state for next policy tick |

Normalization statistics are embedded: DO NOT normalize/clamp observations again.
The network does NOT perform pose transforms, filtering, action clipping,
workspace checks, success detection or safety-stop handling.

## 2. Coordinates, units and pose

All frames are right-handed; positions in meters, rotations in radians, force in N,
moment in N*m. Quaternions are unit Hamilton quaternions in **w,x,y,z** order.
`R_AB` maps coordinates from frame B into A; `q_AB` is the corresponding quaternion.

- H: fixed hole frame, origin at the hole entrance plane, +Z outward/upward.
  Insertion proceeds along -Z. H X/Y encode the hole's chosen yaw reference.
- T: nominal tool/peg-tip frame. Orientation follows the tool body; origin is the
  calibrated nominal peg tip, NOT the robot flange or sensor origin.
- S: physical force/torque sensor frame at the sensor reference point.
- B: robot base/world frame, used only by the external robot adapter.

Convert measured TCP/nominal-tip pose into H:

```text
p_HT = R_HB * (p_BT - p_BH)
q_HT = inverse(q_BH) * q_BT
```

If robot feedback is the flange pose, first apply the calibrated flange-to-tip
rigid transform. Do NOT add a simulation "grasp offset" to a real measured pose.
Calibrate the actual tool; residual errors are not something to inject manually.

The goal pose is in H. A nominal example is `p_goal=[0,0,-0.015]`,
`q_goal=[1,0,0,0]`, meaning a 15 mm target depth and aligned tool axes.
Use the real task's verified target depth, not this example blindly.
Training sampled 5--22.5 mm depth. No shape ID or symmetry folding goes into obs.
Do not assume that a 180-degree-flipped tool frame matches the trained convention.

Observation layout (zero-based):

```text
obs[0,0:3]  = p_goal_H - p_tip_H
obs[0,3:7]  = canonical(inverse(q_HT) * q_goal_H)
obs[0,7:10] = filtered, gravity-compensated force at tip, expressed in T
obs[0,10:13]= filtered, gravity-compensated moment about nominal tip, expressed in T
```

Position error is in **fixed H**, NOT in the moving tool frame.
Relative orientation uses **inverse(current) * goal**, not the opposite order.
Canonical quaternion: normalize, then choose sign so the first component whose
absolute value exceeds 1e-8 in w,x,y,z order is positive. Usually this means w>=0.
Reject invalid/zero quaternions rather than silently replacing them by identity.

## 3. Force/torque processing: implement this outside ONNX

### 3.1 Calibrate and establish signs

Start from the complete six-axis wrench `[Fx,Fy,Fz,Mx,My,Mz]` at S, in S axes.
Subtract calibrated sensor electronic zero offsets and compensate the static
gravity wrench of ALL hardware downstream of S at the current orientation.
Do not tare away an existing contact force. Do not double-compensate if the sensor
driver already compensates gravity.

Simulation disables gravity and models contact minus downstream inertial load.
Its wrench is not an ideal contact-only signal: downstream inertia remains.
The nominal downstream simulated mass is 0.2--0.4 kg in training (not the entire
robot reflected inertia). Do not subtract all dynamic/inertial load by default.
Real mass/COM, acceleration and sign conventions need independent verification.

Vendor sign conventions differ (load-on-sensor versus sensor-on-tool). Match the
simulation's external load-on-tool convention using known static applied forces
and moments in each axis. Do not guess or blindly negate all six channels.
Never directly feed wrist-frame moments without shifting their reference point.

### 3.2 Low-pass at the sensor/wrist reference point

Training low-passes the compensated six-axis sensor wrench in tool-following
sensor axes, BEFORE shifting to the tip:

```text
tau = 0.015 seconds
alpha = 1 - exp(-actual_sensor_dt / tau)
filtered_S = filtered_S + alpha * (compensated_wrench_S - filtered_S)
```

Training updates at 1000 Hz (dt=0.001, alpha approximately 0.064493);
the actor reads the latest result at 50 Hz.
Use timestamped high-rate samples if available. Filtering only at 50 Hz is not
numerically equivalent when there are intervening samples, motion or impacts.
Do not low-pass twice in both driver and application without characterizing it.
Simulation resets the filter to zero each episode. A real filter reset must only
occur in a confirmed safe unloaded state; it must not mask a live contact.

### 3.3 Rotate and shift to nominal tip

Let `R_TS` map S axes to T axes, and let `r_ST_T` be the vector FROM the sensor
reference point TO the nominal tip, expressed in T:

```text
F_T = R_TS * filtered_F_S
M_T = R_TS * filtered_M_S - cross(r_ST_T, F_T)
wrench_tip_T = [F_T, M_T]
```

Use real calibrated extrinsics and lever arm. Training nominally uses aligned
sensor/tool axes and sensor-to-tip displacement `[0,0,-0.08]` m, but 80 mm is NOT
a universal robot constant. Test the sign with a known lateral force applied at
the tip: after the shift, moment about that same tip should be near zero.

No added white noise, artificial zero bias, mounting error or curriculum scaling
is needed on the robot. The simulation's per-episode force/moment biases affect
actor observations only; they are not a preprocessing correction to apply.
Do not assume robustness to the full configured training noise: this checkpoint
was only at 20% noise.

## 4. Actions and external impedance controller

Run at **50 Hz (20 ms)** with synchronized latest pose and processed wrench.
Always reject nonfinite output; then clamp EACH component of `actions` to [-1,1].
With the exported scale (verify `contract.json`):

```text
dp_H = clip(actions[0,0:3], -1,1) * 0.001 m
dr_H = clip(actions[0,3:6], -1,1) * (0.5*pi/180) rad
p_target_H = CURRENT measured nominal-tip position_H + dp_H
q_target_HT = canonical(ExpQuaternion(dr_H) * CURRENT q_HT)
```

Rotation is a rotation vector expressed in H, applied by LEFT multiplication.
0.5 degrees is a PER-COMPONENT limit, not a limit on the total rotation angle.
This is NOT a velocity command, absolute pose, force command or flange displacement.
Do not multiply by 0.02 again, or integrate onto last cycle's commanded target.
Pure rotation keeps the nominal tip position fixed; convert that target to the
robot flange using the real rigid transform, rather than rotating around the flange.

Transform target back to B before passing to a verified Cartesian impedance/
compliance controller. Training uses impedance at the tip and 1 kHz physics/control;
50 Hz policy commands do not replace a high-rate compliant low-level controller.
Sending these targets to a stiff position controller is NOT equivalent.
Training randomizes diagonal stiffness (300--2500 N/m, 4--12 N*m/rad), damping and
stiffness axes. Those are simulation ranges, not validated hardware gains.
Do not invent a low-level gain schedule or execute automatic movements from this file.

## 5. Recurrent state and a minimal inference-only example

```python
import numpy as np
import onnxruntime as ort

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(
    "actor.onnx", sess_options=options, providers=["CPUExecutionProvider"])
h = np.zeros((2, 1, 128), dtype=np.float32)

def reset_policy():
    global h
    h.fill(0)

def infer_observation(obs):
    global h
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    if obs.shape != (1, 13) or not np.isfinite(obs).all():
        raise ValueError("Invalid policy observation; stop through independent safety system")
    actions, next_h = session.run(None, {"obs": obs, "h_in": h})
    if not np.isfinite(actions).all() or not np.isfinite(next_h).all():
        raise RuntimeError("Invalid policy output/state")
    h = next_h
    return np.clip(actions, -1, 1)
```

This example performs NO robot I/O and intentionally does NOT build a control loop.
Keep one state per tool/task; carry `h_out` to the next invocation. Reset to zeros
at each new insertion attempt, after repositioning, emergency/safety stops,
communication interruption, or a model change. Do NOT reset every tick.
Do not advance the GRU twice for the same sensor tick or run one policy state
across unrelated robots. On late/stale/missing data, stop/hold according to the
independent controller's safety procedure; do not silently send zeros and continue.

## 6. Independent safety, success and commissioning

ONNX has NO safety checks. The existing TorchScript wrapper's checks are NOT
embedded in this file. The hardware integration must provide independently:

- High-rate RAW compensated force/moment monitoring, NOT just the 15 ms filter
  used by the actor. Simulation terminates if the tip force norm exceeds 30 N or
  tip moment norm exceeds 2 N*m at ANY 1 ms substep; these are simulation limits,
  not a certification that they are safe for this robot, fixture, peg or person.
  Use hardware-specific, independently validated lower limits where required.
- Workspace, tilt, joint, velocity, acceleration, singularity, self-collision,
  fixture collision and communication watchdog constraints.
- Independent emergency stop and a supervised, unloaded dry run before contact.
  No autonomous recovery/retry based solely on a model output.
- Proper gravity/sign/extrinsic calibration and a force-at-tip lever-arm test.
- Verified compliant controller and conservative hardware motion/force envelopes.

Simulation workspace uses horizontal radial distance <=25 mm, tip Z in
[-25 mm,30 mm], tilt <=35 degrees, and 8 s episode timeout. Treat these as training
semantics, not as hardware approval. Match geometries and scale; simulation has
no robot kinematics, backlash, cable forces, sensor saturation or controller faults.

The legacy iteration-13574 checkpoint uses the strict success rule below.
New training defaults to target depth reached for 100 ms without additional
XY/tilt/yaw success gates; safety limits and shaping costs remain unchanged.
See `success_alignment_required` in the source environment (missing means legacy).
Legacy training success is NOT "looks inserted": target depth within 0.2 mm, XY error
<0.4*clearance, tilt <0.4*clearance/peg_length, symmetry-aware yaw
<0.4*clearance/radius for 5 consecutive 20 ms ticks, with no safety failure.
At 0.1 mm clearance, XY tolerance is 40 micrometers. Real success needs independent
task-specific sensing/validation; do not claim the hardware can observe training
ground-truth quantities at this precision. Symmetry handling is for success/reward,
not for folding ONNX input orientation.

Initial-state correction in this run keeps the entire rotated peg at least 1 mm
above the hole plane, not just its nominal tip center. Do not start a tilted peg
with an edge already intersecting the fixture. Start from a collision-checked pose.

The package has been validated for recurrent numerical equivalence on the export
machine CPU. Actual target-machine end-to-end latency, clocks, sensor pipeline,
robot bandwidth and real contact behavior still require validation. Never interpret
an inference timing report as evidence of physical safety.
