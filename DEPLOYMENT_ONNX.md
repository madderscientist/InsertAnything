# ONNX deployment guide

This is a simulation-trained inference model, not a robot controller or validated safety system. `contract.json` is authoritative for the exported iteration, training configuration, tensor interface and source SHA256. Numerical equivalence does not establish real-robot readiness.

## 1. Files and interface

- `actor.onnx`: deterministic mean policy with embedded observation normalization.
- `contract.json`: interface, coordinate conventions and source configuration.
- Optional `validation.json` and `test_vectors.npz`: recurrent equivalence results and reference observations/actions/hidden states. Test-vector reset flags indicate when to zero the hidden input; compare with `rtol=1e-4, atol=1e-5`.

Inference requires only NumPy and ONNX Runtime:

```bash
python -m pip install numpy onnxruntime
```

| Tensor | Shape | Meaning |
|---|---|---|
| `obs` | `[1,13]` | Unnormalized observation |
| `h_in` | `[2,1,128]` | Previous GRU state |
| `actions` | `[1,6]` | Unbounded deterministic mean |
| `h_out` | `[2,1,128]` | Next GRU state |

All tensors are finite, contiguous `float32`, with fixed batch size one. Do not normalize observations again or sample exploration noise. Exporting with `--onnx --prune-std` removes the unused standard-deviation head without changing mean actions.

## 2. Frames and observations

Use right-handed frames, SI units (`m`, `s`, `rad`, `N`, `N*m`) and Hamilton unit quaternions in **w,x,y,z** order. `R_AB` maps B-frame vectors into A; `q_AB` represents the same rotation.

- **H**: fixed hole frame at the entrance; +Z points outward, insertion proceeds along -Z.
- **T**: nominal tool-tip frame, following tool orientation; not the flange or sensor origin.
- **S**: force/torque sensor frame at its measurement reference point.
- **B**: robot base/world frame.

Transform the calibrated measured tip pose from B into H. Flange feedback first requires the real flange-to-tip transform; do not inject simulation grasp offsets.

```text
p_HT = R_HB * (p_BT - p_BH)
q_HT = inverse(q_BH) * q_BT

obs[0,0:3]   = p_goal_H - p_HT
obs[0,3:7]   = canonical(inverse(q_HT) * q_goal_H)
obs[0,7:10]  = filtered force at nominal tip, in T
obs[0,10:13] = filtered moment about nominal tip, in T
```

The goal is expressed in H; for example, `[0,0,-0.015]` is 15 mm insertion depth, not a universal target. Read the source configuration for training depths. Position error stays in fixed H; do not rotate it into T, add shape IDs or fold orientation by shape symmetry.

`canonical(q)` normalizes q and selects the sign making its first component with absolute value above `1e-8` positive, in w,x,y,z order. Reject zero/invalid quaternions. Training may delay only position observations by 0/20 ms; this is outside ONNX. Measure actual sensor latency rather than adding artificial delays or noise.

## 3. Force/torque preprocessing

Use all six sensor channels. Subtract electronic zero bias and compensate gravity for all downstream hardware, without double compensation or taring away contact. Verify signs using known applied forces/moments: the model expects external load on the tool. Simulation retains downstream inertial load; do not assume an ideal contact-only signal or remove dynamic loads without validation.

Low-pass the compensated wrench in sensor axes **before** shifting its reference point:

```text
tau = 0.015 seconds
alpha = 1 - exp(-actual_sensor_dt / tau)
filtered_S += alpha * (compensated_wrench_S - filtered_S)
```

Training updates at 1 kHz and reads at 50 Hz; filtering only at 50 Hz is not equivalent. Use timestamps and avoid duplicate filtering. Reset a physical filter only in a verified safe unloaded state.

Let `R_TS` map S into T, and `r_ST_T` point from the sensor origin to the nominal tip, expressed in T:

```text
F_T = R_TS * filtered_F_S
M_T = R_TS * filtered_M_S - cross(r_ST_T, F_T)
```

Use measured extrinsics and lever arm, not the simulation's nominal 80 mm offset. A known lateral force applied at the tip should yield approximately zero moment about that tip after transformation. Simulation bias, mounting error, white noise and hysteresis model uncertainty; do not deliberately inject them into robot measurements.

## 4. Actions and controller

At **50 Hz**, use synchronized pose and processed wrench. Scale each action component using `translation_limit_m` and `rotation_component_limit_rad` from the contract; these legacy names denote scales, not bounds. Typical values are 0.001 m and 0.5 degrees:

```text
dp_H = actions[0,0:3] * translation_limit_m
dr_H = actions[0,3:6] * rotation_component_limit_rad
p_target_H = CURRENT measured nominal-tip position_H + dp_H
q_target_HT = canonical(ExpQuaternion(dr_H) * CURRENT q_HT)
```

Rotation is an H-frame rotation vector applied by **left multiplication**, about the nominal tip. Actions are increments, not velocities: do not multiply by 0.02 or integrate onto the previous commanded target. ONNX performs no clipping; clipping externally changes the policy and requires validation. Independent hardware safety envelopes remain mandatory.

Transform targets back to B and convert tip targets to flange targets using calibrated geometry. Training uses high-rate compliant impedance control, not a stiff position servo; its randomized gains are not approved hardware gains. The 50 Hz policy does not replace the low-level controller.

## 5. Minimal inference

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("actor.onnx", providers=["CPUExecutionProvider"])
h = np.zeros((2, 1, 128), dtype=np.float32)

def reset_policy():
    h.fill(0)

def infer(obs):
    global h
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    if obs.shape != (1, 13) or not np.isfinite(obs).all():
        raise ValueError("Invalid observation")
    actions, next_h = session.run(["actions", "h_out"], {"obs": obs, "h_in": h})
    if not np.isfinite(actions).all() or not np.isfinite(next_h).all():
        raise RuntimeError("Invalid policy output/state")
    h = next_h
    return actions
```

This example has no robot I/O. Carry `h_out` into the next tick; reset at each new attempt, repositioning, stop, communication interruption or model change, not every tick. Keep separate state per task and advance it only once per sensor tick. Invalid/stale data or inference errors must invoke the independent controller's stop/hold procedure, not a silent zero-action fallback.

## 6. Safety and validation

ONNX contains no pose transforms, filtering, safety checks or success detection. Hardware integration must independently provide:

- High-rate raw compensated load monitoring, workspace/tilt/joint/motion/collision limits and a communication watchdog.
- Emergency stop, calibrated sensor signs/extrinsics, a verified compliant controller and supervised unloaded commissioning.
- Task-specific success sensing and validated contact tests; no automatic recovery based solely on model output.

Normal simulation terminates above 30 N force norm or 2 N*m tip-moment norm at any 1 ms substep. These are training thresholds, not certified hardware limits; choose independently validated hardware-specific limits. Success and workspace rules vary by checkpoint: consult the source environment's `success_*`, `workspace_xy_radius_m` and episode settings rather than inferring them from appearance.

Start from collision-checked poses. Validate target-machine end-to-end timing, sensor processing, controller bandwidth and contact behavior before motion; inference latency and simulation success do not establish physical safety.
