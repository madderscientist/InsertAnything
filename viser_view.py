"""实时训练快照与最新策略回放可视化，不包含任何真机控制通信"""

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from mujoco._enums import mjtGeom, mjtObj
from mujoco._functions import mj_forward, mj_id2name, mju_mat2Quat, mju_rotVecQuat
from mujoco._structs import MjData, MjModel
import numpy as np
import trimesh
import viser

from insertion import InsertionCfg, build_model, geometry


def replay_index(elapsed: float, count: int) -> int:
    if count < 1:
        raise ValueError("replay requires at least one frame")
    return int(max(0., elapsed) / 0.02) % count


def replay_config(environment: dict, shape: str, zero_noise: bool = False) -> InsertionCfg:
    cfg = replace(InsertionCfg.from_checkpoint(environment),
                  shapes=(shape,), shape_env_counts=(), adaptive_sampling=False,
                  envs_per_shape=1, training=False)
    if zero_noise:
        cfg = replace(cfg, noise_curriculum=False, noise_curriculum_mode="steps",
                      position_noise=0., rotation_noise=0., force_noise=0., moment_noise=0.,
                      position_bias=0., tip_calibration_error=0., grasp_position_error=0.,
                      sensor_rotation_error=0., force_bias=0., moment_bias=0.)
    return cfg


class PolicyReplay:
    """Generate complete episodes separately, then display them at real-time speed."""

    def __init__(self, run_dir: Path, device: str, shape: str, zero_noise: bool = False):
        self.run_dir, self.device, self.shape = run_dir, device, shape
        self.zero_noise = zero_noise
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.episode = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="policy-replay", daemon=True)
        self.thread.start()

    def select(self, shape: str) -> None:
        with self.lock:
            self.shape = shape

    def latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Policy replay failed; inspect viewer log") from self.error
            return self.episode

    def _run(self) -> None:
        try:
            import torch
            from evaluate import load_checkpoint, REASONS
            from insertion import InsertionEnv

            torch.set_num_threads(1)
            env = None
            current_cfg = None
            while not self.stop.is_set():
                with self.lock:
                    shape = self.shape
                started = time.monotonic()
                metadata, actor = load_checkpoint(self.run_dir / "checkpoint.pt", self.device)
                cfg = replay_config(metadata["environment"], shape, self.zero_noise)
                if env is None or current_cfg != cfg:
                    env = InsertionEnv(cfg, self.device)
                    current_cfg = cfg
                else:
                    env.groups[0].reset(torch.arange(1, device=self.device))
                group = env.groups[0]
                obs = env.get_observations()
                frames = []
                reason = "timeout"
                with torch.inference_mode():
                    for _ in range(env.max_episode_length):
                        if self.stop.is_set() or shape != self.shape:
                            break
                        frames.append({
                            "qpos": group.sim.data.qpos[0].cpu().tolist(),
                            "grasp_offset": (group.offset[0]-group.nominal_offset[0]).cpu().tolist(),
                            "goal": group.goal[0].cpu().tolist(),
                            "wrench": group.wrench[0].cpu().tolist(),
                            "stiffness": group.k[0].cpu().tolist(),
                            "damping": group.d[0].cpu().tolist(),
                            "episode_step": int(group.length[0]),
                            "clearance": group.clearance,
                            "difficulty": group.difficulty,
                        })
                        obs, _, dones, info = env.step(actor(obs))
                        actor.reset(dones)
                        if bool(dones[0]):
                            reason = REASONS[int(info["terminal"]["reason"][0])]
                            break
                if self.stop.is_set():
                    return
                if shape != self.shape:
                    continue
                episode = {
                    "shape": shape, "frames": frames,
                    "policy_steps": metadata["progress"]["policy_steps"],
                    "generation_seconds": time.monotonic()-started,
                    "generated_at": time.time(), "reason": reason,
                    "zero_noise": self.zero_noise,
                }
                with self.lock:
                    self.episode = episode
                logging.info("Policy replay ready: shape=%s policy_steps=%d zero_noise=%s "
                             "result=%s frames=%d generation=%.2fs",
                             shape, episode["policy_steps"], self.zero_noise, reason,
                             len(frames), episode["generation_seconds"])
        except Exception as error:
            logging.exception("Policy replay worker failed; training is unaffected.")
            with self.lock:
                self.error = error

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            logging.error("Policy replay did not finish within its shutdown grace period.")


def geom_mesh(model: MjModel, index: int) -> trimesh.Trimesh:
    kind = model.geom_type[index]
    if kind == mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2*model.geom_size[index])
    if kind == mjtGeom.mjGEOM_CYLINDER:
        radius, half_height = model.geom_size[index, :2]
        return trimesh.creation.cylinder(radius=radius, height=2*half_height, sections=64)
    if kind == mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[index]
        vertex, face = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        return trimesh.Trimesh(
            vertices=model.mesh_vert[vertex:vertex+model.mesh_vertnum[mesh]].copy(),
            faces=model.mesh_face[face:face+model.mesh_facenum[mesh]].copy(),
            process=False,
        )
    raise ValueError(f"unsupported insertion geom type: {kind}")


def vector_arrow(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-12:
        return np.zeros((3, 2, 3), dtype=np.float32)
    direction = vector/length
    basis = np.eye(3)[np.argmin(np.abs(direction))]
    side = np.cross(direction, basis)
    side /= np.linalg.norm(side)
    head = min(length*0.3, 0.008)
    base = vector-head*direction
    return np.asarray([[np.zeros(3), vector],
                       [vector, base+head*0.5*side],
                       [vector, base-head*0.5*side]], dtype=np.float32)


def visible_wrench_vector(value: np.ndarray, reference: float) -> np.ndarray:
    return value*(0.03/reference)


class SnapshotScene:
    def __init__(self, server: viser.ViserServer | Any, shape: str, cfg: InsertionCfg):
        self.server = server
        self.model = build_model(geometry(shape, cfg.clearance, cfg.scale, cfg.circle_sides,
                                          cfg.plate_outer_radius_m))
        self.data = MjData(self.model)
        self.base_geom_pos = self.model.geom_pos.copy()
        self.tool_id = self.model.body("tool").id
        self.peg_ids = np.flatnonzero(self.model.geom_bodyid == self.tool_id)
        self.nominal_tip = np.array([0., 0., -0.05*cfg.scale])
        # 使用独立随机命名空间，防止切换形状时客户端残留上一几何的四元数姿态
        self.root = server.scene.add_frame(f"/plant/{shape}-{uuid4().hex}", show_axes=False)
        self.handles = []
        for index in range(self.model.ngeom):
            mesh = geom_mesh(self.model, index)
            rgba = self.model.geom_rgba[index]
            name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, index)
            self.handles.append(server.scene.add_mesh_simple(
                f"{self.root.name}/{name}", mesh.vertices, mesh.faces,
                color=(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)),
                opacity=0.45 if index not in self.peg_ids else 1.,
                side="double",
            ))
        self.tip = server.scene.add_frame(f"{self.root.name}/actual_tip", axes_length=0.006, axes_radius=0.0002)
        self.goal = server.scene.add_frame(f"{self.root.name}/true_goal", axes_length=0.006, axes_radius=0.0002)
        self.hole = server.scene.add_frame(f"{self.root.name}/hole", axes_length=0.015, axes_radius=0.0002)
        self.wrench_frame = server.scene.add_frame(
            f"{self.root.name}/wrench", show_axes=False)
        self.wrench_vectors = []
        for name, color in (("force", (255, 80, 60)), ("torque", (80, 140, 255))):
            self.wrench_vectors.append(server.scene.add_line_segments(
                f"{self.wrench_frame.name}/{name}", vector_arrow(np.zeros(3)),
                color, thickness=6., thickness_units="screen",
                position=(0., 0.008 if name == "force" else -0.008, 0.)))

    def update(self, plant: dict) -> None:
        wrench = np.asarray(plant["wrench"], dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("viewer wrench must contain six finite components")
        self.data.qpos[:] = plant["qpos"]
        self.model.geom_pos[:] = self.base_geom_pos
        self.model.geom_pos[self.peg_ids] += np.asarray(plant["grasp_offset"])
        mj_forward(self.model, self.data)
        with self.server.atomic():
            for index, handle in enumerate(self.handles):
                q = np.empty(4)
                mju_mat2Quat(q, self.data.geom_xmat[index])
                handle.position = self.data.geom_xpos[index]
                handle.wxyz = q
            rotated = np.empty(3)
            mju_rotVecQuat(rotated, self.nominal_tip+plant["grasp_offset"],
                           self.data.qpos[3:7])
            self.tip.position = self.data.qpos[:3]+rotated
            self.tip.wxyz = self.data.qpos[3:7]
            self.goal.position = np.asarray(plant["goal"])
            mju_rotVecQuat(rotated, self.nominal_tip, self.data.qpos[3:7])
            # 偏移显示避免不透明插头遮挡；只表达载荷方向，不代表实际作用点
            self.wrench_frame.position = self.data.qpos[:3]+rotated+np.array([0.03, 0., 0.02])
            self.wrench_frame.wxyz = self.data.qpos[3:7]
            for handle, value, reference in zip(
                    self.wrench_vectors, (wrench[:3], wrench[3:]), (5., 0.1), strict=True):
                handle.points = vector_arrow(visible_wrench_vector(value, reference))

    def remove(self) -> None:
        self.root.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--frames", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--mode", choices=("live", "policy"), default="live")
    parser.add_argument("--device", default="cuda:1", help="independent policy replay device")
    parser.add_argument("--zero-noise", action="store_true",
                        help="disable observation noise and fixed measurement/grasp errors in policy replay")
    args = parser.parse_args()
    if args.frames < 0 or not 1 <= args.port <= 65535:
        parser.error("frames must be nonnegative and port must be valid")
    if args.zero_noise and args.mode != "policy":
        parser.error("--zero-noise requires --mode policy; live training snapshots cannot be altered")
    logging.basicConfig(level=logging.INFO)
    metadata = json.loads((args.run_dir / "config.json").read_text())
    cfg = InsertionCfg.from_checkpoint(metadata["environment"])
    server = viser.ViserServer(host=args.host, port=args.port, label="Insertion training")
    selection = server.gui.add_dropdown("Geometry", options=cfg.shapes, initial_value=cfg.shapes[0])
    mode = server.gui.add_dropdown("Source", options=("Training snapshots", "Latest policy replay"),
                                  initial_value=("Latest policy replay" if args.mode == "policy"
                                                 else "Training snapshots"),
                                  disabled=args.zero_noise)
    frozen = server.gui.add_checkbox("Freeze view (training continues)", initial_value=False)
    status = server.gui.add_markdown("Waiting for the first training snapshot.")
    server.gui.add_markdown(
        "Training snapshots show actual environment 0 at its training speed. "
        "Latest policy replay is a separate deterministic rollout, buffered and "
        "played at 1x simulated time / 50 pose updates per second. The latest "
        "completed episode repeats while the next is generated. It is NOT live "
        "training and does not update the policy. Walls are translucent."
    )
    if args.zero_noise:
        server.gui.add_markdown(
            "**Zero-noise replay:** observation noise and fixed measurement/grasp errors disabled. "
            "Initial pose, target depth and dynamics still randomized; evaluation pose difficulty is 1."
        )

    @server.on_client_connect
    def configure_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0.10, 0.12, 0.08)
        client.camera.look_at = (0., 0., -0.005)
        client.camera.up_direction = (0., 0., 1.)

    scene = None
    selected_shape = None
    selected_clearance = None
    frames = 0
    replay = None
    displayed_episode = None
    replay_started = 0.
    last_status_time = 0.
    last_snapshot_key = None
    last_replay_key = None
    previous_tick = time.monotonic()
    try:
        while args.frames == 0 or frames < args.frames:
            started = time.monotonic()
            elapsed_tick = started-previous_tick
            previous_tick = started
            if frozen.value:
                replay_started += elapsed_tick
            path = args.run_dir / "scene.json"
            if path.exists():
                snapshot = json.loads(path.read_text())
                shape = selection.value
                plant = snapshot["plants"][shape]
                episode = None
                source = "Training snapshots (2 Hz; training simulation runs slower than real time)"
                if mode.value == "Latest policy replay":
                    if replay is None:
                        replay = PolicyReplay(args.run_dir, args.device, shape, args.zero_noise)
                    replay.select(shape)
                    episode = replay.latest()
                    if (frozen.value and displayed_episode is not None
                            and displayed_episode["shape"] == shape):
                        episode = displayed_episode
                    if episode is not None and episode["shape"] == shape:
                        if episode is not displayed_episode:
                            displayed_episode = episode
                            replay_started = time.monotonic()
                            last_replay_key = None
                        index = replay_index(time.monotonic()-replay_started, len(episode["frames"]))
                        plant = episode["frames"][index]
                        source = (f"Buffered policy replay on {args.device}, 1x / 50 Hz | "
                                  f"checkpoint policy steps {episode['policy_steps']} | "
                                  f"episode result: {episode['reason']} | "
                                  f"noise: {'ZERO' if episode['zero_noise'] else 'checkpoint full noise'} | "
                                  f"generation {episode['generation_seconds']:.1f}s | "
                                  "repeats until next episode is ready")
                    else:
                        episode = None
                        if args.zero_noise:
                            status.content = "**Buffering zero-noise policy episode; no training snapshot displayed.**"
                            frames += 1
                            time.sleep(max(0., 0.02-(time.monotonic()-started)))
                            continue
                        source = "Buffering first policy episode; showing training snapshots meanwhile"
                plant_clearance = float(plant.get("clearance", cfg.clearance))
                if (scene is None or shape != selected_shape
                        or plant_clearance != selected_clearance):
                    if scene is not None:
                        scene.remove()
                    scene = SnapshotScene(server, shape, replace(cfg, clearance=plant_clearance))
                    selected_shape = shape
                    selected_clearance = plant_clearance
                snapshot_key = (shape, snapshot["timestamp"])
                replay_key = (shape, id(episode), index) if episode is not None else None
                changed = ((replay_key != last_replay_key) if episode is not None
                           else (snapshot_key != last_snapshot_key))
                if not frozen.value and changed:
                    scene.update(plant)
                    last_snapshot_key, last_replay_key = snapshot_key, replay_key
                if time.monotonic()-last_status_time < 0.5:
                    frames += 1
                    time.sleep(max(0., 0.02-(time.monotonic()-started)))
                    continue
                last_status_time = time.monotonic()
                heartbeat_path = args.run_dir / "heartbeat.json"
                heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.exists() else {}
                age = max(0., time.time()-snapshot["timestamp"])
                status.content = (
                    f"**{source}**  \n"
                    f"Training snapshot: {'STALE' if age > 10 else 'Live'} | age {age:.1f} s  \n"
                    f"PPO iteration: {heartbeat.get('iteration', 'initial rollout')} | "
                    f"policy steps: {snapshot['policy_steps']}  \n"
                    f"Mean completed-episode reward: {heartbeat.get('mean_episode_reward')}  \n"
                    f"Geometry: **{shape}** | clearance: {plant_clearance*1000:.2f} mm | "
                    f"episode step: {plant['episode_step']} | "
                    f"difficulty: {plant['difficulty']:.3f}  \n"
                    f"Target depth: {-plant['goal'][2]*1000:.2f} mm  \n"
                    f"K [N/m, Nm/rad]: {np.round(plant['stiffness'], 3).tolist()}  \n"
                    f"D [Ns/m, Nms/rad]: {np.round(plant['damping'], 3).tolist()}  \n"
                    f"Fixed grasp offset [mm]: "
                    f"{np.round(np.asarray(plant['grasp_offset'])*1000, 3).tolist()}"
                )
            frames += 1
            time.sleep(max(0., 0.02-(time.monotonic()-started)))
    finally:
        if replay is not None:
            replay.close()
        server.stop()


if __name__ == "__main__":
    main()
