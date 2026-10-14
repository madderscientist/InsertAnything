"""冻结策略的本地窗口交互可视化与无头模式快照渲染"""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import mujoco
from mujoco._functions import mj_forward
from mujoco._structs import MjData, MjvCamera
import torch

from evaluate import load_checkpoint
from insertion import ALL_SHAPES, InsertionCfg, InsertionEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--shape", choices=ALL_SHAPES, default="hexagon")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--target-depth", type=float, default=None, help="fixed depth in metres")
    parser.add_argument("--snapshot", type=Path, help="save a terminal RGB image instead of opening a viewer")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    metadata, actor = load_checkpoint(args.checkpoint, args.device)
    base = InsertionCfg.from_checkpoint(metadata["environment"])
    cfg = replace(base, shapes=(args.shape,),
                  fixed_target_depth_m=(base.fixed_target_depth_m if args.target_depth is None
                                        else args.target_depth),
                  envs_per_shape=1, training=False)
    env = InsertionEnv(cfg, args.device)
    obs = env.get_observations()
    group = env.groups[0]

    def advance() -> None:
        nonlocal obs
        with torch.inference_mode():
            obs, _, dones, info = env.step(actor(obs))
            actor.reset(dones)
        if dones.any():
            print({k: v.tolist() for k, v in info["terminal"].items()})

    def sync_cpu(data: MjData) -> None:
        data.qpos[:] = group.sim.data.qpos[0].cpu().numpy()
        data.qvel[:] = group.sim.data.qvel[0].cpu().numpy()
        group.model.geom_pos[:] = group.sim.model.geom_pos[0].cpu().numpy()
        mj_forward(group.model, data)

    if args.headless or args.snapshot:
        for _ in range(args.steps):
            advance()
        if args.snapshot:
            from PIL import Image
            data = MjData(group.model)
            sync_cpu(data)
            camera = MjvCamera()
            camera.lookat[:] = (0, 0, -0.005)
            camera.distance = 0.13
            camera.azimuth, camera.elevation = 140, -35
            with mujoco.Renderer(group.model, height=480, width=640) as renderer:
                renderer.update_scene(data, camera=camera)
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render()).save(args.snapshot)
        return
    from mujoco import viewer as mj_viewer
    data = MjData(group.model)
    with mj_viewer.launch_passive(group.model, data) as viewer:
        viewer.cam.lookat[:] = (0, 0, 0)
        viewer.cam.distance = 0.15
        for _ in range(args.steps):
            if not viewer.is_running():
                break
            start = time.monotonic()
            advance()
            sync_cpu(data)
            viewer.sync()
            time.sleep(max(0., 0.02 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
