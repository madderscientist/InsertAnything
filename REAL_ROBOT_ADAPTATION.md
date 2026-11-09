# 真机部署问题诊断与训练环境改进说明（PiPER-X 六轴机械臂）

本文记录在 **PiPER-X 桌面六轴串联机械臂（`arm_4`）+ PaXini PX6D-75A 六维腕部力/力矩传感器** 上部署 `actor.onnx` 时暴露的 Sim-to-Real 差异、物理与数学根因推导，以及对本仓库训练环境作出的相应改动与后续算力机训练指南。

---

## 1. 真机硬件参数与控制链路概况

| 模块 | 真机配置 | 原仿真默认假设 |
|---|---|---|
| **运动学结构** | 6 自由度串联关节臂（J1–J3 主臂平移定位，J4–J6 腕部姿态旋转） | 6 自由度解耦浮动刚体（`freejoint`） |
| **传感器至尖端力臂 $\|r_{ST}\|$** | **`205 mm`**（`r_ST = [0, 0, -0.205] m`，含法兰转接件与长插头） | 名义 `80 mm`（刚体速度/控制直接作用于尖端，几何力臂在无偏置时精确对消） |
| **底层关节阻抗增益** | `kp = [20, 18, 15, 6, 5, 4] N·m/rad`，临界阻尼 `kd`，零误差自适应刚度 `s_a = 2.0` | 尖端笛卡尔三轴正交弹簧（平移 `[300, 2500] N/m`，旋转 `[4, 12] N·m/rad`） |
| **尖端等效平移刚度** | 腕部柔顺经 `20.5 cm` 力臂放大后，单轴等效刚度约 **`120–350 N/m`** | 下界 `300 N/m`（初期课程偏向 `1500 N/m`） |
| **接触滑动摩擦系数 $\mu$** | 3D 打印件层纹与孔口棱边在受压接触下等效摩擦达 **`0.6–0.85`** | 均匀采样于 `[0.25, 0.6]` |
| **人工提拔初态范围** | 孔内记目标点后人工拔出，起始高度 **`10–20 mm`**、倾角 **`5°–8°`** | `initial_z_max = 10 mm`，`initial_tilt = 3°` |

---

## 2. 真机实测现象与物理根因分析

### 2.1 现象一：长力臂放大力传感器残差，引发虚假尖端力矩与反向逃逸

- **实测现象**：在真机运行日志（如 `step 5931`，尖端位于 `p_ht = [-17.7, -11.0, 30.9] mm`）中，插头尚未进入孔内，腕部传感器因姿态变化和线缆牵拉存在仅 `Fx = -0.41 N` 的微小横向力残差；经 `r_ST = [0, 0, -0.205] m` 搬移至尖端后：
  $$M_T = M_S - r_{ST} \times F_T \implies M_y = -(-0.205)\times(-0.41) \approx -0.084\text{ N}\cdot\text{m}\text{（实测 }-0.087\text{ N}\cdot\text{m）}$$
  此时 `actor.onnx` 将该 `-0.087 N·m` 误判为孔口边缘碰撞力矩，X 轴动作瞬间从 `+1.88 mm`（朝向孔中心）反转为 `-0.86 mm`（背离孔中心），导致插头卡在 `-18 mm` 处无法接近孔口。
- **为什么不需要在仿真中随机化刚体几何长度 `8 cm`？**
  在 `InsertAnything` 的 `PlantGroup` 中，无论是策略观测（`p_goal - p_tip`）、动作增量作用点（`action_target` 以 `p_tip` 为旋转和平移中心），还是底层阻抗控制（`physics_control` 直接在 `p_tip` 施加弹簧力并通过 `r_ct × force` 抵消质心力矩），**全部直接定义在最末端尖端 `T`**。在 `update_sensor()` 中，传感器位置 `sensor_offset` 在 `COM -> S` 与 `S -> T` 两次力旋量变换之间代数抵消。
  因此，刚体物理长度与末端移动指令完全无关；真机 `20.5 cm` 长力臂对策略观测的唯一实质影响，是将腕部传感器的力噪声与力零偏通过叉乘 $r_{ST} \times \Delta F_S$ 放大为**尖端力矩噪声（`moment_noise`）与尖端力矩偏置（`moment_bias`）**。
- **原配置缺陷**：原 `InsertionCfg.moment_noise` 仅为 `0.001 N·m`（`1 mN·m`），甚至低于 `8 cm` 力臂下 `0.2 N` 力白噪声对应的搬移力矩（$0.08 \times 0.2 = 0.016\text{ N}\cdot\text{m}$），更远低于 `20.5 cm` 力臂下的真机尖端力矩噪声水平（`0.015–0.02 N·m`）；且部分早期检查点使用 `--moment-bias 0.01` 训练，导致 GRU 对 `0.05–0.10 N·m` 量级的尖端力矩极为敏感。

### 2.2 现象二：入孔前零倾角惩罚（`preinsertion_tilt_weight = 0.0`）诱发大幅摆头与腕-臂运动耦合

- **实测现象**：插头在孔板上方搜索时，策略频繁输出高达 `3–16`（对应单步 `1.5°–8°`）的旋转动作 `actions[3:6]`，导致插头在孔口上方剧烈歪斜。
- **数学根因**：
  在 `insertion/env.py` 的 `step_outcome` 与 `completion_cost_terms` 中，纵向剩余距离使用插头碰撞几何的**最低点高度**：
  $$z_{\min} = z_{\mathrm{tip}} + z_{\mathrm{offset}}(q), \qquad z_{\mathrm{offset}}(\theta) \approx -R\sin\theta - L_{\mathrm{peg}}(1-\cos\theta) < 0$$
  当原配置 `preinsertion_tilt_weight = 0.0` 时，插头在孔板上方（$z_{\min} \ge 0$）倾斜**完全不产生任何姿态惩罚**，却能通过增大倾角 $\theta$ 直接压低几何最低点 $z_{\min}$（例如 $R=10\text{ mm}$ 插头倾斜 $5^\circ$ 即可白赚 $0.87\text{ mm}$ 的“伪下探深度”并显著降低 `position_penalty`）。
  在仿真中，浮动刚体绕尖端纯旋转毫无代价；但在真机 6 轴串联臂上，绕 `20.5 cm` 尖端旋转 $\Delta\theta = 5^\circ$ 需要主臂 J1–J3 反向平移 $205\text{ mm} \times \sin 5^\circ \approx 17.9\text{ mm}$ 来维持尖端位置不变。由于主臂关节与腕部关节带宽、摩擦不同步，大幅旋转指令直接引发厘米级末端寄生横移与姿态发散。

### 2.3 现象三：等效末端笛卡尔刚度偏软且接触摩擦偏大，下压搜索时发生粘滞卡死

- **实测现象**：当插头以 `Fz ≈ 5 N` 压在孔板上并尝试横向搜索时，真机末端出现明显弹性滞后，单步 `1 mm` 动作无法像仿真中那样干脆地推动插头。
- **物理根因**：PiPER-X 腕部关节 J4/J5/J6 标称刚度为 `kp = [6, 5, 4] N·m/rad`。经 $L = 0.205\text{ m}$ 力臂映射后，腕部柔顺决定的尖端单轴等效平移刚度为：
  $$K_{\mathrm{tip}} \approx \frac{k_{p,5}}{L^2} = \frac{5.0}{0.205^2} \approx 119\text{ N/m}$$
  即便计入底层驱动的零误差自适应刚度（`s_a = 2.0`），大误差段刚度仍回落至 `120–250 N/m` 区间，旋转刚度在 `2–5 N·m/rad` 区间。而原仿真 `dynamics="train"` 的采样下界为平移 `300 N/m`、旋转 `4 N·m/rad`（甚至未能覆盖评估模式 `soft` 的 `1.5–3.0 N·m/rad`），导致策略未见过真机级别的柔顺响应。
  同时，真机关节静摩擦虽已由底层 `piper_x_driver` 的库仑+谐波静摩擦前馈补偿，但 3D 打印插头与孔板表面层纹、孔口倒角边缘在受压下的等效接触摩擦系数可达 `0.6–0.85`（原仿真上限仅 `0.6`）。由于入孔前存在 `5 N` 力成本死区，若策略在 `150 N/m` 低刚度与 `0.8` 高摩擦下仍以 `5 N` 下压力死压孔板，最大静摩擦力（$\mu F_z \approx 4\text{ N}$）将迫使末端累积超过 `25 mm` 的目标超前量才能滑动。提高仿真接触摩擦上限至 `0.85` 并配合 `150 N/m` 刚度下界，能直接在物理仿真中复现“重压即卡死”现象，迫使策略学会在孔板横向搜索时主动减轻下压力。

### 2.4 现象四：人工提拔起始位姿超出训练初态分布（OOD）

- **实测现象**：真机采用“孔内记录目标位姿 $\to$ 人工将插头拔出孔外 $\to$ 启动策略”的操作流程。人工拔出后的起始高度通常在孔口上方 `10–20 mm`，伴随 `5°–8°` 初始倾角。
- **根因**：原训练配置 `initial_z_max = 0.01 m`（`10 mm`）、`initial_tilt = 3°`。真机一起步即处于训练分布之外，导致双层 GRU 隐状态在接近阶段发生偏移。

---

## 3. 本仓库代码与默认参数改动汇总

针对上述四项根因，对 [insertion/config.py](./insertion/config.py)、[insertion/env.py](./insertion/env.py) 与 [train.py](./train.py) 进行了如下修改：

| 参数 / 逻辑位置 | 原默认值 | 新默认值 | 改动目的 |
|---|---:|---:|---|
| `InsertionCfg.moment_noise` (`--moment-noise`) | `0.001 N·m` | **`0.015 N·m`** | 匹配长力臂（`20.5 cm`）下力传感器噪声搬移至尖端后的真实力矩噪声量级（`15 mN·m`） |
| `InsertionCfg.moment_bias` (`--moment-bias`) | `0.1 N·m` | **`0.1 N·m`** | 保持 `0.1 N·m` 默认上限（对应 `0.5 N × 0.2 m` 力臂），训练时切勿调低至 `0.01` |
| `InsertionCfg.preinsertion_tilt_weight` (`--preinsertion-tilt-weight`) | `0.0` | **`0.5`** | 消除孔板上方“靠歪斜插头压低最低点 $z_{\min}$ 骗取深度分”的投机行为，入孔后前 `4 mm` 仍平滑过渡至 `1.0` |
| `InsertionCfg.initial_z_max` (`--initial-z-max`) | `0.01 m` | **`0.02 m`** | 初始尖端高度采样上限从 `10 mm` 扩展至 `20 mm`，覆盖人工拔出孔外的真实起手高度 |
| `InsertionCfg.initial_tilt` (`--initial-tilt-deg`) | `3°` | **`8°`** | 初始旋转向量 X/Y 半幅从 `3°` 扩展至 `8°`，覆盖真机起手姿态偏差 |
| `PlantGroup.reset` (`dynamics="train"` 刚度范围) | `[300, 2500] N/m`<br>`[4, 12] N·m/rad` | **`[150, 2500] N/m`**<br>**`[2, 12] N·m/rad`** | 下探覆盖桌面串联机械臂在长工具下的真实等效末端平移/旋转刚度，并与 `soft` 评估模式对齐 |
| `PlantGroup.reset` (`geom_friction` 滑动摩擦范围) | `[0.25, 0.6]` | **`[0.25, 0.85]`** | 覆盖 3D 打印件层纹与孔口边缘高摩擦接触，配合低刚度迫使策略在横向搜索时控制下压力 |
| `train.restore_environment_progress` | 未监控 `moment_noise` 与 `preinsertion_tilt_weight` | **加入 `changed_fields`** | 从旧检查点迁移至新参数续训时，自动保留当前课程等级并清空旧分布下的成功率统计窗口 |

---

## 4. 算力机训练与导出指南

### 4.1 方案 A：从头启动单卡/双卡持续训练（推荐）

在具备 GPU 与 `mj` 环境的算力机上，直接拉取最新代码后执行（代码默认值已包含全部新参数）：

```bash
# 单卡持续训练（例如 RTX 4090 24GB）
CUDA_VISIBLE_DEVICES=0 micromamba run -r ~/micromamba -n mj python -u train.py \
  --output runs/real_robot_v1 --device cuda:0 \
  --envs-per-shape 512 --steps 256 \
  --success-noise-curriculum --adaptive-sampling \
  --physics-cuda-graph --observation-cuda-graph

# 双卡分布式持续训练
micromamba run -r ~/micromamba -n mj python -u train.py \
  --output runs/real_robot_v1 --device cuda:0 \
  --simulation-devices cuda:0 cuda:1 --envs-per-shape 408 --steps 256 \
  --success-noise-curriculum --adaptive-sampling \
  --physics-cuda-graph --observation-cuda-graph \
  --distributed-ppo
```

### 4.2 方案 B：基于已有高轮次 Checkpoint 热启动续训

如果希望复用已有检查点（例如已通过孔间隙课程的 `checkpoint.pt`）快速适应新噪声、新初态与新入孔前倾角权重：

1. 新建运行目录并复制旧 `checkpoint.pt` 与 `config.json`：
   ```bash
   mkdir -p runs/real_robot_resume
   cp runs/<old_run>/checkpoint.pt runs/real_robot_resume/
   cp runs/<old_run>/config.json runs/real_robot_resume/
   ```
2. 更新 `runs/real_robot_resume/config.json` 中 `"environment"` 字典的对应字段：
   ```bash
   micromamba run -r ~/micromamba -n mj python -c '
   import json, math
   from dataclasses import asdict, replace
   from pathlib import Path
   from insertion import InsertionCfg
   from train import contract

   p = Path("runs/real_robot_resume/config.json")
   meta = json.loads(p.read_text())
   cfg = replace(
       InsertionCfg.from_checkpoint(meta["environment"]),
       moment_noise=0.015,
       force_bias=0.5,
       moment_bias=0.1,
       preinsertion_tilt_weight=0.5,
       initial_z_max=0.02,
       initial_tilt=math.radians(8.0),
   )
   meta["environment"] = asdict(cfg)
   meta["contract"] = contract(cfg)
   meta.pop("adaptive_sampling_state", None)
   p.write_text(json.dumps(meta, indent=2) + "\n")
   '
   ```
3. 启动 `--resume`（训练器会自动检测到环境参数变更，保留当前课程等级并清空旧窗口重新统计达标率）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 micromamba run -r ~/micromamba -n mj python -u train.py \
     --resume --output runs/real_robot_resume --device cuda:0
   ```

### 4.3 导出真机部署 ONNX 模型

训练达到满课程等级（`level = 15`，即 `0.1 mm` 间隙 + `1.0` 满噪声尺度）且各形状成功率稳定后，导出裁剪探索标准差头的确定性 ONNX 模型并拷贝回真机工作空间 `/home/a/piper_x_ws/src/insert/actor.onnx`：

```bash
micromamba run -r ~/micromamba -n mj python export_policy.py \
  runs/real_robot_v1/checkpoint.pt --onnx --prune-std --output runs/export_onnx
```
