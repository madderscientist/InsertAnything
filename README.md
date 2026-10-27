# 基于 mjlab 的通用六自由度多几何力控插入策略

本项目使用 `mjlab` 批量 MuJoCo GPU 仿真与双层 GRU 循环 PPO，训练**单一策略网络**同时完成三角形、方形、正五边形、正六边形和 L 形的高精度力控插孔，并在未见过的圆形孔上检验零样本泛化能力。

奖励函数、运行成本上界与终止判据详见 [REWARD.md](./REWARD.md)。

观测误差消融及限制见 [OBSERVATION_ERROR_ANALYSIS.md](./OBSERVATION_ERROR_ANALYSIS.md)。
方形和 L 形使用 box 组合，其他多边形使用凸 mesh。仿真容量
`nconmax/njmax=64/256`，圆形为 `512/2048`；接触池按同组环境数共享。

## 1. 原始 InsertAnything 实现的缺陷与我们的方法对比

通过对原始 [InsertAnything](https://arxiv.org/abs/2609.24511) 开源仓库代码的审查，可以发现其实现为掩盖奖励诱导悬停与高维探索困难，依赖了大量在通用真机场景下不成立的硬编码简化：

| 维度 | 原仓库 (`InsertAnything`) 的问题 | 本项目的方法与改进 |
|---|---|---|
| **悬停刷分与回合终止** | **全正奖励 + 永不提前终止**：原仓库对 XY 对齐与偏航角持续发放正奖励，若提前终止会导致智能体损失后续正分；为此原仓库在 `_get_dones` 中删去了成功终止与安全终止，所有回合硬跑满固定时长（10–20 s）。即便如此，孔口上方“无接触悬停”仍是旱涝保收的局部极大值，一旦下探接触孔口反而触发平方接触力惩罚与 `1°` 偏航硬门控截断 | **严格负运行成本 + 一次性成功终局奖励**：每步非终止状态严格为负分（含固定时间成本 `-0.02/步`），彻底消除悬停刷分与拖延动机；提供 `5 N / 0.1 N·m` 接触死区鼓励利用接触力探索孔口；达标保持 5 步后立即终止并发放 `+20` 奖励 |
| **控制自由度** | **假 6 轴、实 4 轴**：虽然声明 6 维动作空间，但在 `_apply_action` 中每步硬编码覆盖 `roll = π, pitch = 0`，强行锁死两个倾角自由度，不允许插头发生任何倾斜 | **真 6 自由度全姿态控制**：开放完整三维平移 + 三维旋转增量控制，初始状态即包含 `±3°` 随机倾角，配合 6 轴各向异性阻抗控制、刚度主轴随机旋转与时变刚度游走 |
| **多形状泛化与观测作弊** | **一形状一模型 + 观测预折叠**：每个形状与公差单独训练一个网络；且在输入给 Actor 前先用已知几何的 `symmetry_angles_deg` 将相对四元数折叠到最近对称轴 | **单一权重通吃 5 种形状 + 未见圆形测试**：Actor 严格只接收 **13 维物理观测**，不输入形状 ID，也不对 Actor 观测做对称性折叠，完全依靠双层 GRU 从接触力旋量历史中隐式推断几何约束 |
| **力觉传感器物理** | 仅读取 3 维线性接触力（无力矩），关闭重力且忽略传感器下游质量惯性与陀螺力矩；在多个任务配置中甚至默认关闭了力观测 | **完整 6 轴腕部力/力矩传感器建模**：建模接触载荷减去下游质量加速度与陀螺力矩，经低通滤波后从腕部变换至名义尖端（$M_T = M_S - r_{ST}\times F_S$），并叠加传感器安装角误差、尖端标定误差与真实夹持偏移 |
| **成功与安全标准** | 成功判据硬编码为 `xy_dist < 2.5 mm`（远大于孔间隙）、仅固定插到底部、无倾角与过载检查，且单步擦边即记为成功 | **随机目标深度 + 深度保持判据**：目标深度每回合在 `5–22.5 mm` 随机采样；真实尖端达到目标深度（允许浅 `0.2 mm`）并**连续保持 5 步**，不额外要求 XY/倾角/偏航对齐，依赖孔壁物理约束；峰值力 `> 30 N` 或力矩 `> 2 N·m` 触发 `-200` 安全失败终止 |
| **课程学习机制** | 无自动课程，依赖人工拆分为 `_I (2 mm)`、`_II (0.5 mm)`、`_III (0.1 mm)` 等独立配置分别调参 | **全自动孔间隙 + 难度 + 噪声课程**：从 `2.0 mm` 孔间隙自动按五种形状 `80%` 成功率逐级收紧至 `0.1 mm`，同步提升初始位姿/刚度难度，达标后再逐级开启全量传感器与标定噪声 |


## 2. 环境配置与安装

运行环境需要支持 CUDA 的 Linux 系统与 Python 3.11+。推荐使用 `micromamba` 或 `conda` 创建虚拟环境，并通过 [requirements.txt](./requirements.txt) 安装依赖：

```bash
micromamba create -r ~/micromamba -n mj -c conda-forge python=3.13 pip -y
micromamba run -r ~/micromamba -n mj python -m pip install -r requirements.txt
```

## 3. 核心命令

```bash
# 运行全部单元测试与 GPU 物理集成测试
micromamba run -n mj python -m unittest discover -s test -t . -v

# 短周期冒烟训练（验证环境与管线）
micromamba run -n mj python train.py \
  --envs-per-shape 4 --steps 16 --iterations 2 --output runs/smoke

# 单卡有限轮次训练（默认开启孔间隙课程）
micromamba run -n mj python train.py \
  --envs-per-shape 64 --steps 256 --iterations 2000 \
  --noise-curriculum --physics-cuda-graph --observation-cuda-graph \
  --output runs/train

# 双卡分布式持续训练（不传 --iterations 时默认进入带守护进程的持续训练模式）
micromamba run -n mj python train.py \
  --output runs/continuous_dual --device cuda:0 \
  --simulation-devices cuda:0 cuda:1 --envs-per-shape 408 --steps 256 \
  --success-noise-curriculum --physics-cuda-graph --observation-cuda-graph \
  --distributed-ppo

# 调整正在运行的训练环境数（保留网络、归一化器、优化器与课程状态）
micromamba run -n mj python train.py \
  --resize-envs --envs-per-shape 204 --output runs/continuous_dual

# 优雅停止持续训练（当前 PPO 轮次保存完成后退出）
touch runs/continuous_dual/STOP

# 从检查点恢复持续训练
rm -f runs/continuous_dual/STOP
micromamba run -n mj python train.py \
  --resume --output runs/continuous_dual

# Viser 网页实时查看训练快照（CPU 正向运动学，不占用 GPU 采样）
# 红/蓝箭头为滤波力/力矩（非安全峰值，不含额外白噪声和 bias）；
# 5 N / 0.1 N·m 各对应 30 mm，箭头偏移不代表作用点。
micromamba run -n mj python viser_view.py \
  --run-dir runs/continuous_dual --host 127.0.0.1 --port 8080

# Viser 独立缓冲回放最新策略（按 1x 仿真时间 50 Hz 播放）
micromamba run -n mj python viser_view.py \
  --run-dir runs/continuous_dual --mode policy --device cuda:1 \
  --host 127.0.0.1 --port 8080

# Geometry 常驻包含圆形；未训练形状自动切换回放，不改变训练任务。
# 圆柱与网格接触仍受仿真器单接触点限制。
micromamba run -n mj python viser_view.py \
  --run-dir runs/continuous_dual --mode policy --shape circle --device cuda:0

# --ignore-overload 仅关闭仿真回放的过载终止，保留载荷/越界/超时检查，非安全评估。

# 冻结策略多条件泛化评估（输出 trials.csv 与 summary.json）
micromamba run -n mj python evaluate.py runs/continuous_dual/checkpoint.pt \
  --seeds 0 42 2026 --trials 16 --dynamics nominal soft stiff \
  --clearances 0.0001 0.0005 --scales 1.0 0.8 1.2 \
  --target-depths 0.005 0.015 0.020 --output runs/evaluation

# 本地 MuJoCo 交互窗口或无头模式渲染快照
micromamba run -n mj python play.py runs/continuous_dual/checkpoint.pt --shape l
MUJOCO_GL=egl micromamba run -n mj python play.py runs/continuous_dual/checkpoint.pt \
  --shape hexagon --headless --steps 50 --snapshot runs/hexagon.png

# 导出 TorchScript 离线策略并校验序列等价性与单步延迟
micromamba run -n mj python export_policy.py runs/continuous_dual/checkpoint.pt \
  --output runs/export
```


### 单卡持续训练

以下为 24 GB RTX 4090 的参考配置，较小显存应降低环境数。启动新训练时不要复用已有输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -r ~/micromamba -n mj python -u train.py \
  --output runs/continuous_single --device cuda:0 \
  --envs-per-shape 512 --steps 256 \
  --success-noise-curriculum --physics-cuda-graph --observation-cuda-graph

```

此配置共 2560 个环境，每轮 655360 条 transition。扩容保留模型、优化器和课程，
重置回合及 GRU 状态；恢复使用保存配置，命令行噪声参数不会覆盖它。

每轮更新 `checkpoint.pt`、上一轮 `previous.pt`；每 500 轮归档至
`checkpoints/checkpoint_<八位轮次>.pt`，同名不覆盖、不自动清理。
Release 续训需要完整 checkpoint 和对应 `config.json`；ONNX 仅供推理。
恢复保留 actor、critic、归一化器、优化器和进度，但重新初始化回合与 GRU 状态。

每轮保存后释放未使用的 CUDA 缓存，显存统计见 `heartbeat.json`。
`termination_counts.json` 为当前 worker 启动以来的互斥终止计数，多卡仅含 rank 0。
`overload_stages` 分别按高度、时间、倾角分桶，取 20 ms 终止步末状态，
不是 1 ms 冲击瞬间，不能据此认定失败根因。

### 回合固定力觉偏置与柔和自适应采样

新建持续训练时可在以上命令追加：

```bash
--force-bias 0.5 --moment-bias 0.1 --adaptive-sampling
```

`force_bias`、`moment_bias` 分别为力/力矩零点偏置的三维向量幅度上限，
单位为 N 和 N·m。默认 `wrench_bias_distribution=uniform_magnitude`：
力与力矩分别独立采样球面均匀方向，幅度在 `[0, 上限]` 上均匀采样（不是球体体积均匀）。
每回合 reset 时采样并乘以当时的 `noise_scale`，回合内固定；
叠加于滤波后的 Actor 力旋量，不改变真实载荷、奖励、安全判据或 57 维 Critic。
当前新训练默认力 bias 幅度 0–0.5 N、力矩 bias 幅度 0–0.1 N·m；
显式设为零可关闭。旧检查点缺少分布字段时按 `per_axis` 恢复原逐轴采样；
旧检查点缺少 bias 数值字段时仍按零加载，保持历史行为；
零噪声回放也会禁用这些偏置。
力/力矩测量另含 play-operator 滞回：新训练默认回线宽度为配置量程的
0.1%（按 30 N、2 N·m 量程即 0.03 N、0.002 N·m），内部半宽为其一半；
输出跟踪滤波后腕部测量，保留加载/卸载方向记忆，不随噪声课程缩放。
滞回也会进入使用该滤波测量的负载奖励；原始载荷安全检查和 Critic 不受影响。
这是无具体传感器型号时采用的通用近似；可用
`--wrench-hysteresis-fraction 0` 关闭。旧检查点缺字段时滞回为零。
恢复时若幅度或分布改变，保留课程等级但清空旧成功率窗口，避免混用不同噪声分布。
`--resume` 使用保存配置，命令行噪声参数不会覆盖已有配置。

自适应采样仅支持成功率课程的守护式持续训练：每 30 轮检查一次，窗口填满后按
**30% 均匀 + 70% 失败率比例**重分配环境（失败率 EMA 新值权重 0.2）。
总环境数不变，按 GPU 数取整，每形状每卡至少一个；零失败时均分，课程升级重置 EMA。
窗口每批等待所有环境各完成一个回合，避免快速成功造成偏差；重启丢弃未完成批次。
布局在保存后重建，保留模型、优化器及课程，重置回合和 GRU；
调整记录见 `sampling_<iteration>.json`，心跳记录各形状环境数。

### 初始位姿范围

初始位置指**真实尖端**相对孔口的位置。新训练默认 X/Y 各独立均匀采样于 ±20 mm，
Z 在孔口上方 4–10 mm；姿态旋转向量 X/Y 分量各 ±3°，Z 分量 ±15°。
X/Y 位置和三个旋转分量随 `difficulty` 缩放，Z 高度不缩放。
旋转向量分量不是独立欧拉角，也不是总转角上限。

横向安全圆盘半径为 `workspace_xy_radius_m=0.035`，配置验证初始正方形完全位于其中。
该半径不随 `scale` 缩放；旧检查点缺字段时沿用 `0.025*scale`，
缺 `initial_xy` 时沿用 3 mm。实际运行范围以保存配置为准。

`initial_plane_margin>0` 时根据实际外形抬高 Z 下界，确保整个插头高于孔板。
仅在 reset 计算；不可行姿态最多重采样 16 次，仍失败则报错，不突破 Z 上限。
默认零保持旧采样方式。

`plate_outer_radius_m=0.080` 使孔板至少具有 80 mm 内切半径，覆盖工作空间、
插头端面及安全倾角投影；旧配置缺字段时保留原板面。孔口和间隙不变。

`success_opening_required=true` 要求成功保持期间真实端面中心 XY 在孔内，
不以带噪位置判定，也不要求整个投影包含；旧检查点缺字段按 false 恢复。
位姿范围、板面或成功规则变化后需重建环境并清空旧成功统计，保留课程等级。

## 4. 架构与方法设计

### 4.1 观测、动作与网络结构
孔坐标系 `H` 固定于孔口中心；工具坐标系 `T` 原点位于**名义插头尖端**；虚拟六轴腕部传感器坐标系 `S` 位于质心上方。单位统一采用国际单位制（`m, s, rad, N, N·m`），四元数顺序为 `wxyz`。

- **Actor 输入（13 维）**：
  1. `H` 系中目标位置减去当前估计尖端位置（3 维）
  2. 规范化相对四元数 `inverse(current) * target`（4 维，消除 `q/-q` 双重覆盖）
  3. `T` 系中关于名义尖端的重力补偿后滤波力旋量 `wrench`（6 维）
- **位置延迟**：`position_observation_delay=true` 时，每回合固定随机延迟
  0/20 ms，仅影响 Actor 的完整带噪位置误差；姿态、力觉及 Critic 仍取当前帧。
  reset 初始化历史，每策略步只推进一次，读取和 CUDA 捕获不推进历史。
  `--no-position-observation-delay` 可关闭；旧检查点缺字段按 false 恢复。
- **入孔负载成本**：`inserted_load_cost=true` 时，真实端面中心 `z<-4 mm`
  后惩罚孔系 Fx/Fy 及全部力矩，无死区；此前使用 5 N/0.1 N·m 全范数死区。
  原始载荷安全检查不变。`--no-inserted-load-cost` 或旧配置缺字段时沿用原奖励，
  详见 [REWARD.md](./REWARD.md)。
- **Critic 特权输入（57 维）**：包含无噪真实状态、线速度/角速度、6 轴刚度与阻尼、质量/主惯量、下游质量/惯量、标定误差、夹持偏移及几何类别独热编码。
- **循环策略网络**：观测经验归一化 → 两层 `GRU(128)` → `ELU MLP(128, 64)` → 每个动作的均值和 `log_std`（各 6 维）。采用 rsl_rl 原生 `HeteroscedasticGaussianDistribution`，各维独立采样，标准差限制为 `[0.02, 1.0]`；根据观测与循环历史学习探索强度，不硬编码入孔阶段。评估和 TorchScript/ONNX 导出仍只使用 6 维均值动作。
  不再维护独立的全局 std 参数；`init_std` 只用于新网络的输出层初始化。日志 `Mean action std` 是最后一次策略分布计算中 std 张量的整体均值，不是按入孔阶段统计的 rollout 均值。
- **动作语义**：输出 `H` 系下无裁剪的三维平移与三维旋转向量；分别乘以 `1 mm` 与 `0.5°` 的每分量尺度，以估计名义尖端为旋转中心合成下一策略周期的阻抗目标：
  $$p_{\mathrm{target}} = p_{\mathrm{current}} + \Delta p, \qquad q_{\mathrm{target}} = \mathrm{Exp}(\Delta r) \otimes q_{\mathrm{current}}$$

旧运行不会自动切换分布。安全停止后，可迁移到**新的输出目录**，保留原运行用于对照和回退：

```bash
python migrate_policy_std.py runs/continuous_single/checkpoint.pt \
  --output runs/continuous_state_std
python train.py --resume --output runs/continuous_state_std --device cuda:0
```

迁移保留均值策略、GRU、归一化、Critic、进度及兼容的优化器状态；
std 新权重置零、偏置继承旧 `log_std`，新增优化器矩置零，删除旧全局 std。
课程和采样历史重置，环境均分。可选 `--initial-std 0.1 --noise-level 1`
指定初始 std 和 10% 噪声；省略时继承旧 std，从课程 0 开始。迁移收益需另行评估。

### 4.2 参数化凸几何与物理模型
- **几何构建**（[insertion/geometry.py](./insertion/geometry.py)）：无需外部 USD 资产，程序化生成正三角形、方形、正五边形、正六边形、L 形及测试用圆形。正多边形外接圆半径 `10 mm`，L 形包络 `20 mm`，孔深 `25 mm`，插头长 `35 mm`。孔壁拆分为独立凸棱柱以保持中心孔贯通，L 形插头拆分为双盒体保留凹口。
  圆孔默认自适应选择边数，也可通过 `InsertionCfg.circle_sides` 显式指定；圆柱插头不变。`clearance` 为总间隙（直径差），例如 `0.0002 m`、64 边形圆孔的最小实际单侧间隙约为 `87.95 μm`；过低边数导致几何干涉时会报错。该配置不影响其他形状。
- **传感器动力学**（[insertion/env.py](./insertion/env.py)）：关闭重力代表理想重力补偿。腕部传感器测量外部接触载荷减去传感器下游质量加速度与陀螺力矩，经时间常数 `15 ms` 的一阶低通滤波后搬移至名义尖端。
- **各向异性阻抗与时变游走**：支持 `train`、`nominal`、`soft`、`stiff` 四种动力学模式。训练模式下刚度主轴随机旋转，且刚度在回合初始值 `±10%` 范围内每秒采样局部目标并以 `1 s` 时间常数平滑游走（变化率上限 `5%/s`）。
- **CUDA Graph 加速**：`--physics-cuda-graph` 分别捕获动作处理、完整 20 个物理子步、奖励/终止计算及重置时的模型常量重计算；`--observation-cuda-graph` 捕获确定性观测变换。保持 `1000 Hz` 物理步长和逐子步控制/传感器更新；有限值检查、随机采样、动态重置和课程推进仍在图外。预热后捕获，模型重建后重新捕获，返回张量不受后续重放覆盖。启用 mjlab NaN 调试守卫时保留逐子步调用以执行调试回调。单卡与多卡各 rank 共用优化路径，多卡端到端收益尚未实测。

### 4.3 孔间隙、难度与噪声三阶段自动课程
[insertion/curriculum.py](./insertion/curriculum.py) 中的 `SuccessNoiseCurriculum` 按每种形状最近 **1000 个完成回合**的真实成功率推进课程（所有训练形状成功率均达到 **80%** 时升级，清空窗口并过滤跨等级旧回合）：

1. **阶段一（孔间隙与初始难度课程，零噪声）**：
   孔间隙按 `2.0 mm → 1.0 mm → 0.5 mm → 0.25 mm → 0.15 mm → 0.10 mm` 逐级收紧；同时初始位姿与刚度随机化难度 `difficulty` 从 `0.25` 线性提升至 `1.0`。升级瞬间自动重建 MuJoCo 几何模型并重捕获 CUDA Graph。成功要求达到目标深度并连续保持 100 ms，不额外施加严格居中或姿态门槛。
2. **阶段二（传感器与标定噪声课程，固定 `0.10 mm` 间隙与 `1.0` 难度）**：
   在目标 `0.10 mm` 间隙下，每次达标将 `noise_scale` 提升 `0.1` 直至 `1.0`，逐步激活逐步观测噪声（位置 `0.5 mm`、姿态 `1.5°`、力 `0.2 N`、力矩 `0.001 N·m`）与回合固定误差（真实夹持偏移 `1.25 mm`、尖端标定误差 `0.5 mm`、目标位置偏差 `0.5 mm`、传感器旋转误差 `0.5°`）。


### 4.4 当前训练与独立诊断

运行参数以对应 `config.json` 和 checkpoint 为准，不以代码默认值推断历史训练。
普通入口不启用失败池、随机意图或恢复课程；独立恢复试验见
[RECOVERY_EXPERIMENT.md](./RECOVERY_EXPERIMENT.md)。
观测消融入口为 [evaluate_noise_ablation.py](./evaluate_noise_ablation.py)，
结果及限制见 [OBSERVATION_ERROR_ANALYSIS.md](./OBSERVATION_ERROR_ANALYSIS.md)；
oracle 对照不代表部署能力。

## 5. 离线推理接口

ONNX（显式 GRU 状态、确定性动作均值、内置观测归一化）导出：

```bash
micromamba run -n mj python -m pip install onnx onnxscript onnxruntime
micromamba run -n mj python export_policy.py runs/continuous_single/checkpoint.pt \
  --onnx --output runs/export_onnx
```

导出自动验证 ONNX/PyTorch 多步动作与隐藏状态等价性，并附带
[DEPLOYMENT_ONNX.md](./DEPLOYMENT_ONNX.md)。模型不包含坐标变换、滤波、动作限幅或安全检查；
实机部署需由独立控制器负责，仿真策略尚未完成实机验证。

[export_policy.py](./export_policy.py) 将训练好的 Actor 导出为确定性有状态 TorchScript 模型，并提供无机器人 I/O 依赖的 `OfflinePolicy` 封装：

```python
from pathlib import Path
from export_policy import OfflinePolicy

policy = OfflinePolicy(Path("runs/export"))
policy.reset()  # 每个新回合开始或异常中止后重置 GRU 隐状态
p_target, q_target = policy.step(p_tip, q_tool, p_goal, q_goal, wrench_tip)
```
