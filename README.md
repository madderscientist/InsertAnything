# 基于 mjlab 的通用六自由度多几何力控插入策略

本项目使用 `mjlab` 批量 MuJoCo GPU 仿真与 GRU 循环 PPO，训练**单一策略网络**同时完成三角形、方形、正五边形、正六边形和 L 形的高精度力控插孔，并在未见过的圆形孔上检验零样本泛化能力。

奖励函数、运行成本上界与终止判据详见 [REWARD.md](./REWARD.md)；真机适配见第 6 节。

方形和 L 形使用 box 组合，其他多边形使用凸 mesh。仿真容量
`nconmax/njmax=64/256`，圆形为 `512/2048`；接触池按同组环境数共享。

## 1. 原始 InsertAnything 实现的缺陷与我们的方法对比

通过对原始 [InsertAnything](https://arxiv.org/abs/2609.24511) 开源仓库代码的审查，可以发现其实现为掩盖奖励诱导悬停与高维探索困难，依赖了大量在通用真机场景下不成立的硬编码简化：

| 维度 | 原仓库 (`InsertAnything`) 的问题 | 本项目的方法与改进 |
|---|---|---|
| **悬停刷分与回合终止** | **全正奖励 + 永不提前终止**：原仓库对 XY 对齐与偏航角持续发放正奖励，若提前终止会导致智能体损失后续正分；为此原仓库在 `_get_dones` 中删去了成功终止与安全终止，所有回合硬跑满固定时长（10–20 s）。即便如此，孔口上方“无接触悬停”仍是旱涝保收的局部极大值，一旦下探接触孔口反而触发平方接触力惩罚与 `1°` 偏航硬门控截断 | **严格负运行成本 + 一次性成功终局奖励**：每步非终止状态严格为负分（含固定时间成本 `-0.02/步`），彻底消除悬停刷分与拖延动机；提供 `5 N / 0.1 N·m` 接触死区鼓励利用接触力探索孔口；达标保持 5 步后立即终止并发放 `+20` 奖励 |
| **控制自由度** | **假 6 轴、实 4 轴**：虽然声明 6 维动作空间，但在 `_apply_action` 中每步硬编码覆盖 `roll = π, pitch = 0`，强行锁死两个倾角自由度，不允许插头发生任何倾斜 | **真 6 自由度全姿态控制**：开放完整三维平移 + 三维旋转增量控制，初始状态即包含 `±3°` 随机倾角，配合 6 轴各向异性阻抗控制、刚度主轴随机旋转与时变刚度游走 |
| **多形状泛化与观测作弊** | **一形状一模型 + 观测预折叠**：每个形状与公差单独训练一个网络；且在输入给 Actor 前先用已知几何的 `symmetry_angles_deg` 将相对四元数折叠到最近对称轴 | **单一权重通吃 5 种形状 + 未见圆形测试**：Actor 严格只接收 **13 维物理观测**，不输入形状 ID，也不对 Actor 观测做对称性折叠，完全依靠 actor 的 GRU 从接触力旋量历史中隐式推断几何约束 |
| **力觉传感器物理** | 仅读取 3 维线性接触力（无力矩），关闭重力且忽略传感器下游质量惯性与陀螺力矩；在多个任务配置中甚至默认关闭了力观测 | **完整 6 轴腕部力/力矩传感器建模**：建模接触载荷减去下游质量加速度与陀螺力矩，经低通滤波后从腕部变换至名义尖端（$M_T = M_S - r_{ST}\times F_S$），并叠加传感器安装角误差、尖端标定误差与真实夹持偏移 |
| **成功与安全标准** | 成功判据硬编码为 `xy_dist < 2.5 mm`（远大于孔间隙）、仅固定插到底部、无倾角与过载检查，且单步擦边即记为成功 | **随机目标深度 + 深度保持判据**：目标深度每回合在 `5–22.5 mm` 随机采样；真实尖端达到目标深度（允许浅 `0.2 mm`）并**连续保持 5 步**，不额外要求 XY/倾角/偏航对齐，依赖孔壁物理约束；峰值力 `> 30 N` 或力矩 `> 2 N·m` 施加过载惩罚，越界触发 `-800` 安全失败终止 |
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
窗口每批等待所有环境各完成一个回合，避免快速成功造成偏差；重启丢弃未完成批次，
故仅当某形状环境数变化达 10% 才重启，否则只更新 EMA、维持当前布局（ADR 每批约需 15–20 轮，每 30 轮重启会丢掉约一半判定）。
每批完成时记录各形状两维边界回合成功率 `curriculum/boundary_success/{noise,clearance}/<shape>`。
布局在保存后重建，保留模型、优化器及课程，重置回合和 GRU；
调整记录见 `sampling_<iteration>.json`，心跳记录各形状环境数。
搜索训练（`search_prior/train_search.py`）默认开启自适应采样；课程要求所有形状达标才升级，双插头 dual 常是最慢的一个。

### 初始位姿范围

初始位置指**真实尖端**相对孔口的位置。新训练默认 X/Y 各独立均匀采样于 ±20 mm，
Z 在孔口上方 4–20 mm；姿态旋转向量 X/Y 分量各 ±8°，Z 分量 ±15°。
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
- **循环策略网络**：观测经验归一化 → `GRU(128)`（actor 一层、critic 两层）→ `ELU MLP(128, 64)` → 每个动作的均值和 `log_std`（各 6 维）。采用 rsl_rl 原生 `HeteroscedasticGaussianDistribution`，各维独立采样，标准差限制为 `[0.02, 1.0]`；根据观测与循环历史学习探索强度，不硬编码入孔阶段。评估和 TorchScript/ONNX 导出仍只使用 6 维均值动作。
  不再维护独立的全局 std 参数；`init_std` 只用于新网络的输出层初始化。日志 `Mean action std` 是最后一次策略分布计算中 std 张量的整体均值，不是按入孔阶段统计的 rollout 均值。
- **搜索训练网络**：actor 和 critic 均使用 [insertion/model.py](./insertion/model.py) 的 `SkipRNNModel`：actor 使用一层 `GRU(256)`，critic 使用两层。归一化观测经过 GRU，输出与当前观测拼接 → `ELU MLP(256, 128)`。GRU 用 chrono 初始化（更新门偏置 $\log U(1, 499)$，初始记忆时间常数铺满 2–500 步），输出不加 LayerNorm；默认初始化的更新门约 0.5，记忆只有约 2 步。critic 也需记忆：只看当前特权状态的 MLP critic 在部分可观测任务中有偏，两次训练均崩塌。

  搜索训练每个 minibatch 按策略 KL 同步调整 actor 和 critic 的学习率，默认下限为 `1e-5`。训练配置中的 `critic_learning_rate="actor"` 表示联动，`--critic-lr 1e-4` 可切换为固定学习率。续训使用检查点配套的网络层数。
  新建搜索训练默认使用阶段监督。线性分类头从 actor 的 GRU 输出预测悬空、贴板、孔口受阻和已入孔四个阶段。交叉熵权重为 `0.1`，分类头学习率固定为 `1e-3`。`--phase-coef 0` 可关闭，已有实验按保存的配置续训。

  ```mermaid
  flowchart LR
    subgraph A["Actor（部署，GRU 隐状态 1×256 逐步传递）"]
      direction TB
      ao["观测 13 维<br/>位置误差 3 + 相对四元数 4 + 力旋量 6"] --> an["经验归一化"]
      an --> ag["GRU 256 × 1 层<br/>chrono 初始化"]
      an -->|跳连| ac(("拼接 269"))
      ag --> ac
      ac --> am1["Linear 256 + ELU"] --> am2["Linear 128 + ELU<br/>末层特征 z"] --> amu["Linear 6<br/>均值 μ"]
      am2 -.->|截断梯度、单位化| gs["gSDE 噪声 z·(ε⊙σ)<br/>σ 为 128×6 参数，ε 每 25 步重抽"]
      amu --> act(["动作 = μ + 噪声（训练）/ μ（部署）<br/>平移 ×1 mm，旋转 ×0.5°"])
      gs -.-> act
    end
    subgraph C["Critic（仅训练）"]
      direction TB
      co["特权观测 63 维"] --> cn["经验归一化"] --> cg["GRU 256 × 2 层"]
      cn -->|跳连| cc(("拼接 319"))
      cg --> cc --> cm["ELU MLP 256, 128"] --> v(["价值 V"])
    end
  ```
- **搜索训练的探索噪声（gSDE）**：[search_prior/train_search.py](./search_prior/train_search.py) 改用 [insertion/model.py](./insertion/model.py) 中的 `GSDEDistribution`（generalized State-Dependent Exploration）。逐步独立的高斯噪声在 50 Hz 下会相互抵消，累计位移只按 $\sqrt{N}$ 增长且被动作变化惩罚压制；gSDE 使探索在一段时间内方向连贯。
  - 记 MLP 最后一个隐层特征为 $z(s)$（截断梯度并单位化，$\lVert z\rVert=1$），可学习参数为 $d\times 6$ 的 $\log\sigma$。每个环境保存探索矩阵 $\varepsilon\sim\mathcal N(0,1)^{d\times 6}$，动作为
    $$a = \mu(s) + z(s)^\top(\varepsilon\odot\sigma)$$
    给定状态时 $a\mid s\sim\mathcal N(\mu(s),\hat\sigma^2(s))$，$\hat\sigma_j^2(s)=\sum_i z_i^2\sigma_{ij}^2$；PPO 的对数概率、熵与 KL 均按该对角高斯计算，忽略 $\varepsilon$ 的跨步相关（与原论文及 Stable-Baselines3 一致）。$\hat\sigma$ 截断到 `std_range` 时噪声同比缩放，采样与似然保持一致。
  - $\varepsilon$ 在回合结束时按环境重抽，并每 `--gsde-resample-steps`（默认 25 步即 0.5 s）整体重抽；持有期间同一状态得到同一扰动。存储的是标准正态 $\varepsilon$，噪声尺度始终跟随当前 $\sigma$。
  - 初始化 $\sigma_{ij}=$ `init_std`（特征单位化后即 $\hat\sigma=0.4$），探索强度不随特征模长漂移；日志 `Mean action std` 为 $\hat\sigma$ 均值。
  - 训练批次只重算 $\hat\sigma$ 不采样；评估与 TorchScript/ONNX 导出只用均值，输出层即 6 维均值（无 std 头，`--prune-std` 不适用）。
  - 对照（单卡 1024 环境/形状，课程第 0 级，无初始下压偏置，60 轮）：gSDE 约 25 轮后自行学会下压，第 29 轮成功率 0.80，第 59 轮 0.93；逐步独立噪声 40 轮成功率始终约 0，悬停在板面上方约 1 mm 且 std 收缩到 0.29。
- **搜索训练第一阶段固定关闭停滞惩罚**（从零训练，`search_prior/train_search.py` 写死为 0，不提供开关）：停滞项只在尖端净推进慢于阈值时扣分，板面受阻时必然触发，空中漂移则可规避。从零训练时它把策略推离板面：开启时（其余同上）超时尖端高度中位数从 14 mm 升到 54 mm，60 轮成功率为 0；关闭后同配置达到 0.93。安全罚分 800/250/100 对成功率无影响，不是原因。
- **动作语义**：输出 `H` 系下无裁剪的三维平移与三维旋转向量；分别乘以 `1 mm` 与 `0.5°` 的每分量尺度，以估计名义尖端为旋转中心合成下一策略周期的阻抗目标：
  $$p_{\mathrm{target}} = p_{\mathrm{current}} + \Delta p, \qquad q_{\mathrm{target}} = \mathrm{Exp}(\Delta r) \otimes q_{\mathrm{current}}$$

### 4.2 参数化凸几何与物理模型
- **几何构建**（[insertion/geometry.py](./insertion/geometry.py)）：无需外部 USD 资产，程序化生成正三角形、方形、正五边形、正六边形、L 形及测试用圆形。正多边形外接圆半径 `10 mm`，L 形包络 `20 mm`，孔深 `25 mm`，插头长 `35 mm`。孔壁拆分为独立凸棱柱以保持中心孔贯通，L 形插头拆分为双盒体保留凹口。
  圆孔默认自适应选择边数，也可通过 `InsertionCfg.circle_sides` 显式指定；圆柱插头不变。`clearance` 为总间隙（直径差），例如 `0.0002 m`、64 边形圆孔的最小实际单侧间隙约为 `87.95 μm`；过低边数导致几何干涉时会报错。该配置不影响其他形状。
- **形状定义**（[insertion/shapes/](./insertion/shapes/)）：每种形状一个文件，集中定义插头与孔板的截面、碰撞体和对称性；通用场景组装仍由几何模块负责。`get_shape(name).mirror_axis_xy` 是孔坐标系中穿过原点的镜像轴方向，例如 `(0, 1)` 表示 `x → −x`，L 形用 `(1, 1)` 表示 `x ↔ y`。维护注册表时不要重排已有形状，顺序决定 Critic 的形状通道和随机种子偏移。
- **传感器动力学**（[insertion/env.py](./insertion/env.py)）：关闭重力代表理想重力补偿。腕部传感器位于尖端上方 8–16 cm（real2sim 18–22 cm），测量外部接触载荷减去下游刚体（0.4–0.9 kg、质心在传感器下方 0.3–0.6 倍距离并横向偏移 ±5 mm、细杆惯量）的惯性力旋量，经时间常数 `15 ms` 的一阶低通滤波后搬移至名义尖端。
- **各向异性阻抗与时变游走**：支持 `train`、`nominal`、`soft`、`stiff` 四种动力学模式。训练模式下刚度主轴随机旋转，且刚度在回合初始值 `±10%` 范围内每秒采样局部目标并以 `1 s` 时间常数平滑游走（变化率上限 `5%/s`）。
- **CUDA Graph 加速**：`--physics-cuda-graph` 分别捕获动作处理、完整 20 个物理子步、奖励/终止计算及重置时的模型常量重计算；`--observation-cuda-graph` 捕获确定性观测变换。保持 `1000 Hz` 物理步长和逐子步控制/传感器更新；有限值检查、随机采样、动态重置和课程推进仍在图外。预热后捕获，模型重建后重新捕获，返回张量不受后续重放覆盖。启用 mjlab NaN 调试守卫时保留逐子步调用以执行调试回调。单卡与多卡各 rank 共用优化路径，多卡端到端收益尚未实测。
- **物理步长不改 2 ms（已评估否决）**：单卡 10240 环境基准中 1 ms→2 ms 每步只从 92 ms 降到 52 ms（1.75×，策略推理、重置、观测是固定开销），而 0.1 mm 间隙下接触变软、单步位移接近间隙、力观测失真和低阻尼发散的风险不可接受。求解器迭代从 50/20 降到 20/10 或 10/5 也无加速（求解提前收敛）。提速请走形状组并发流与图捕获，勿再尝试这两项。

### 4.3 孔间隙、难度与噪声三阶段自动课程
[insertion/curriculum.py](./insertion/curriculum.py) 中的 `SuccessNoiseCurriculum` 按每种形状最近 **1000 个完成回合**的真实成功率推进课程（所有训练形状成功率均达到 **80%** 时升级，清空窗口并过滤跨等级旧回合）：

1. **阶段一（孔间隙与初始难度课程，零噪声）**：
   孔间隙按 `2.0 mm → 1.0 mm → 0.5 mm → 0.25 mm → 0.15 mm → 0.10 mm` 逐级收紧；同时域随机化难度 `difficulty` 从 `0` 线性提升至 `1.0`，各采样区间按难度从中心值放大到全幅：初始位姿与高度（从 `initial_z_min` 起）、刚度、表观质量、阻尼比（中心 1.0）、下游质量、传感器距离与质心、板面摩擦、静摩擦、孔壁摩擦、枢轴（搜索训练不施加粘滑）；凸沿与困难初态重放的概率乘以难度。升级瞬间自动重建 MuJoCo 几何模型并重捕获 CUDA Graph。成功要求达到目标深度并连续保持 100 ms，不额外施加严格居中或姿态门槛。
2. **阶段二（传感器与标定噪声课程，固定 `0.10 mm` 间隙与 `1.0` 难度）**：
   在目标 `0.10 mm` 间隙下，每次达标将 `noise_scale` 提升 `0.1` 直至 `1.0`，逐步激活逐步观测噪声（位置 `0.5 mm`、姿态 `1.5°`、力 `0.2 N`、力矩 `0.015 N·m`）与回合固定误差（真实夹持偏移 `1.25 mm`、尖端标定误差 `0.5 mm`、目标位置偏差 `0.5 mm`、传感器旋转误差 `0.5°`）。

**搜索训练改用两维 ADR**（`noise_curriculum_mode="adr"`，`ADRCurriculum`），取代上面的分级课程。每形状独立维护 φ = (噪声, 间隙) ∈ [0, 2]²：
- **放开（φ ≤ 1）**：噪声回合系数 s 在 `[0, φ0]` 均匀采样，乘到逐步观测噪声、回合固定误差、振动与阶跃偏置上；回合间隙在 `[c(φ1), 2.0 mm]` 对数均匀采样，c 在 `2.0 mm` 与目标 `0.10 mm` 间对数插值。
- **收拢（φ > 1）**：难侧已满，易侧边界向难侧移动——s 的下限为 `φ0-1`，间隙上限为 `c(φ1-1)`；φ = 2 时整维钉在最难值（s = 1、间隙 `0.10 mm`）。s = 1 时噪声与误差本身仍在零附近到满幅间随机，只是不再有整体更安静的回合。
- **全局 `difficulty` = 各形状 min(φ0, 1)**：初始 X/Y（±20 mm × difficulty）、倾斜、yaw、刚度、阻尼、摩擦、枢轴、凸沿与困难初态概率都按它缩放，最慢形状决定所有形状的初始范围；这些量无回合级系数，difficulty = 1 即全幅均匀分布，无需收拢。
- **调节**：50% 回合为边界回合（每维约 25% 回合在难侧训练），一维钉在难侧边界（s = min(φ0, 1) 或间隙 c(φ1)），另一维照常采样；`curriculum/success/<形状>` 为含边界回合在内的全部回合成功率，仅用于显示和自适应采样；每形状每维累计 200 个边界回合后，成功率 > 0.8 时 φ += 0.05，< 0.5 时 φ -= 0.05（先回退收拢、再回退放开）。课程中途变化或困难状态回放导致未钉在边界的回合不计入。


### 4.4 当前训练与独立诊断

运行参数以对应 `config.json` 和 checkpoint 为准，不以代码默认值推断历史训练。

第二阶段 [train_rough_surface.py](./train_rough_surface.py) 在孔口外侧铺设随机凹凸微砖块并取消超时，训练自主越障与脱困：

```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -n mj python -u train_rough_surface.py \
  --output runs/rough_surface_no_timeout \
  --source-checkpoint runs/continuous_single/checkpoint.pt --device cuda:0
```

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

## 6. real2sim 真机适配

[real2sim/](./real2sim/) 从已部署 checkpoint 续训，将 `REAL2SIM_VALUES`（[insertion/config.py](./insertion/config.py)）的实测对象参数覆盖到源配置；第一阶段默认不启用部署低通及下列对象非理想性；观测 13 维、动作 6 维与契约坐标约定不变，旧检查点缺字段时由 `InsertionCfg.from_checkpoint` 恢复历史行为。主要建模：

- **部署目标低通**：临界阻尼二阶（平移 30、转动 50 rad/s）写入 contract；表观质量 0.08–1.5 kg 对数均匀，经低通加速度前馈复现；按 `target_lowpass_off_probability`（默认 1，即实机部署不加低通）关闭低通，此时平移按 $\min(\sqrt{k/m},100)$ rad/s 的轻质量响应追阶跃。
- **对象非理想性**：Z 向与 XY 转动独立静摩擦死区、转动→平移杠杆耦合、接触粘滑、阶跃力旋量零偏、由自身运动激发且仅在悬空时存在的 2–10 Hz 窄带振动力，以及上节传感器几何与下游惯性力。
- **训练**：从难点状态（贴板、孔口卡滞、半插入）重置；悬空/接触动作变化成本 0.6/0.2；`SmoothPPO` 系数由监控在 0.0125–0.05 间自动调整。
- **第一阶段已并入的通用平滑手段**：悬空/接触动作变化成本 0.5/0.1、动作成本饱和尺度 12、安全罚分 250、传感器几何与下游惯性力、悬空振动噪声（随噪声课程缩放），以及 [insertion/ppo.py](./insertion/ppo.py) 的 `SmoothPPO`（仅在悬空步扰动力旋量观测并惩罚动作均值变化，系数 0.0125）。

```bash
# 从部署 checkpoint 准备并启动守护式续训
micromamba run -n mj python -m real2sim.train_real2sim \
  --source runs/releases/stage2-outer-xy-best-961/checkpoint.pt \
  --output runs/real2sim/stage10 --envs-per-shape 1024 --steps 128

# 每小时健康巡检：存活/停滞重启、开/关低通两档实机评估、最佳保存、停滞/过慢/退化自动调参与回滚
nohup setsid micromamba run -n mj python -m real2sim.monitor --run runs/real2sim/stage10 &

# 实机档评估（固定实测对象参数；--lowpass-off 评估无低通部署）
micromamba run -n mj python -m real2sim.evaluate_real2sim \
  baseline=runs/releases/stage2-outer-xy-best-961/checkpoint.pt \
  new=runs/real2sim/stage10/best_real_checkpoint.pt --output runs/real2sim/eval_real

# 实机档策略回放
micromamba run -n mj python viser_view.py --run-dir runs/real2sim/stage10 \
  --mode policy --real-profile --ignore-overload --device cuda:0
```
