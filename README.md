# 基于 mjlab 的通用六自由度多几何力控插入策略

本项目使用 `mjlab` 批量 MuJoCo GPU 仿真与双层 GRU 循环 PPO，训练**单一策略网络**同时完成三角形、方形、正五边形、正六边形和 L 形的高精度力控插孔，并在未见过的圆形孔上检验零样本泛化能力。

奖励函数、运行成本上界与终止判据详见 [REWARD.md](./REWARD.md)。

位置与力觉的消融数据、dx/dy 合成偏差分布及当前瓶颈总结见
[位置与力觉误差分析](./OBSERVATION_ERROR_ANALYSIS.md)。
当前证据更支持末端参考点偏差和力/力矩固定零偏是主要观测瓶颈，
而非当前幅度的逐步白噪声；XY 偏差与白噪声尚未完成严格独立拆分。

碰撞几何：正方形使用一个旋转 box 插头和四个不重叠的 box 孔壁，
保持原有几何边界、深度和间隙，但接触点及力响应可能与 mesh 版本不同。
L 形也使用 box 组合，其他多边形仍使用凸 mesh。

GPU 仿真容量使用 `nconmax=64`、`njmax=256`，初始化及课程切换重建时一致。
接触池由同组环境共享，总容量为环境数乘以 64；约束容量为每环境 256。

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

核心依赖版本（见 [requirements.txt](./requirements.txt)）：
- `torch>=2.14.1`
- `mujoco>=3.11.0`, `mujoco-warp>=3.11.0`, `warp-lang>=1.18.0`
- `mjlab>=1.6.0`, `rsl-rl-lib>=5.4.2`, `tensordict>=0.14.3`
- `numpy>=2.4.6`, `trimesh>=5.1.1`, `viser>=1.1.1`, `pillow>=12.3.0`, `tensorboard>=2.21.0`

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
# 力/力矩为滤波传感器值，含惯性负载、标定和坐标变换误差；
# 不含策略观测额外添加的随机力噪声和固定力偏差，也不是安全判定的原始子步峰值。
# 三维场景插头旁：红色为力，蓝色为力矩轴向量（右手定则），偏移原点不代表实际作用点。
# 箭头随工具旋转，无数值标签；长度严格线性：5 N / 0.1 N·m 对应 30 mm，
# 不做平方根放大或长度截断；两种载荷的箭头长度不能直接比较。
micromamba run -n mj python viser_view.py \
  --run-dir runs/continuous_dual --host 127.0.0.1 --port 8080

# Viser 独立缓冲回放最新策略（按 1x 仿真时间 50 Hz 播放）
micromamba run -n mj python viser_view.py \
  --run-dir runs/continuous_dual --mode policy --device cuda:1 \
  --host 127.0.0.1 --port 8080

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

# 已有训练在线扩容：当前轮次保存后由守护进程恢复
micromamba run -r ~/micromamba -n mj python train.py \
  --resize-envs --envs-per-shape 512 --output runs/continuous_single

# 优雅停止
touch runs/continuous_single/STOP

# 停止后恢复（使用保存的环境数和课程配置）
rm -f runs/continuous_single/STOP
CUDA_VISIBLE_DEVICES=0 micromamba run -r ~/micromamba -n mj python -u train.py \
  --resume --output runs/continuous_single --device cuda:0
```

此配置共 2560 个环境，每轮采样 655360 条 transition。接触状态和回合长度会影响
吞吐与峰值显存。扩容保留权重、归一化器、优化器和课程，但重置进行中的回合及
GRU 隐藏状态，并改变 PPO 采样批量；恢复训练使用已保存的配置。

持续训练在每轮检查点保存后释放 PyTorch 未使用的 CUDA 分配缓存，避免变长 GRU
训练工作区长期占据显存、拖慢下一轮仿真；分布式各 rank 分别释放本卡缓存。
这不释放存活的模型、优化器或 CUDA Graph 缓冲区，也不能降低 PPO 更新本身的峰值需求。
`heartbeat.json` 的 `cuda_memory` 记录各设备的存活分配、释放前后保留量及剩余显存。

守护式持续训练每轮更新 `checkpoint.pt`，并保留上一轮 `previous.pt`。
控制台的环境指标按完整名称排序，同前缀（如 `curriculum/success/`、
`curriculum/episodes/`、`reward/`）相邻且顺序固定；不改变指标汇总或 TensorBoard 数值。
另外，每到 500 的正整数倍 PPO 轮次，独立归档完整检查点到
`checkpoints/checkpoint_<八位轮次>.pt`；这里的步数不是物理步或 transition。
归档包含网络、优化器、归一化器、环境配置和课程状态，不自动清理。
回退后再次经过同名轮次时保留已有归档，不覆盖它。单卡及分布式 rank 0 共用保存路径。

持续训练每轮保存 `termination_counts.json`，按形状统计当前 worker 启动以来的
完成回合数、成功、仅力过载、仅力矩过载、两者过载、工作空间/倾角越界及超时。
按现有终止优先级互斥计数；这不是课程的最近 1000 回合窗口，也不证明失败的物理根因。
worker 重启后重新计数；分布式模式目前只报告 rank 0 本地样本。
同一文件的 `overload_stages` 将过载按真实尖端高度、回合时间、倾角分别分桶；
三组各自互斥。位置/倾角为 reset 前的 20 ms 终止步末状态，不是原始 1 ms
过载子步瞬间或首次接触位置，不能仅据此断言初始碰撞或孔内卡死的物理根因。
启用物理 CUDA Graph 时，终止原因及阶段计数也使用独立 CUDA Graph 批量累加，
减少逐步统计的内核启动开销；不改变计数口径，首次捕获不重复计数。

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

自适应采样只支持带成功率课程的**守护式持续训练**。总环境数不变，
每 30 个 PPO 轮次最多检查一次；当前等级每种形状的 1000 回合窗口填满后，
将失败率用 EMA（新值权重 0.2）平滑，以 **30% 均匀 + 70% 失败率比例**分配环境。
课程升级后重置 EMA 并等待下一检查周期。取消单次变化幅度和每形状上下限；
均匀项提供基准的 30% 保底，按 GPU 数取整（误差不超过一个分片），每形状每卡至少一个环境。
达到零失败率时回归均匀目标；仍按真实回合统计课程，不对成功率加权。
课程窗口按完整环境批次更新：每个并行环境每批只贡献一个当前等级完成回合，
等待所有环境（包括慢速超时）收齐后才入账并检查晋级，避免重启后快速成功抢先
抬高成功率。重启保留历史窗口，但丢弃尚未完成的统计批次；训练 rollout 不受影响。

布局变化通过当前轮次保存后重建环境与 CUDA Graph 生效；保留网络、优化器、
归一化器及课程，重置进行中的回合和 GRU 状态，不改变 PPO 总采样批量。
多卡各 rank 使用同一全局布局并等分各形状。控制器状态保存在运行配置及检查点中，
调整记录写入 `sampling_<iteration>.json`，心跳包含 `shape_environment_counts`。

### 初始位姿范围

初始位置指**真实尖端**相对孔口的位置。新训练默认 X/Y 各独立均匀采样于 ±20 mm，
Z 在孔口上方 4–10 mm；姿态旋转向量 X/Y 分量各 ±3°，Z 分量 ±15°。
X/Y 位置和三个旋转分量随 `difficulty` 缩放，Z 高度不缩放。
旋转向量分量不是独立欧拉角，也不是总转角上限。

这不是半径 20 mm 的圆盘：初始正方形角点半径为 28.28 mm。新训练显式使用
`workspace_xy_radius_m=0.035`（`--workspace-xy-radius-m 0.035`）的横向安全圆盘，
不改变 Z、倾角或力/力矩安全限制。配置会验证完整初始正方形位于安全圆盘内。
半径以米表示，不再隐式随 `scale` 缩放；旧检查点缺失该字段仍按原 `0.025 * scale` m 恢复。
旧检查点的 `initial_xy` 原样保留；该字段也缺失时使用历史默认 0.003 m。
**冻结的 iteration 19610 检查点实际保存 `initial_xy=0.01`，即每轴 ±10 mm，
不是历史代码默认的 ±3 mm，也不是新训练默认的 ±20 mm。**

以下显式参数对应此前扩大后的范围（已有训练恢复时使用保存配置）：

```bash
--initial-xy 0.01 --initial-z-min 0.001 --initial-z-max 0.01 \
--initial-tilt-deg 20 --initial-yaw-deg 20
```

这对应 X/Y 各 ±10 mm、Z 为 1–10 mm，旋转向量三个分量各 ±20°；
最大水平偏离约 14.14 mm，总转角最大约 34.64°（`difficulty=1`）。
扩大范围可能增加孔口初始接触和安全终止，不能据此保证成功率不降。
改变这些范围后恢复训练会保留课程等级、清空旧成功率窗口；
旧检查点缺失的 Z 参数按原始 4–10 mm 解释，保持兼容。

`initial_plane_margin` 大于零时启用姿态与高度联动：按旋转后的实际插头外形
（含 L 形两个部分的外轮廓，圆柱使用解析支撑半径）计算最低点，在满足整个插头
高于孔口平面指定间隙的合法 Z 区间内均匀采样。真实夹持偏移已通过真实尖端定位抵消。
当前训练使用 0.001 m，Z 上限仍为 0.01 m，X/Y 和旋转范围不变。
不可行姿态最多采样 16 次，仍失败则显式报错，不静默越过高度上限。
此规则偏保守：不允许任何插头部分低于孔口平面，即使该位置本可位于孔洞内。
默认零保持旧采样行为；修改该值后恢复会清空成功率窗口、保留课程等级。
仅在 reset 时进行 GPU 批量几何计算，不在逐物理步增加碰撞搜索。

新训练 `plate_outer_radius_m=0.080`（`--plate-outer-radius-m`）将所有形状实体
孔板外轮廓扩大到至少 **80 mm 内切半径**，孔口轮廓、间隙、材料摩擦与求解器不变。
正多边形沿原径向分块延伸，方形/L 形使用无体积重叠的盒子拼接，孔洞不填充。
覆盖验证下界是工作空间半径 + `sqrt(2)*10 mm*scale` 端面包络半径
+ `35 mm*scale*sin(35°)` 倾斜侧向投影 + `1 mm*scale` 余量，
默认约 70.22 mm，选择 80 mm 留出额外裕量。相对真实尖端，抓取平移已抵消；
无界高斯误差与超过安全倾角的异常状态不承诺任意有限板面都覆盖。
旧检查点缺字段使用 0，保留原约 20 mm 外轮廓。

新训练 `success_opening_required=true`（`--no-success-opening-required` 可关闭）：
每个成功保持步都要求**真实物理端面中心 XY 在孔口内部或边界**。
多边形采用支持凹 L 形的射线奇偶判定，圆孔采用解析半径；不使用带噪/延迟位置，
不恢复旧 `0.4*clearance` 精度门槛，也不要求整个插头投影完全包含，实际适配由碰撞约束负责。
因此板外下探不会计为成功。旧检查点缺字段按 false 恢复。改变板外形或成功规则后
必须重建环境/CUDA Graph 并清空成功窗口及自适应统计。

## 4. 架构与方法设计

### 4.1 观测、动作与网络结构
孔坐标系 `H` 固定于孔口中心；工具坐标系 `T` 原点位于**名义插头尖端**；虚拟六轴腕部传感器坐标系 `S` 位于质心上方。单位统一采用国际单位制（`m, s, rad, N, N·m`），四元数顺序为 `wxyz`。

- **Actor 输入（13 维）**：
  1. `H` 系中目标位置减去当前估计尖端位置（3 维）
  2. 规范化相对四元数 `inverse(current) * target`（4 维，消除 `q/-q` 双重覆盖）
  3. `T` 系中关于名义尖端的重力补偿后滤波力旋量 `wrench`（6 维）
- **仅位置观测延迟**：新训练 `position_observation_delay=true`。每个环境每次 reset
  独立均匀采样 **0 或 1 个策略帧（0/20 ms）**，回合内固定。只延迟 Actor 的
  `dx/dy/dz`：目标偏差、名义尖端标定与该历史帧的位置白噪声一并保留。
  四元数与六轴力旋量仍取当前帧；Critic 始终使用当前干净真实状态，维数不变。
  reset 用新回合初始带噪位置误差填满历史，部分 reset 不影响其他槽位。
  历史按策略步 `length` 更新一次，重复读取、CUDA 预热/捕获不推进时间或队列；
  确定性 `observations_from_noise` 不修改历史。
  `--no-position-observation-delay` 恢复原即时位置观测；旧检查点缺字段自动按 false 加载。
- **入孔后负载成本**：新训练 `inserted_load_cost=true`；真实端面中心 `z<-4 mm`
  后仅惩罚固定孔系 Fx/Fy 与全部三个力矩，无死区，沿用已有归一化安全阈值、
  有界平方成本和权重。入孔前维持 5 N/0.1 N·m 全范数死区；
  完整原始三维力/力矩安全检查不变。详见 [REWARD.md](./REWARD.md)。
  `--no-inserted-load-cost` 或旧检查点缺字段时沿用原奖励。
  `--resume` 不会用命令行开关覆盖保存配置；启用新机制需显式迁移环境配置并清空
  成功率窗口/自适应统计，不能把旧模型评估默认为新机制。
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

迁移保留均值策略、GRU、归一化、价值网络、训练步数及未改变参数的优化器状态；
新增 std 输出权重初始化为零，偏置为旧各维有效 `log_std`，因此初始动作分布不变。
输出层优化器矩保留原均值部分，新增 std 部分从零开始；原全局 std 参数及其优化器状态移除。
标准差上下限沿用源运行配置，课程归零并清空成功窗口、自适应采样历史，环境数恢复各形状均分。
这不是从随机权重重训；新分布可能过早收缩探索，收益仍需通过成功率、超限率和确定性评估验证。

可选 `--initial-std 0.1 --noise-level 1`：保留均值策略，将六维 std 初始化为 0.1，
从目标间隙、10% 噪声幅度开始（仍按幅度线性递增，不采用平方根缩放）。
省略这些参数则继承旧有效 std 并从完整课程 0 开始。std 输出权重为零不妨碍该层
权重和偏置获得梯度；初次反传中共享网络仍由均值分支训练，之后 std 分支也可向共享层传播梯度。

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

当前续训目录为 `runs/continuous_position_delay20mm/`，继承第 19704 轮权重及 Adam
状态，使用 0/20 ms 位置延迟、X/Y 各 ±20 mm、35 mm 横向边界及 80 mm 最小内切
半径孔板。成功要求真实端面中心在孔口内；中心 `z < -4 mm` 后切换为无死区
横向力/全力矩成本。满噪声、满难度、0.1 mm 间隙固定，不推进课程。

默认训练不包含失败池、随机意图或恢复课程。旧失败池和随机意图尝试分别归档于
`feat/failure-replay`、`feat/coherent-exploration`，未取得预期脱困效果，已暂停。
独立恢复试验也未完成收益验证，协议见 [RECOVERY_EXPERIMENT.md](./RECOVERY_EXPERIMENT.md)；
这些试验不会由普通训练入口自动启用。

普通策略的观测误差诊断入口为 [evaluate_noise_ablation.py](./evaluate_noise_ablation.py)，
支持逐项消融、夹持误差通道分解及精确 XY/力觉联合对照。结果与限制见
[OBSERVATION_ERROR_ANALYSIS.md](./OBSERVATION_ERROR_ANALYSIS.md)。
联合 oracle 对照的 632/640 不代表部署能力或当前训练环境的成功率。

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
