# 奖励函数说明

代码位置：[insertion/env.py](./insertion/env.py)。总装函数 `completion_cost_terms`，各项由 `position_distance`、`orientation_cost`、`load_costs`、`action_costs`、`stagnation_level` 计算，终止判定在 `PlantGroup.step_outcome` 中。
默认参数见 [insertion/config.py](./insertion/config.py)；运行时以各 run 目录下的 `config.json` 为准。

## 1. 设计原则

- **只罚不奖、终局一次性奖励**：每个非终止步的奖励严格为负，不存在悬停或往返刷分；成功时一次性给 `+20`。
- **各项有界**：距离、姿态、负载和动作项都做了饱和，单步成本有理论上界。安全罚分必须大于这个上界的无穷步折扣和，策略因此无法靠触发安全终止来逃避成本（第 5 节）。
- **按阶段改变约束**：入孔前，倾角不罚、力有死区，鼓励用接触去找孔；入孔后，逐步加大倾角约束，撤掉轴向力成本和死区。
- **空中要稳，接触可以晃**：动作变化成本在悬空时更重。

## 2. 单步奖励

$$
r_t=-\Delta t\sum_k w_k\,c_k(t)\;+\;B\,\mathbf 1_{\text{success}}\;-\;P_X\,\mathbf 1_{\text{safety}}\;-\;P_T\,\mathbf 1_{\text{timeout}}
$$

其中 $\Delta t=20$ ms。下表中"日志键"即训练日志中的 `reward/<键>`：

| 日志键 | 成本 $c_k$ | 权重字段（第一阶段默认值） | 作用 |
|---|---|---|---|
| `time_penalty` | 1 | `time_cost_per_second`（1.0） | 每步固定 −0.02，促使尽快完成 |
| `position_penalty` | $D$，上界为 `position_cost_cap` | `position_cost_per_second`（2.0） | 引导对准孔口并下探到目标深度（3.1） |
| `orientation_penalty` | $O\in[0,2)$ | `orientation_cost_per_second`（1.0） | 偏航对齐；入孔后倾角回正（3.2） |
| `force_penalty` | $f^2$，$f\le c_L$ | `force_cost_per_second`（1.0） | 超过死区的接触力（3.3） |
| `moment_penalty` | $m^2$，$m\le c_L$ | `moment_cost_per_second`（1.0） | 超过死区的接触力矩（3.3） |
| `action_penalty` | $\lVert\bar a_t\rVert$ | `action_cost_per_second`（0.05） | 动作幅度（3.4） |
| `action_change_penalty` | $\lVert\bar a_t-\bar a_{t-1}\rVert$ | 悬空 `free_air_action_change_cost_per_second`（0.5）；接触 `action_change_cost_per_second`（0.1） | 动作平滑（3.4） |
| `stagnation_penalty` | $s\in[0,1]$ | `stagnation_cost_per_second`（2.0，第二阶段 2.5） | 未达目标深度时推进太慢（3.5） |
| `completion_bonus` | — | `completion_bonus`（20） | 成功终止（第 4 节） |
| `safety_failure_penalty` | — | `safety_failure_penalty`（800） | 安全失败终止（第 4 节） |
| `timeout_penalty` | — | `timeout_penalty`（10，第二阶段 60） | 超时且未成功、未安全失败 |

## 3. 运行成本

### 3.1 位置 $D=D_{xy}+D_z$

记 $c_{1/2}=\tfrac12$`position_cost_cap`，$R$ 为形状半径，$e_{xy}$ 为尖端端面中心到孔轴的横向距离，$z_{\min}$ 为插头碰撞几何最低点相对孔口的高度，$d$ 为本回合目标深度：

$$
D_{xy}=c_{1/2}\Big[(1-w_{\text{out}})\tanh\!\big(u_{xy}^2/c_{1/2}\big)+w_{\text{out}}\,\phi_{\text{out}}(e_{xy})\Big],\qquad u_{xy}=\frac{e_{xy}}{k_{xy}R}
$$

$$
D_z=c_{1/2}\tanh\!\big(u_z/c_{1/2}\big),\qquad u_z=\frac{\max(z_{\min}+d,\,0)}{d}
$$

- **近场项** $\tanh(u_{xy}^2)$：在孔轴处导数为零，允许在孔口附近用接触盲搜。参考尺度为 $k_{xy}R$，其中 $k_{xy}$ 即 `xy_cost_radius_fraction`=0.25。
- **远场项** $\phi_{\text{out}}$：在 `workspace_xy_radius_m` 以内是平滑的径向斜率，弥补近场项饱和后的零梯度，权重为 `xy_outer_cost_weight`=0.2。两项按凸组合相加，所以 $D_{xy}\le c_{1/2}$ 始终成立。
- **深度用最低点**：深度项按最低点计算，策略可以斜着先把一个角插进去来降低成本。XY 成本、观测和成功判定仍以端面中心为准。
- **第二阶段越障免罚带**：`surface_z_free_m`>0 时，最低点在板面以上、这个高度以内的抬升不增加 $D_z$，策略可以抬起来越过台阶。

### 3.2 姿态 $O$

$$
O=p\,h(\theta)+h(\psi),\qquad h(\alpha)=\frac{(\alpha/\alpha_0)^2}{1+(\alpha/\alpha_0)^2},\qquad p=\mathrm{clip}\!\left(\frac{-z_{\min}}{4\,\text{mm}},0,1\right)
$$

- **偏航 $\psi$**：按形状的旋转对称性折叠（正 $n$ 边形周期 $2\pi/n$，L 形 $2\pi$，圆形恒为 0），全程都罚。
- **倾角 $\theta$**：最低点还在孔口及以上时不罚；最低点进入孔口后的前 4 mm 内，权重从 0 线性升到 1。4 mm 是启发式取值，与目标深度无关。参考角 $\alpha_0$ 为 `orientation_cost_reference_deg`=5°。

### 3.3 负载 $f$、$m$

使用滤波后、换算到名义尖端的力旋量，不加 Actor 观测里的噪声和零偏。

$$
f=\mathrm{clip}\!\left(\frac{\lVert F\rVert_{w_z}-F_0w_z}{F_{\max}-F_0w_z},0,c_L\right),\qquad
m=\mathrm{clip}\!\left(\frac{\lVert M\rVert-M_0w_z}{M_{\max}-M_0w_z},0,c_L\right)
$$

$$
w_z=\mathrm{clip}\!\left(1+\frac{z_{\text{center}}}{d_{\text{load}}},0,1\right),\qquad
\lVert F\rVert_{w_z}=\sqrt{F_{H,x}^2+F_{H,y}^2+(w_zF_{H,z})^2}
$$

- **死区**：$F_0$=`force_cost_free_n`=5 N，$M_0$=`moment_cost_free_nm`=0.1 N·m。
- **归一化**：$F_{\max}$=30 N、$M_{\max}$=2 N·m 是安全阈值；$c_L$=`load_cost_cap`=1.5 使超限区仍有梯度，二次代价最高 2.25 倍。
- **入孔过渡**（`inserted_load_cost`=true）：端面中心 $z_{\text{center}}$ 从孔口进入到 $d_{\text{load}}$=`inserted_load_depth_m`=3 mm 的过程中，$w_z$ 从 1 线性降到 0，孔系轴向力 $F_{H,z}$ 和两个死区一起撤掉。入孔后只罚横向力和全部力矩。
- **不影响安全检查**：这只是奖励的变化；逐子步的完整力/力矩峰值安全检查不受影响。

### 3.4 动作 $\bar a$

$\bar a_t=c_a\tanh(a_t/c_a)$，其中 $c_a$=`action_cost_cap`=12。软饱和只用于计算成本，保证单步成本有界，同时对大动作仍有梯度；送进控制器的动作不受影响。

- **幅度**：$\lVert\bar a_t\rVert$，六个分量统一计算，不区分方向。
- **变化**：$w_{\Delta a}\lVert\bar a_t-\bar a_{t-1}\rVert$。上一个控制周期内真实约束力大于 0.05 N 即视为接触，$w_{\Delta a}$ 取 0.1；否则视为悬空，取 0.5。这里用的是特权的真实接触信息，不是带噪声的观测。

### 3.5 停滞 $s$

对尖端位置做时间窗为 $\tau$=`stagnation_window_seconds` 的指数平滑，得到 $\bar p$；净推进速率 $v=\lVert p-\bar p\rVert/\tau$。

$$
s=\underbrace{\min(1,t/\tau)}_{\text{回合预热}}\cdot\mathbf 1[z_{\text{tip}}>z_{\text{goal}}+0.5\,\text{mm}]\cdot\mathrm{clip}(1-v/v_{\text{th}},0,1)^2
$$

凹凸表面（`rough_surface`）下只统计 $[x,y,\min(z,0)]$，板面以上的上下弹跳不算推进。第一阶段取权重 2.0、$v_{\text{th}}$=6 mm/s、$\tau$=0.4 s；第二阶段和 real2sim 取 2.5、2.5 mm/s、0.7 s。

## 4. 终止条件

| 结果 | 条件 | 一次性奖励 |
|---|---|---|
| 成功 | 尖端连续 `success_hold_steps`=5 步（100 ms）满足 $z_{\text{tip}}\le z_{\text{goal}}+0.2$ mm，且端面中心 XY 在孔口轮廓内（`success_opening_required`） | $+B$=20 |
| 安全失败 | 逃出工作空间（横向超过 `workspace_xy_radius_m`、高度超过 `workspace_z_max_m`、低于孔底）、倾角大于 35°、单环境物理发散（NaN 或接触缓冲溢出） | $-P_X$=800 |
| 超时 | 达到 `episode_seconds`，且既未成功也未安全失败 | $-P_T$ |

- **过载**：默认 `terminate_on_overload`=false。某一步内 1 ms 子步的未滤波峰值力超过 30 N 或力矩超过 2 N·m 时，回合不终止，只是成功保持计数清零，并由 3.3 的二次负载成本提供梯度。设为 true 时，过载直接按安全失败终止，viser 回放时默认这样处理。
- **优先级**：安全失败优先于成功；超时只在既未成功也未安全失败时计入。

## 5. 防自杀约束

`InsertionCfg` 在构造时校验：

$$
P_X>\max\!\left(\frac{C_{\text{step,max}}}{1-\gamma},\;C_{\text{step,max}}+P_T\right)
$$

$C_{\text{step,max}}$ 是第 3 节各项上界之和（`completion_step_cost_bound`），动作变化项取悬空和接触两个系数中较大者。提高任何成本权重时，都要同时检查安全罚分是否还满足这一条件。

## 6. 各阶段取值

只列出与第一阶段默认值不同的字段：

| 字段 | 第一阶段（默认） | 第二阶段 `train_rough_surface.py` | real2sim |
|---|---|---|---|
| `action_change_cost_per_second`（接触 / 悬空） | 0.1 / 0.5 | 0.05 / 不区分 | 0.2 / 0.6 |
| `stagnation_cost_per_second` / 阈值 / 窗口 | 2.0 / 6 mm/s / 0.4 s | 2.5 / 2.5 mm/s / 0.7 s | 同第二阶段 |
| `surface_z_free_m` | 0 | 0.006 | 0.006 |
| `episode_seconds` / `timeout_penalty` | 8 s / 10 | 30 s / 60 | 30 s / 60 |
| `safety_failure_penalty` | 800 | 600 | 800 |

第二阶段那一列取自产出 961 的那次训练。之后从第一阶段新训练进入第二阶段时，会沿用新的平滑默认值。

旧检查点缺少的字段，由 `InsertionCfg.from_checkpoint` 填入历史取值；已退役的字段在加载时直接忽略。
