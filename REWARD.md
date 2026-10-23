# 奖励函数说明

本文对应 [insertion/env.py](./insertion/env.py) 的 `completion_cost_terms` 与 `PlantGroup.step`，默认参数定义于 [insertion/config.py](./insertion/config.py)。

## 1. 设计动机

原始 InsertAnything 仓库使用密集正奖励（`xy_align + z_insert + orientation`），并在孔口上方施加接触力平方惩罚和 `1°` 偏航硬门控，极易导致策略陷入“孔口上方无接触悬停刷分”的局部最优。

本项目采用**持续负运行成本 + 分阶段接触负载成本 + 一次性终端成功奖励**：

- 非终止步严格为负分（含固定每步时间成本），彻底消除悬停刷分或往返刷分动机
- 平面误差单独平方、剩余深度保持线性，再相加并饱和；减轻中心附近横向探索的成本，同时保留下探引导
- 入孔前持续引导偏航对齐，不施加倾角成本；入孔后前 4 mm 内随深度线性增强倾角约束
- 深度成本使用插头几何最低点，配合入孔前零倾角成本，**引导斜着先插入一个角**，再随进入深度逐渐回正；不强制必须倾斜，也不为首次入孔额外发奖
- 真正端面中心入孔前提供 `5 N / 0.1 N·m` 负载死区；中心入孔后惩罚孔系横向力与全部力矩，不设死区

## 2. 总公式与默认参数

设运行成本为：

$$
C_t=w_t+w_p D_t+w_o O_t+w_F f_t^2+w_M m_t^2
    +w_a\|\bar a_t\|_2+w_{\Delta a}\|\bar a_t-\bar a_{t-1}\|_2
$$

其中 $\bar a_t=\mathrm{clip}(a_t,-1,1)$。单步奖励为：

$$
r_t=-\Delta t\,C_t+B\,\mathbf{1}_S
    -P_X\,\mathbf{1}_X
    -P_T\,\mathbf{1}_{T\land\neg S\land\neg X}
$$

| 配置字段 | 默认值 | 含义 |
|---|---:|---|
| `time_cost_per_second` | 1.0 | $w_t$，每秒时间成本（50 Hz 下每步 −0.02） |
| `position_cost_per_second` | 2.0 | $w_p$，每秒距离成本系数 |
| `position_cost_cap` | 2.0 | $c$，距离指标饱和上限 |
| `xy_cost_radius_fraction` | 0.25 | $k_{xy}$，平面距离参考尺度占半径 $R$ 的比例 |
| `orientation_cost_per_second` | 1.0 | $w_o$，每秒姿态成本系数 |
| `orientation_cost_reference_deg` | 5.0° | $\alpha_0$，姿态误差参考角度尺度 |
| `preinsertion_tilt_weight` | 0.0 | $w_{\mathrm{tilt},0}$，入孔前倾角回正基础权重 |
| `force_cost_per_second` | 1.0 | $w_F$，每秒接触力超死区成本系数 |
| `moment_cost_per_second` | 1.0 | $w_M$，每秒接触力矩超死区成本系数 |
| `force_cost_free_n` | 5.0 N | 力成本死区（按三维范数） |
| `moment_cost_free_nm` | 0.1 N·m | 力矩成本死区（按三维范数） |
| `inserted_load_cost` | true | 真实端面中心 `z < -4 mm` 后启用无死区横向力/全力矩成本 |
| `inserted_load_depth_m` | 0.004 | 上述成本切换深度，严格小于负深度才启用 |
| `action_cost_per_second` | 0.05 | $w_a$，动作幅度成本系数 |
| `action_change_cost_per_second` | 0.05 | $w_{\Delta a}$，相邻步动作变化成本系数 |
| `completion_bonus` | 20.0 | $B$，达标保持 5 步后的一次性成功奖励 |
| `safety_failure_penalty` | 200.0 | $P_X$，安全超限或逃出工作空间的一次性惩罚 |
| `timeout_penalty` | 10.0 | $P_T$，超时未完成的一次性惩罚 |

## 3. 距离成本

定义插头实际碰撞几何的最低点高度 $z_{\min}$（孔口平面为零，孔内为负），目标深度 $d>0$。非负相对剩余深度 $u_z$ 与平面归一化误差 $u_{xy}$ 为：

$$
u_z=\frac{\max(z_{\min}+d,0)}{d},
\qquad
u_{xy}=\frac{e_{xy}}{k_{xy}R},
\qquad
q_t=u_{xy}^2+u_z,
\qquad
D_t=c\tanh(q_t/c)
$$

位置项为 $r_p=-\Delta t\,w_p D_t$。XY 单独平方使中心附近的小幅横移代价更小，允许接触探索，但也减弱精细居中的引导；Z 保持线性以继续鼓励下探。不是将总和平方，因此没有 XY 与 Z 的交叉项。外层 `tanh` 保留成本上界；这是一项待对照验证的奖励实验，不保证提高成功率。

剩余深度与下节倾角权重均使用最低点高度，允许倾斜时一个角先进入来降低深度成本。多边形插头计算上下端面所有顶点的最低高度，圆柱使用解析最低高度；包含实际抓取偏移。最低点越过孔口高度不额外要求其 XY 位于开口内。XY 成本、策略观测、动作参考点和成功判定仍使用端面中心，不随最低点变化。

## 4. 姿态成本

定义倾角约束进度 $p=\mathrm{clip}(-z_{\min}/0.004,0,1)$，即最低点进入前 4 mm 的进度，与最终目标深度无关。距离成本中的 $u_z$ 仍按最终目标深度计算。有界角度代价为 $h(\alpha)=\frac{(\alpha/\alpha_0)^2}{1+(\alpha/\alpha_0)^2}$：

$$
w_{\mathrm{tilt}}(p)=w_{\mathrm{tilt},0}+(1-w_{\mathrm{tilt},0})p,
\qquad
O_t=w_{\mathrm{tilt}}(p)h(\theta)+h(\psi)
$$

姿态项为 $r_o=-\Delta t\,w_o O_t$，其中 $\theta$ 为工具倾角，$\psi$ 为按几何旋转对称性折叠后的偏航角误差（正 $n$ 边形周期为 $2\pi/n$，L 形为 $2\pi$，圆形为 $0$）。

默认 $w_{\mathrm{tilt},0}=0$：最低点在孔口及上方时不惩罚倾角，进入 1/2/3/4 mm 时权重分别为 0.25/0.5/0.75/1，之后保持为 1。偏航成本与倾角安全上限不变。旧运行恢复时沿用保存的基础权重，但最低点及 4 mm 线性进度使用当前代码；要取消入孔前倾角成本，需显式将旧运行的基础权重改为 0。

### 为什么选择 4 mm

允许入孔前倾斜探索、一个角先进入；随着进入长度增加，狭小间隙下的孔壁会明显限制倾斜，需要较早强调回正。原先按最终目标深度平方归一化，目标为 20 mm 时进入 4 mm 的权重仅为 0.04，且同一进入深度的权重随目标变化，不符合这一意图。

因此独立采用实际 4 mm 作为满权重尺度，不随目标深度或几何缩放变化。线性渐变既保留孔口附近的倾斜空间，又比平方渐变更早强调回正，且不会在过孔口时跳变到满权重。

4 mm 是启发式设计值，尚未实验确认最优，不是姿态硬锁定。实际转动余量由形状、间隙和姿态决定；最低点进入 4 mm 不等于整个端面进入 4 mm。碰撞仿真负责物理约束，奖励只提供姿态引导。

## 5. 负载与动作成本

当真实物理端面中心 $z_{\mathrm{center}}\ge-0.004\ \mathrm{m}$（包括恰好 -4 mm），或
`inserted_load_cost=false` 时，保留原有三维范数及死区：

$$
f_t=\mathrm{clip}\left(\frac{\|F\|_2-5}{30-5},0,1\right),
\qquad
m_t=\mathrm{clip}\left(\frac{\|M\|_2-0.1}{2-0.1},0,1\right)
$$

新训练默认 `inserted_load_cost=true`。当且仅当真实物理端面中心
$z_{\mathrm{center}}<-0.004\ \mathrm{m}$ 时，改用：

$$
F_H=R_{HT}F_T,\qquad
f_t=\mathrm{clip}\left(\frac{\sqrt{F_{H,x}^2+F_{H,y}^2}}{F_{\max}},0,1\right),
\qquad
m_t=\mathrm{clip}\left(\frac{\sqrt{M_x^2+M_y^2+M_z^2}}{M_{\max}},0,1\right)
$$

使用当前滤波后、名义尖端处的力旋量（与旧奖励相同的数据源），不叠加 Actor
额外噪声或固定力零偏。力先从当前工具坐标转到固定孔/世界坐标，再仅取 X/Y；
工具倾斜时不能直接丢弃工具系 Fz。全部三个力矩分量都保留，包括 Mz。
不设力或力矩死区；归一化分母为已有安全阈值 `max_force/max_moment`。
轴向 Fz 只从入孔后的**奖励力项**移除，原始完整三维力/力矩逐子步峰值安全检查不变。
切换条件不是最低点、带噪位置、延迟观测或目标深度；最低点先入孔而中心尚在上方时，
仍使用原死区奖励。其他成本、平方形式、权重和逐步上界均不变。

$$
r_F=-\Delta t\,w_F f_t^2,
\qquad
r_M=-\Delta t\,w_M m_t^2,
\qquad
r_a=-\Delta t\,w_a\|\bar a_t\|_2,
\qquad
r_{\Delta a}=-\Delta t\,w_{\Delta a}\|\bar a_t-\bar a_{t-1}\|_2
$$

载荷成本固定使用平方形式，不提供形状选项。
兼容读取平方版检查点中已废弃的 `load_cost_power=2` 字段；
线性实验检查点会明确拒绝加载，需先显式迁移奖励配置。
旧检查点缺少 `inserted_load_cost` 时按 `false` 恢复，不会自动改变历史奖励。
旧检查点缺少 `inserted_load_depth_m` 时按 0 恢复原切换深度；当前续训显式迁移到 0.004。
显式迁移时应清空成功率窗口和自适应失败率历史，保留网络、优化器及课程等级。

## 6. 终止条件与防自杀边界

- **成功终止（+20）**：真实尖端连续 **5 个策略步**（100 ms）满足 $z_t\le-d+0.0002\text{ m}$，
  且真实端面中心 XY 每步位于孔口内部或边界，未触发安全越界。新训练默认
  `success_opening_required=true`，防止板外下探计为成功；旧检查点缺字段按 false 保留历史规则。
  凹 L 形使用射线奇偶包含判定，圆孔使用解析半径。不要求额外精细居中、倾角或偏航达标，
  不恢复旧 `0.4*clearance` 门槛；物理碰撞约束可行插入姿态。

`success_alignment_required` 新训练默认为 `false`。旧检查点缺少该字段时按 `true`
恢复原有 `0.4 × clearance` 的 XY/倾角/偏航成功门槛，确保历史评估兼容。
显式切换成功判定或保持步数后，恢复训练保留课程等级但清空成功率窗口；
迁移运行配置时还应清空自适应采样 EMA，避免混用不同判定的失败率。
- **安全失败（−200）**：未滤波峰值力 $>30\text{ N}$、峰值力矩 $>2\text{ N}\cdot\text{m}$、逃出工作空间或倾角 $>35^\circ$。
- 新训练的横向安全工作空间是 `workspace_xy_radius_m=0.035` 的圆盘，
  容纳初始 X/Y 各独立 ±20 mm 的正方形（角点半径 28.28 mm）。
  旧检查点缺少半径字段时仍用原 `0.025 * scale` m。Z、倾角及完整力/力矩安全阈值未改变。
- **防自杀约束**：默认单步运行成本理论上界 $C_{\mathrm{step,max}}\approx0.18735$，折扣因子 $\gamma=0.9985$ 下无穷步累计运行成本上界为 $\frac{C_{\mathrm{step,max}}}{1-\gamma}\approx124.90 < P_X=200$。配置同时检查安全罚分大于单步上界加超时罚分，以避免安全终止成为逃避运行成本的廉价选择；这不是策略绝不会违规的保证。验证测试见 [test/test_env_curriculum.py](./test/test_env_curriculum.py)。
