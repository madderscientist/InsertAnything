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

其中 $\bar a_t=c_a\tanh(a_t/c_a)$（默认 $c_a=4.0$），既保持动作成本有界以满足安全罚分下界定理，又在无界动作空间 $|a|>1$ 区间提供平滑恢复梯度以抑制失控大幅跳变，同时不限制送入物理目标变换的策略动作。单步奖励为：

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
| `xy_cost_radius_fraction` | 0.25 | $k_{xy}$，平面近场漏斗参考尺度占半径 $R$ 的比例 |
| `xy_outer_cost_weight` | 0.2 | $w_{\mathrm{out}}$，远场有界径向斜率的凸组合权重（旧检查点缺省 `0.0`） |
| `orientation_cost_per_second` | 1.0 | $w_o$，每秒姿态成本系数 |
| `orientation_cost_reference_deg` | 5.0° | $\alpha_0$，姿态误差参考角度尺度 |
| `force_cost_per_second` | 1.0 | $w_F$，每秒接触力超死区成本系数 |
| `moment_cost_per_second` | 1.0 | $w_M$，每秒接触力矩超死区成本系数 |
| `load_cost_cap` | 1.5 | $c_L$，归一化力与力矩成本的上限（保留超限区二次惩罚梯度） |
| `overload_cost_per_second` | 2.0 | $w_{\mathrm{ov}}$，单步原始力/力矩峰值超限时的额外每秒惩罚系数 |
| `terminate_on_overload` | false | 力/力矩超限时是否立即终止回合（默认不终止，重罚并继续运行） |
| `force_cost_free_n` | 5.0 N | 力成本死区（按三维范数） |
| `moment_cost_free_nm` | 0.1 N·m | 力矩成本死区（按三维范数） |
| `inserted_load_cost` | true | 浅入孔 `3 mm` 内将轴向 $Z$ 力/动作权重与负载死区从 `1` 线性衰减至 `0` |
| `inserted_load_depth_m` | 0.003 | 浅入孔渐变深度（`3 mm`），在此深度内 $w_z$ 由 `1` 渐变至 `0` |
| `action_cost_per_second` | 0.05 | $w_a$，动作幅度成本系数 |
| `action_change_cost_per_second` | 0.05 | $w_{\Delta a}$，相邻步动作变化成本系数 |
| `action_cost_cap` | 4.0 | $c_a$，动作与动作差分成本的 `tanh` 平滑饱和尺度 |
| `completion_bonus` | 20.0 | $B$，达标保持 5 步后的一次性成功奖励 |
| `safety_failure_penalty` | 200.0 | $P_X$，安全超限或逃出工作空间的一次性惩罚 |
| `timeout_penalty` | 10.0 | $P_T$，超时未完成的一次性惩罚 |

## 3. 距离成本

定义插头实际碰撞几何的最低点高度 $z_{\min}$（孔口平面为零，孔内为负），目标深度 $d>0$，工作空间横向半径 $R_{\mathrm{ws}}$，原点平滑尺度 $\epsilon=0.05R$：

$$
u_z=\frac{\max(z_{\min}+d,0)}{d},
\qquad
u_{xy}=\frac{e_{xy}}{k_{xy}R},
\qquad
\phi_{\mathrm{out}}(e_{xy})=\frac{\sqrt{\min(e_{xy},R_{\mathrm{ws}})^2+\epsilon^2}-\epsilon}{\sqrt{R_{\mathrm{ws}}^2+\epsilon^2}-\epsilon}
$$

$$
c_{1/2}=\frac{c}{2},
\qquad
D_{xy}=c_{1/2}\left[(1-w_{\mathrm{out}})\tanh\!\left(\frac{u_{xy}^2}{c_{1/2}}\right)+w_{\mathrm{out}}\phi_{\mathrm{out}}(e_{xy})\right],
\qquad
D_t=D_{xy}+c_{1/2}\tanh\!\left(\frac{u_z}{c_{1/2}}\right)
$$

位置项为 $r_p=-\Delta t\,w_p D_t$。近场平方项在孔口中心导数为零以容许接触盲搜；远场项 $\phi_{\mathrm{out}}$ 在外圈维持恒定径向恢复梯度，消除纯 $\tanh(u_{xy}^2)$ 饱和后的绕行死区，同时通过凸组合严格保持 $D_{xy}\le c_{1/2}$ 与 $D_t\le c$（旧检查点缺省 $w_{\mathrm{out}}=0$）。XY 与 Z 各自独立有界，防止大平面偏差抹平 Z 轴下压奖励梯度。

剩余深度与下节倾角权重均使用最低点高度，允许倾斜时一个角先进入来降低深度成本。多边形插头计算上下端面所有顶点的最低高度，圆柱使用解析最低高度；包含实际抓取偏移。最低点越过孔口高度不额外要求其 XY 位于开口内。XY 成本、策略观测、动作参考点和成功判定仍使用端面中心，不随最低点变化。

## 4. 姿态成本

定义倾角约束进度 $p=\mathrm{clip}(-z_{\min}/0.004,0,1)$，即最低点进入前 4 mm 的进度，与最终目标深度无关。距离成本中的 $u_z$ 仍按最终目标深度计算。有界角度代价为 $h(\alpha)=\frac{(\alpha/\alpha_0)^2}{1+(\alpha/\alpha_0)^2}$：

$$
O_t=p\,h(\theta)+h(\psi)
$$

姿态项为 $r_o=-\Delta t\,w_o O_t$，其中 $\theta$ 为工具倾角，$\psi$ 为按几何旋转对称性折叠后的偏航角误差（正 $n$ 边形周期为 $2\pi/n$，L 形为 $2\pi$，圆形为 $0$）。

所有训练阶段都在最低点位于孔口及上方时将倾角权重置零。进入 1/2/3/4 mm 时权重线性升至 0.25/0.5/0.75/1，之后保持为 1。偏航成本与 35° 倾角越界终止不变。旧运行恢复时忽略保存的 `preinsertion_tilt_weight`；若旧值非零，保留课程等级并清空旧成功率窗口。

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

新训练默认 `inserted_load_cost=true`、`inserted_load_depth_m=0.003`（`3 mm`）。定义浅入孔轴向权重：

$$
w_z = \mathrm{clip}\!\left(1 + \frac{z_{\mathrm{center}}}{d_{\mathrm{load}}},\ 0,\ 1\right)
$$

在孔口及上方（$z_{\mathrm{center}}\ge 0$）时 $w_z=1$；在浅入孔前 `3 mm`（$z_{\mathrm{center}}\in[-0.003,0]\ \mathrm{m}$）内，$w_z$ 从 `1` 线性衰减至 `0`；在 $z_{\mathrm{center}}\le -0.003\ \mathrm{m}$ 后保持 $w_z=0$。孔坐标系下的加权接触力、负载死区与向下动作幅度成本同步按 $w_z$ 过渡：

$$
F_H=R_{HT}F_T,\qquad
\|F_H\|_{w_z}=\sqrt{F_{H,x}^2+F_{H,y}^2+(w_z F_{H,z})^2}
$$
$$
f_t=\mathrm{clip}\!\left(\frac{\|F_H\|_{w_z}-5w_z}{F_{\max}-5w_z},\ 0,\ c_L\right),
\qquad
m_t=\mathrm{clip}\!\left(\frac{\|M\|_2-0.1w_z}{M_{\max}-0.1w_z},\ 0,\ c_L\right)
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
- **过载连续惩罚与成功门控（默认 `terminate_on_overload=false`）**：当当前 `20 ms` 步内的 `1 ms` 子步未滤波峰值力 $>30\text{ N}$ 或峰值力矩 $>2\text{ N}\cdot\text{m}$ 时，**不终止回合**，而是将该步的成功保持计数 `hold` 重置为 `0`（严禁在超限状态下混过成功判定），同时施加二次负载超限梯度（上限 `load_cost_cap=1.5`，即最高 `2.25` 倍二次惩罚）与额外运行惩罚 $-\Delta t\,w_{\mathrm{ov}}$（默认每秒 `-2.0`），迫使策略学习从过载/卡阻状态主动退让卸力。
- **未达标停滞惩罚（`stagnation_cost_per_second`）**：在真实尖端尚未到达目标深度（$z_t > z_{\mathrm{goal}}+0.5\text{ mm}$）时，按窗口 $\tau_{\mathrm{stag}}$ 跟踪指数平滑位置 $\bar p_t$（凹凸表面模式下按进度投影 $p_{\mathrm{prog}}=[x_t,y_t,\min(z_t,0)]$ 统计，防止台面原地上下抖动刷速度），计算等效净推进速率 $v_{\mathrm{eff}}=\|\,p-\bar p_t\,\|/\tau_{\mathrm{stag}}$；当 $v_{\mathrm{eff}} < v_{\mathrm{th}}$ 时施加二次惩罚 $-\Delta t\,w_{\mathrm{stag}}\,\mathrm{clip}(1-v_{\mathrm{eff}}/v_{\mathrm{th}},0,1)^2$，配合二阶段板面上方越障免罚带 `surface_z_free_m` 打破台阶凹坑死顶的局部最优。
- **安全失败（−250）**：逃出工作空间或倾角 $>35^\circ$（若显式开启 `terminate_on_overload=true`，则未滤波峰值力 $>30\text{ N}$ 或峰值力矩 $>2\text{ N}\cdot\text{m}$ 亦触发安全终止）。
- 新训练的横向安全工作空间是 `workspace_xy_radius_m=0.035` 的圆盘，
  容纳初始 X/Y 各独立 ±20 mm 的正方形（角点半径 28.28 mm）。
  旧检查点缺少半径字段时仍用原 `0.025 * scale` m。Z、倾角及完整力/力矩安全阈值未改变。
- **无超时/长时防僵尸脱困模式**：平坦表面保持 `8 s` 超时；第二阶段凹凸表面训练可配置长时防僵尸回收上限（默认 `episode_seconds=35.0, timeout_penalty=60.0`），既保留充足的越障脱困探索窗口，又防止极少数死锁样本长期占据并行环境槽位。
- **防自杀约束**：配置自动校验安全失败罚分（默认 `safety_failure_penalty=250.0`）严格大于包含停滞惩罚与过载惩罚在内的最差无穷步折扣累计运行成本上界 $\frac{C_{\mathrm{step,max}}}{1-\gamma}$ 及单步上界加超时罚分，避免安全终止成为逃避运行成本的廉价选择；验证测试见 [test/test_env_curriculum.py](./test/test_env_curriculum.py)。
