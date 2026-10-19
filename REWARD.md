# 奖励函数说明

本文对应 [insertion/env.py](./insertion/env.py) 的 `completion_cost_terms` 与 `PlantGroup.step`，默认参数定义于 [insertion/config.py](./insertion/config.py)。

## 1. 设计动机

原始 InsertAnything 仓库使用密集正奖励（`xy_align + z_insert + orientation`），并在孔口上方施加接触力平方惩罚和 `1°` 偏航硬门控，极易导致策略陷入“孔口上方无接触悬停刷分”的局部最优。

本项目采用**持续负运行成本 + 接触负载死区 + 一次性终端成功奖励**：

- 非终止步严格为负分（含固定每步时间成本），彻底消除悬停刷分或往返刷分动机
- 平面误差与深度误差解耦相加并缩放，确保孔口表面（$u_z=1$）亚毫米级的平面微调收益显著高于动作惩罚
- 入孔前持续引导偏航对齐并提供温和的倾角回正梯度，入孔后随深度平方增强倾角约束
- 提供 `5 N / 0.1 N·m` 负载死区，允许安全范围内的接触摸索

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
| `preinsertion_tilt_weight` | 0.25 | $w_{\mathrm{tilt},0}$，入孔前倾角回正基础权重 |
| `force_cost_per_second` | 1.0 | $w_F$，每秒接触力超死区成本系数 |
| `moment_cost_per_second` | 1.0 | $w_M$，每秒接触力矩超死区成本系数 |
| `force_cost_free_n` | 5.0 N | 力成本死区（按三维范数） |
| `moment_cost_free_nm` | 0.1 N·m | 力矩成本死区（按三维范数） |
| `action_cost_per_second` | 0.05 | $w_a$，动作幅度成本系数 |
| `action_change_cost_per_second` | 0.05 | $w_{\Delta a}$，相邻步动作变化成本系数 |
| `completion_bonus` | 20.0 | $B$，达标保持 5 步后的一次性成功奖励 |
| `safety_failure_penalty` | 200.0 | $P_X$，安全超限或逃出工作空间的一次性惩罚 |
| `timeout_penalty` | 10.0 | $P_T$，超时未完成的一次性惩罚 |

## 3. 距离成本

定义非负相对剩余深度 $u_z$ 与平面归一化误差 $u_{xy}$：

$$
u_z=\frac{\max(e_z,0)}{d},
\qquad
u_{xy}=\frac{e_{xy}}{k_{xy}R},
\qquad
q_t=u_{xy}+u_z,
\qquad
D_t=c\tanh(q_t/c)
$$

位置项为 $r_p=-\Delta t\,w_p D_t$。解耦相加避免了 $\sqrt{(e_{xy}/R)^2+u_z^2}$ 在孔口表面（$u_z=1, e_{xy}\to0$）处因平方根耦合导致的平面梯度消失。

## 4. 姿态成本

定义插入进度 $p=\mathrm{clip}(1-u_z,0,1)$ 及有界角度代价 $h(\alpha)=\frac{(\alpha/\alpha_0)^2}{1+(\alpha/\alpha_0)^2}$：

$$
w_{\mathrm{tilt}}(p)=w_{\mathrm{tilt},0}+(1-w_{\mathrm{tilt},0})p^2,
\qquad
O_t=w_{\mathrm{tilt}}(p)h(\theta)+h(\psi)
$$

姿态项为 $r_o=-\Delta t\,w_o O_t$，其中 $\theta$ 为工具倾角，$\psi$ 为按几何旋转对称性折叠后的偏航角误差（正 $n$ 边形周期为 $2\pi/n$，L 形为 $2\pi$，圆形为 $0$）。

## 5. 负载与动作成本

$$
f_t=\mathrm{clip}\left(\frac{\|F\|_2-5}{30-5},0,1\right),
\qquad
m_t=\mathrm{clip}\left(\frac{\|M\|_2-0.1}{2-0.1},0,1\right)
$$

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

## 6. 终止条件与防自杀边界

- **成功终止（+20）**：真实尖端连续 **5 个策略步**（100 ms）满足 $z_t\le-d+0.0002\text{ m}$、$e_{xy}<0.4\,\text{clearance}$、$\theta<0.4\,\text{clearance}/L_{\mathrm{peg}}$、$\psi<0.4\,\text{clearance}/R$，且未触发安全越界。
- **安全失败（−200）**：未滤波峰值力 $>30\text{ N}$、峰值力矩 $>2\text{ N}\cdot\text{m}$、逃出工作空间或倾角 $>35^\circ$。
- **防自杀约束**：默认单步运行成本理论上界 $C_{\mathrm{step,max}}\approx0.18735$，折扣因子 $\gamma=0.9985$ 下无穷步累计运行成本上界为 $\frac{C_{\mathrm{step,max}}}{1-\gamma}\approx124.90 < P_X=200$。配置同时检查安全罚分大于单步上界加超时罚分，以避免安全终止成为逃避运行成本的廉价选择；这不是策略绝不会违规的保证。验证测试见 [test/test_env_curriculum.py](./test/test_env_curriculum.py)。
