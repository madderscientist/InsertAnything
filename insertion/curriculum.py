"""按完整回合成功率驱动的孔间隙、初始难度与传感器噪声课程"""

from collections import deque
import math


def adaptive_environment_counts(counts: tuple[int, ...], failures: list[float],
                                unit: int = 1) -> tuple[int, ...]:
    if (not counts or len(counts) != len(failures) or type(unit) is not int or unit < 1
            or any(type(n) is not int or n < unit or n % unit for n in counts)
            or any(not math.isfinite(f) or not 0 <= f <= 1 for f in failures)):
        raise ValueError("invalid adaptive sampling counts, failure rates or device unit")
    total = sum(counts) // unit
    baseline = total / len(counts)
    failure_sum = sum(failures)
    targets = [0.3 * baseline + 0.7 * total * (
        f / failure_sum if failure_sum else 1 / len(counts)) for f in failures]
    result = [max(1, math.floor(0.3 * baseline))] * len(counts)
    for _ in range(total - sum(result)):
        index = max(range(len(counts)), key=lambda i: targets[i]-result[i])
        result[index] += 1
    return tuple(n*unit for n in result)


class SuccessNoiseCurriculum:
    window_size = 1000
    threshold = 0.8

    def __init__(self, shapes: tuple[str, ...],
                 clearances: tuple[float, ...] = (0.0001,),
                 noise_levels: int = 10):
        if not clearances or any(
                not math.isfinite(c) or c <= 0 for c in clearances
        ) or any(a <= b for a, b in zip(clearances, clearances[1:])):
            raise ValueError("curriculum clearances must be finite, positive and strictly decreasing")
        if type(noise_levels) is not int or noise_levels < 0:
            raise ValueError("noise_levels must be a nonnegative integer")
        self.clearances = tuple(float(c) for c in clearances)
        self.noise_levels = noise_levels
        self.clearance_stages = len(self.clearances) - 1
        self.max_level = self.clearance_stages + self.noise_levels
        self.level = 0
        self.windows = {shape: deque(maxlen=self.window_size) for shape in shapes}

    @property
    def clearance(self) -> float:
        # 先逐级收紧孔间隙至目标值，再进入噪声提升阶段
        return self.clearances[min(self.level, self.clearance_stages)]

    @property
    def difficulty(self) -> float:
        # 初始位姿与刚度随机化难度随课程同步推进，避免零成功率时难度失控
        if self.max_level == 0:
            return 1.0
        if self.clearance_stages > 0:
            return min(1.0, self.level / self.clearance_stages)
        return min(1.0, self.level / self.max_level)

    @property
    def scale(self) -> float:
        if self.noise_levels == 0:
            return 1.0
        if self.level <= self.clearance_stages:
            return 0.0
        return (self.level - self.clearance_stages) / self.noise_levels

    def scale_for(self, shape: str) -> float | None:
        # None：本课程不管理噪声，沿用配置的步数噪声课程
        del shape
        return self.scale if self.noise_levels > 0 else None

    def clearance_for(self, shape: str) -> float:
        del shape
        return self.clearance

    def load_levels(self, state: dict) -> None:
        self.level = state["level"]

    def clear_history(self) -> None:
        for window in self.windows.values():
            window.clear()

    def update(self, outcomes: dict[str, list[bool]],
               boundary: dict[str, list[bool]] | None = None) -> bool:
        del boundary
        if set(outcomes) != set(self.windows):
            raise ValueError("curriculum outcomes must include every configured shape")
        for shape, values in outcomes.items():
            self.windows[shape].extend(values)
        if self.level < self.max_level and all(
                len(window) == self.window_size
                and sum(window) / len(window) >= self.threshold
                for window in self.windows.values()):
            self.level += 1
            self.clear_history()
            return True
        return False

    def state_dict(self) -> dict:
        return {"level": self.level,
                "windows": {shape: list(window) for shape, window in self.windows.items()}}

    def load_state_dict(self, state: dict) -> None:
        if (type(state["level"]) is not int or not 0 <= state["level"] <= self.max_level
                or set(state["windows"]) != set(self.windows)):
            raise ValueError("invalid success noise curriculum checkpoint")
        for values in state["windows"].values():
            if len(values) > self.window_size or any(type(value) is not bool for value in values):
                raise ValueError("invalid curriculum outcome window")
        self.level = state["level"]
        for shape, values in state["windows"].items():
            self.windows[shape].clear()
            self.windows[shape].extend(values)


ADR_BOUNDARY_FRACTION = 0.5


class ADRCurriculum:
    """两维自动域随机化：每形状噪声 φ[0] 与间隙 φ[1] ∈ [0, 2] 各按自己的边界回合成功率双向调节。

    φ ≤ 1 时放开难侧边界，易侧固定（噪声下限 0、间隙上限最松值）；φ > 1 时难侧已满，
    易侧边界按 φ-1 向难侧收拢，φ = 2 时整维钉在最难值（噪声系数 1、目标间隙）。
    边界回合把一维钉在难侧边界（噪声取 min(φ[0], 1)，间隙取最紧值），另一维照常采样；
    c(φ) 在最松与目标间隙之间对数插值；初始位姿、阻尼等其余随机化难度取 min φ[0]（截到 1）"""
    window_size = 1000
    expand_above = 0.8
    shrink_below = 0.5
    step = 0.05
    limit = 2.0
    # 每次调节 φ 前需累计的边界回合数
    boundary_episodes = 200

    def __init__(self, shapes: tuple[str, ...], loosest: float, tightest: float):
        if not (math.isfinite(loosest) and 0 < tightest <= loosest):
            raise ValueError("ADR clearance range must satisfy 0 < tightest <= loosest")
        self.loosest, self.tightest = float(loosest), float(tightest)
        self.phi = {shape: [0.0, 0.0] for shape in shapes}
        self.boundary = {shape: [[0, 0], [0, 0]] for shape in shapes}
        self.windows = {shape: deque(maxlen=self.window_size) for shape in shapes}

    @property
    def difficulty(self) -> float:
        return min(min(phi[0], 1.) for phi in self.phi.values())

    @property
    def scale(self) -> float:
        return self.difficulty

    def scale_for(self, shape: str) -> float:
        return min(self.phi[shape][0], 1.)

    def noise_floor_for(self, shape: str) -> float:
        return max(self.phi[shape][0] - 1., 0.)

    def clearance_at(self, progress: float) -> float:
        return self.loosest * (self.tightest/self.loosest) ** progress

    def clearance_for(self, shape: str) -> float:
        return self.clearance_at(min(self.phi[shape][1], 1.))

    def clearance_ceiling_for(self, shape: str) -> float:
        return self.clearance_at(max(self.phi[shape][1] - 1., 0.))

    def load_levels(self, state: dict) -> None:
        self.phi.update({shape: list(phi) for shape, phi in state["phi"].items() if shape in self.phi})

    def clear_history(self) -> None:
        for shape, window in self.windows.items():
            window.clear()
            self.boundary[shape] = [[0, 0], [0, 0]]

    def update(self, outcomes: dict[str, list[bool]],
               boundary: dict[str, tuple[list[bool], list[bool]]] | None = None) -> bool:
        boundary = boundary or {}
        if set(outcomes) != set(self.windows) or not set(boundary) <= set(self.windows):
            raise ValueError("curriculum outcomes must include every configured shape")
        changed = False
        for shape, per_dimension in boundary.items():
            for d, values in enumerate(per_dimension):
                stats = self.boundary[shape][d]
                stats[0] += sum(values)
                stats[1] += len(values)
                if stats[1] < self.boundary_episodes:
                    continue
                rate, old = stats[0] / stats[1], self.phi[shape][d]
                if rate > self.expand_above:
                    self.phi[shape][d] = min(self.limit, round(old + self.step, 6))
                elif rate < self.shrink_below:
                    self.phi[shape][d] = max(0.0, round(old - self.step, 6))
                self.boundary[shape][d] = [0, 0]
                changed |= self.phi[shape][d] != old
        for shape, values in outcomes.items():
            self.windows[shape].extend(values)
        return changed

    def state_dict(self) -> dict:
        return {"phi": {shape: list(phi) for shape, phi in self.phi.items()},
                "boundary": {shape: [list(s) for s in stats] for shape, stats in self.boundary.items()},
                "windows": {shape: list(window) for shape, window in self.windows.items()}}

    def load_state_dict(self, state: dict) -> None:
        if (set(state.get("phi", ())) != set(self.phi)
                or any(len(phi) != 2 or not all(0 <= p <= self.limit for p in phi) for phi in state["phi"].values())
                or set(state.get("boundary", ())) != set(self.boundary)
                or set(state.get("windows", ())) != set(self.windows)):
            raise ValueError("invalid ADR curriculum checkpoint")
        for values in state["windows"].values():
            if len(values) > self.window_size or any(type(value) is not bool for value in values):
                raise ValueError("invalid curriculum outcome window")
        self.load_levels(state)
        self.boundary = {shape: [[int(s), int(n)] for s, n in stats] for shape, stats in state["boundary"].items()}
        for shape, values in state["windows"].items():
            self.windows[shape].clear()
            self.windows[shape].extend(values)
