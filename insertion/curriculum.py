"""按完整回合成功率驱动的孔间隙、初始难度与传感器噪声课程"""

from collections import deque
import math


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
            progress = min(1.0, self.level / self.clearance_stages)
        else:
            progress = min(1.0, self.level / self.max_level)
        return 0.25 + 0.75 * progress

    @property
    def scale(self) -> float:
        if self.noise_levels == 0:
            return 1.0
        if self.level <= self.clearance_stages:
            return 0.0
        return (self.level - self.clearance_stages) / self.noise_levels

    def update(self, outcomes: dict[str, list[bool]]) -> bool:
        if set(outcomes) != set(self.windows):
            raise ValueError("curriculum outcomes must include every configured shape")
        for shape, values in outcomes.items():
            self.windows[shape].extend(values)
        if self.level < self.max_level and all(
                len(window) == self.window_size
                and sum(window) / len(window) >= self.threshold
                for window in self.windows.values()):
            self.level += 1
            for window in self.windows.values():
                window.clear()
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

