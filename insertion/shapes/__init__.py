"""插头与孔板的形状注册表。"""

from .base import InsertionShape
from .circle import SHAPE as CIRCLE
from .hexagon import SHAPE as HEXAGON
from .l import SHAPE as L
from .pentagon import SHAPE as PENTAGON
from .square import SHAPE as SQUARE
from .triangle import SHAPE as TRIANGLE

# 保持顺序：索引用于 Critic 的 one-hot 编码以及环境随机种子偏移。
_SHAPES = {shape.name: shape for shape in (TRIANGLE, SQUARE, PENTAGON, HEXAGON, L, CIRCLE)}
ALL_SHAPES = tuple(_SHAPES)
TRAIN_SHAPES = tuple(name for name, shape in _SHAPES.items() if shape.train_by_default)


def get_shape(name: str) -> InsertionShape:
    return _SHAPES[name]


__all__ = ["ALL_SHAPES", "TRAIN_SHAPES", "InsertionShape", "get_shape"]
