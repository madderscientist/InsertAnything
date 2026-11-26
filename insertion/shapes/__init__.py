"""插头与孔板的形状注册表。"""

from .base import InsertionShape
from .circle import SHAPE as CIRCLE
from .dual import SHAPE as DUAL
from .hexagon import SHAPE as HEXAGON
from .l import SHAPE as L
from .pentagon import SHAPE as PENTAGON
from .rectangle import SHAPE as RECTANGLE
from .square import SHAPE as SQUARE
from .triangle import SHAPE as TRIANGLE

# 保持顺序：索引用于 Critic 的 one-hot 编码以及环境随机种子偏移；新形状只能追加在末尾。
_SHAPES = {shape.name: shape for shape in (TRIANGLE, SQUARE, PENTAGON, HEXAGON, L, CIRCLE,
                                           RECTANGLE, DUAL)}
ALL_SHAPES = tuple(_SHAPES)
# 不开启形状随机化时 Critic 只编码原有 6 种形状，保持旧 checkpoint 的 Critic 维度
LEGACY_SHAPE_COUNT = 6
TRAIN_SHAPES = tuple(name for name, shape in _SHAPES.items() if shape.train_by_default)


def get_shape(name: str) -> InsertionShape:
    return _SHAPES[name]


__all__ = ["ALL_SHAPES", "LEGACY_SHAPE_COUNT", "TRAIN_SHAPES", "InsertionShape", "get_shape"]
