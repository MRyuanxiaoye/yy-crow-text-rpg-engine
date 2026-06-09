"""引擎层加载器兼容入口。"""

from __future__ import annotations

from typing import Any

from src.scripts.loader import load_board_regions as _load_board_regions


def load_board_regions(script_id: str) -> dict[str, Any]:
    """从剧本YAML加载棋盘区域定义，转换为运行时格式。"""

    return _load_board_regions(script_id)
