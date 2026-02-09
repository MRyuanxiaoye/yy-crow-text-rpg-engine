"""实验公共工具函数"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# === 路径 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RULES_DIR = RESULTS_DIR / "rules"
EXPERIMENT_DIR = RESULTS_DIR / "experiment"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# === 日志 ===
def setup_logging():
    """配置日志，同时输出到控制台和文件"""
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(EXPERIMENT_DIR / "experiment.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger():
    return logging.getLogger("experiment")


def log(msg):
    get_logger().info(msg)


# === 计时 ===
class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        if self.name:
            log(f"  [{self.name}] 开始...")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            log(f"  [{self.name}] 完成，耗时 {self.elapsed:.1f}s")


# === 数据IO ===
def save_json(data, path):
    """保存JSON，处理numpy类型"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    log(f"  已保存: {path.name}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path, **arrays):
    """保存numpy数组"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    log(f"  已保存: {path.name}")


def load_npz(path):
    return np.load(path, allow_pickle=True)


# === 归一化 ===
class Normalizer:
    """简单的Z-score归一化器"""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, name, values):
        arr = np.array(values, dtype=float)
        self.means[name] = arr.mean()
        self.stds[name] = arr.std() if arr.std() > 0 else 1.0

    def transform(self, name, value):
        return (value - self.means[name]) / self.stds[name]

    def fit_transform(self, name, values):
        self.fit(name, values)
        return np.array([(v - self.means[name]) / self.stds[name] for v in values])


# === 条件解析 ===
CONDITION_TYPES = ['val', 'dir', 'odd', 'even', 'big', 'small', 'diff', 'sum', 'span', 'ac', 'consec']
CONDITION_TYPE_INDEX = {t: i for i, t in enumerate(CONDITION_TYPES)}


def parse_condition_type(cond):
    """解析条件字符串，返回条件类型索引(0-10)"""
    parts = cond.split('_')
    if parts[0].startswith('P'):
        # P{n}_{type}_...
        ctype = parts[1]
    elif parts[0] == 'combo':
        # combo_{stat}_{lo}_{hi}
        ctype = parts[1]
    else:
        ctype = parts[0]
    return CONDITION_TYPE_INDEX.get(ctype, 0)


def parse_condition_position(cond):
    """解析条件涉及的位置，返回0-4或None"""
    parts = cond.split('_')
    if parts[0].startswith('P') and len(parts[0]) == 2:
        return int(parts[0][1])
    return None


def parse_target(target):
    """解析target字符串，返回(位置, 方向索引)
    target格式: next_P{n}_{U/D/E}
    方向索引: U=0, D=1, E=2
    """
    parts = target.split('_')
    pos = int(parts[1][1])
    dir_map = {'U': 0, 'D': 1, 'E': 2}
    direction = dir_map[parts[2]]
    return pos, direction
