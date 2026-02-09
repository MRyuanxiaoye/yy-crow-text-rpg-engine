# -*- coding: utf-8 -*-
"""E1 Layer 2: 任务调度器 — 可中断/可续跑

管理任务队列，支持：
  - generate: 生成完整任务清单
  - run: 开始/继续执行（Ctrl+C 安全退出）
  - status: 查看进度

用法:
  python3 -m src.research.experiment.e1_scheduler generate
  python3 -m src.research.experiment.e1_scheduler run [--lottery X] [--priority N] [--dim D]
  python3 -m src.research.experiment.e1_scheduler status
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, RESULTS_DIR,
)
from research.experiment.e1_search_engine import run_single_task

# === 路径 ===
E1_DIR = RESULTS_DIR / "e1_search"
TASK_DIR = E1_DIR / "tasks"
MANIFEST_PATH = E1_DIR / "task_manifest.json"
PROGRESS_PATH = E1_DIR / "progress.json"

E1_DIR.mkdir(parents=True, exist_ok=True)
TASK_DIR.mkdir(parents=True, exist_ok=True)

# === 中断信号 ===
_interrupted = False


def _signal_handler(sig, frame):
    global _interrupted
    _interrupted = True
    log("收到中断信号，当前任务完成后安全退出...")


# === 任务定义 ===

# 单维度任务
SINGLE_DIMS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6',
               'D7', 'D8', 'D9', 'D10', 'D11', 'D12']

# 二维交叉（计划中的高价值交叉）
CROSS_2D = [
    # Step10a 已有的 5 种
    ('D1', 'D3'), ('D1', 'D4'), ('D2', 'D3'), ('D4', 'D6'), ('D5', 'D3'),
    # 新增高价值交叉
    ('D1', 'D7'), ('D1', 'D8'), ('D3', 'D12'), ('D7', 'D3'),
    ('D8', 'D10'), ('D5', 'D7'), ('D6', 'D12'), ('D1', 'D12'),
]

# 三维交叉（条件性启用，先定义候选）
CROSS_3D_CANDIDATES = [
    ('D1', 'D3', 'D7'),
    ('D1', 'D3', 'D12'),
    ('D1', 'D4', 'D8'),
]

# 优先级映射
PRIORITY_MAP = {
    # P1: 最高优先级 — 跨位置关联 + 组合统计量
    'D7': 1, 'D8': 1,
    # P2: 高优先级 — 冷热度 + 最强交叉
    'D12': 2,
    # P3: 中优先级 — 其余新维度
    'D9': 3, 'D10': 3, 'D11': 3,
    # 已有维度作为基线
    'D1': 3, 'D2': 3, 'D3': 3, 'D4': 3, 'D5': 3, 'D6': 3,
}

CROSS_PRIORITY = {
    ('D1', 'D7'): 2,
    ('D1', 'D8'): 2,
    ('D7', 'D3'): 2,
    ('D3', 'D12'): 3,
    ('D8', 'D10'): 3,
    ('D5', 'D7'): 3,
    ('D6', 'D12'): 3,
    ('D1', 'D12'): 3,
    # Step10a 已有交叉，低优先级（基线复用）
    ('D1', 'D3'): 4,
    ('D1', 'D4'): 4,
    ('D2', 'D3'): 4,
    ('D4', 'D6'): 4,
    ('D5', 'D3'): 4,
}

LOTTERIES = ['daletou', 'shuangseqiu']
LOTTERY_SHORT = {'daletou': 'dlt', 'shuangseqiu': 'ssq'}
POSITIONS = {'daletou': 5, 'shuangseqiu': 6}


def make_task_id(lottery, pos, dims):
    """生成任务ID"""
    short = LOTTERY_SHORT[lottery]
    dim_str = '_x_'.join(dims)
    return f"{short}_P{pos}_{dim_str}"


def generate_manifest():
    """生成完整任务清单"""
    tasks = []

    for lottery in LOTTERIES:
        n_pos = POSITIONS[lottery]
        # 单维度
        for dim in SINGLE_DIMS:
            priority = PRIORITY_MAP.get(dim, 3)
            for pos in range(n_pos):
                tasks.append({
                    'task_id': make_task_id(lottery, pos, [dim]),
                    'lottery': lottery,
                    'position': pos,
                    'dimensions': [dim],
                    'priority': priority,
                })

        # 二维交叉
        for dim_a, dim_b in CROSS_2D:
            priority = CROSS_PRIORITY.get((dim_a, dim_b), 3)
            for pos in range(n_pos):
                tasks.append({
                    'task_id': make_task_id(lottery, pos, [dim_a, dim_b]),
                    'lottery': lottery,
                    'position': pos,
                    'dimensions': [dim_a, dim_b],
                    'priority': priority,
                })

        # 三维交叉候选（priority=5，最后执行）
        for dims in CROSS_3D_CANDIDATES:
            for pos in range(n_pos):
                tasks.append({
                    'task_id': make_task_id(lottery, pos, list(dims)),
                    'lottery': lottery,
                    'position': pos,
                    'dimensions': list(dims),
                    'priority': 5,
                })

    # 按优先级排序
    tasks.sort(key=lambda t: (t['priority'], t['lottery'], t['position']))

    manifest = {
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'total_tasks': len(tasks),
        'tasks': tasks,
    }

    save_json(manifest, MANIFEST_PATH)
    log(f"任务清单已生成: {len(tasks)} 个任务")

    # 统计
    by_priority = {}
    by_lottery = {}
    by_ndim = {}
    for t in tasks:
        p = t['priority']
        by_priority[p] = by_priority.get(p, 0) + 1
        lt = t['lottery']
        by_lottery[lt] = by_lottery.get(lt, 0) + 1
        nd = len(t['dimensions'])
        by_ndim[nd] = by_ndim.get(nd, 0) + 1

    log(f"  按优先级: {dict(sorted(by_priority.items()))}")
    log(f"  按彩种: {by_lottery}")
    log(f"  按维度数: {by_ndim}")

    return manifest


def get_completed_tasks():
    """扫描结果目录，返回已完成的任务ID集合"""
    completed = set()
    for f in TASK_DIR.glob("*.json"):
        try:
            result = load_json(f)
            if result.get('status') == 'completed':
                completed.add(f.stem)
        except Exception:
            pass
    return completed


def get_pending_tasks(lottery_filter=None, priority_filter=None, dim_filter=None):
    """获取待执行任务列表"""
    if not MANIFEST_PATH.exists():
        log("任务清单不存在，请先运行 generate")
        return []

    manifest = load_json(MANIFEST_PATH)
    completed = get_completed_tasks()

    pending = []
    for t in manifest['tasks']:
        if t['task_id'] in completed:
            continue
        if lottery_filter and t['lottery'] != lottery_filter:
            continue
        if priority_filter is not None and t['priority'] > priority_filter:
            continue
        if dim_filter and dim_filter not in t['dimensions']:
            continue
        pending.append(t)

    pending.sort(key=lambda t: (t['priority'], t['lottery'], t['position']))
    return pending


def save_task_result(task_id, result):
    """原子保存单个任务结果"""
    path = TASK_DIR / f"{task_id}.json"
    save_json(result, path)


def update_progress(n_done, n_total, n_failed, start_time, by_lottery, by_dim):
    """更新进度文件"""
    elapsed = time.time() - start_time
    progress = {
        'last_updated': datetime.now().isoformat(),
        'total_tasks': n_total,
        'completed': n_done,
        'failed': n_failed,
        'pending': n_total - n_done - n_failed,
        'elapsed_seconds': round(elapsed, 1),
        'avg_seconds_per_task': round(elapsed / max(n_done, 1), 1),
        'by_lottery': by_lottery,
        'by_dimension': by_dim,
    }
    save_json(progress, PROGRESS_PATH)


def run_all(lottery_filter=None, priority_filter=None, dim_filter=None):
    """主执行循环"""
    global _interrupted
    _interrupted = False
    signal.signal(signal.SIGINT, _signal_handler)

    pending = get_pending_tasks(lottery_filter, priority_filter, dim_filter)
    if not pending:
        log("没有待执行的任务")
        return

    completed_set = get_completed_tasks()
    manifest = load_json(MANIFEST_PATH)
    n_total = manifest['total_tasks']
    n_already_done = len(completed_set)

    log(f"待执行: {len(pending)} 个任务 (已完成: {n_already_done}/{n_total})")

    # 缓存已加载的数据
    data_cache = {}
    start_time = time.time()
    n_done_session = 0
    n_failed = 0
    by_lottery = {}
    by_dim = {}

    for task in pending:
        if _interrupted:
            log(f"安全退出。本次完成 {n_done_session}，"
                f"总进度 {n_already_done + n_done_session}/{n_total}")
            break

        task_id = task['task_id']
        lottery = task['lottery']
        pos = task['position']
        dims = task['dimensions']

        log(f"[{n_already_done + n_done_session + 1}/{n_total}] "
            f"{task_id} (P{task['priority']})")

        # 加载数据（缓存）
        if lottery not in data_cache:
            log(f"  加载 {lottery} 数据...")
            data_cache[lottery] = LotteryData(lottery)

        data = data_cache[lottery]
        train_end_idx = int(data.n_draws * 0.6)

        try:
            t0 = time.time()
            result = run_single_task(data, pos, dims, train_end_idx)
            elapsed_task = time.time() - t0

            result['task_id'] = task_id
            result['lottery'] = lottery
            result['elapsed_seconds'] = round(elapsed_task, 1)

            save_task_result(task_id, result)
            n_done_session += 1

            # 更新统计
            lt_stats = by_lottery.setdefault(lottery, {'completed': 0, 'total': 0})
            lt_stats['completed'] += 1

            dim_key = 'x'.join(dims)
            dim_stats = by_dim.setdefault(dim_key, {
                'completed': 0, 'total': 0, 'valid_patterns': 0})
            dim_stats['completed'] += 1
            dim_stats['valid_patterns'] += result.get('n_validated', 0)

            log(f"  完成: {result['n_patterns_found']} 模式, "
                f"{result['n_validated']} 验证通过, {elapsed_task:.1f}s")

        except Exception as e:
            n_failed += 1
            log(f"  失败: {e}")
            # 保存失败记录
            save_task_result(task_id, {
                'status': 'failed',
                'task_id': task_id,
                'lottery': lottery,
                'position': pos,
                'dimensions': dims,
                'error': str(e),
            })

        # 每 10 个任务更新一次进度
        if (n_done_session + n_failed) % 10 == 0:
            update_progress(n_already_done + n_done_session, n_total,
                            n_failed, start_time, by_lottery, by_dim)

    # 最终进度更新
    update_progress(n_already_done + n_done_session, n_total,
                    n_failed, start_time, by_lottery, by_dim)

    total_elapsed = time.time() - start_time
    log(f"\n本次运行完成: {n_done_session} 个任务, {n_failed} 个失败, "
        f"耗时 {total_elapsed:.0f}s")


def show_status():
    """显示当前进度"""
    if not MANIFEST_PATH.exists():
        log("任务清单不存在，请先运行 generate")
        return

    manifest = load_json(MANIFEST_PATH)
    completed = get_completed_tasks()
    n_total = manifest['total_tasks']
    n_done = len(completed)

    log(f"\n{'═' * 50}")
    log(f"  E1 搜索进度: {n_done}/{n_total} ({n_done/n_total*100:.1f}%)")
    log(f"{'═' * 50}")

    # 按优先级统计
    by_priority = {}
    by_lottery = {}
    by_ndim = {}
    total_patterns = 0
    total_validated = 0

    for t in manifest['tasks']:
        p = t['priority']
        lt = t['lottery']
        nd = len(t['dimensions'])
        done = t['task_id'] in completed

        bp = by_priority.setdefault(p, {'done': 0, 'total': 0})
        bp['total'] += 1
        if done:
            bp['done'] += 1

        bl = by_lottery.setdefault(lt, {'done': 0, 'total': 0})
        bl['total'] += 1
        if done:
            bl['done'] += 1

        bn = by_ndim.setdefault(nd, {'done': 0, 'total': 0})
        bn['total'] += 1
        if done:
            bn['done'] += 1

    # 统计已完成任务的模式数
    for f in TASK_DIR.glob("*.json"):
        try:
            result = load_json(f)
            if result.get('status') == 'completed':
                total_patterns += result.get('n_patterns_found', 0)
                total_validated += result.get('n_validated', 0)
        except Exception:
            pass

    log(f"\n  按优先级:")
    for p in sorted(by_priority.keys()):
        s = by_priority[p]
        log(f"    P{p}: {s['done']}/{s['total']}")

    log(f"\n  按彩种:")
    for lt, s in by_lottery.items():
        log(f"    {lt}: {s['done']}/{s['total']}")

    log(f"\n  按维度数:")
    for nd in sorted(by_ndim.keys()):
        s = by_ndim[nd]
        log(f"    {nd}维: {s['done']}/{s['total']}")

    log(f"\n  累计模式: {total_patterns} 个发现, {total_validated} 个验证通过")

    if PROGRESS_PATH.exists():
        prog = load_json(PROGRESS_PATH)
        log(f"  上次更新: {prog.get('last_updated', '?')}")
        log(f"  平均耗时: {prog.get('avg_seconds_per_task', '?')}s/任务")


# ============================================================
#  命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='E1 条件空间搜索调度器')
    parser.add_argument('command', choices=['generate', 'run', 'status'],
                        help='generate=生成任务清单, run=执行, status=查看进度')
    parser.add_argument('--lottery', type=str, default=None,
                        choices=['daletou', 'shuangseqiu'],
                        help='只跑指定彩种')
    parser.add_argument('--priority', type=int, default=None,
                        help='只跑优先级<=N的任务')
    parser.add_argument('--dim', type=str, default=None,
                        help='只跑包含指定维度的任务 (如 D7)')

    args = parser.parse_args()

    setup_logging()

    if args.command == 'generate':
        generate_manifest()
    elif args.command == 'run':
        run_all(args.lottery, args.priority, args.dim)
    elif args.command == 'status':
        show_status()


if __name__ == '__main__':
    main()
