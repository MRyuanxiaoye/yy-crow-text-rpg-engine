# -*- coding: utf-8 -*-
"""
多核并行引擎

封装 multiprocessing，提供统一的并行任务执行接口。
支持进度回调和结果收集。
"""

import os
import time
import multiprocessing as mp
from typing import Callable, List, Any, Dict, Optional, Iterable
from functools import partial


def get_n_workers(requested: int = 7) -> int:
    """获取可用的工作进程数"""
    cpu_count = os.cpu_count() or 4
    return min(requested, max(1, cpu_count - 1))


def parallel_map(
    func: Callable,
    tasks: List[Any],
    n_workers: int = 7,
    chunk_size: int = 100,
    desc: str = "",
) -> List[Any]:
    """
    并行执行任务列表。

    参数:
        func: 处理函数，接受单个任务参数
        tasks: 任务列表
        n_workers: 工作进程数
        chunk_size: 每个进程的任务块大小
        desc: 任务描述（用于日志）

    返回:
        结果列表（与 tasks 顺序对应）
    """
    n_workers = get_n_workers(n_workers)
    total = len(tasks)

    if total == 0:
        return []

    # 任务太少时不用多进程
    if total <= chunk_size or n_workers <= 1:
        print(f"[并行] {desc}: 单进程执行 {total} 个任务")
        results = []
        for i, task in enumerate(tasks):
            results.append(func(task))
            if (i + 1) % 10000 == 0:
                print(f"[并行] {desc}: {i+1}/{total}")
        return results

    print(f"[并行] {desc}: {n_workers} 进程执行 {total} 个任务, "
          f"chunk_size={chunk_size}")

    start = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(func, tasks, chunksize=chunk_size)
    elapsed = time.time() - start

    print(f"[并行] {desc}: 完成, 耗时 {elapsed:.1f}s, "
          f"速度 {total/elapsed:.0f} 任务/秒")

    return results


def parallel_starmap(
    func: Callable,
    tasks: List[tuple],
    n_workers: int = 7,
    chunk_size: int = 100,
    desc: str = "",
) -> List[Any]:
    """
    并行执行多参数任务。

    参数:
        func: 处理函数，接受多个参数
        tasks: 参数元组列表
        n_workers: 工作进程数
        chunk_size: 每个进程的任务块大小
        desc: 任务描述

    返回:
        结果列表
    """
    n_workers = get_n_workers(n_workers)
    total = len(tasks)

    if total == 0:
        return []

    if total <= chunk_size or n_workers <= 1:
        print(f"[并行] {desc}: 单进程执行 {total} 个任务")
        results = []
        for i, args in enumerate(tasks):
            results.append(func(*args))
            if (i + 1) % 10000 == 0:
                print(f"[并行] {desc}: {i+1}/{total}")
        return results

    print(f"[并行] {desc}: {n_workers} 进程执行 {total} 个任务")

    start = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(func, tasks, chunksize=chunk_size)
    elapsed = time.time() - start

    print(f"[并行] {desc}: 完成, 耗时 {elapsed:.1f}s")

    return results


def chunked(iterable: List, size: int) -> List[List]:
    """将列表分块"""
    return [iterable[i:i+size] for i in range(0, len(iterable), size)]
