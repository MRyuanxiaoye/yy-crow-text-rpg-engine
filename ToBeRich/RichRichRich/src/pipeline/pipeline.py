# -*- coding: utf-8 -*-
"""
Pipeline 主类：串联阶段0-4，形成完整方法链。
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from pipeline.config import get_config, LotteryConfig
from pipeline.stage0_preprocess import run_stage0, load_draws
from pipeline.stage1_exclusion import run_stage1
from pipeline.stage2_weight import run_stage2
from pipeline.stage3_combination import run_stage3
from pipeline.stage4_purchase import run_stage4


class Pipeline:
    """方法链管道：从原始数据到购买方案"""

    def __init__(self, lottery_type: str, budget: int = 20):
        self.lottery_type = lottery_type
        self.config = get_config(lottery_type)
        self.budget = budget

        # 各阶段结果
        self.stage0_result: Optional[Dict] = None
        self.stage1_result: Optional[Dict] = None
        self.stage2_result: Optional[Dict] = None
        self.stage3_result: Optional[Dict] = None
        self.stage4_result: Optional[Dict] = None
        self.timing: Dict[str, float] = {}

    def run(self) -> Dict[str, Any]:
        """运行完整管道"""
        print(f"{'='*50}")
        print(f"  {self.config.name} 方法链分析")
        print(f"  预算: {self.budget}元")
        print(f"{'='*50}")

        total_start = time.time()

        # 阶段0
        t0 = time.time()
        self.stage0_result = run_stage0(self.lottery_type)
        self.timing["stage0"] = time.time() - t0

        # 阶段1
        t0 = time.time()
        self.stage1_result = run_stage1(self.stage0_result, self.lottery_type)
        self.timing["stage1"] = time.time() - t0

        # 阶段2
        t0 = time.time()
        self.stage2_result = run_stage2(
            self.stage0_result, self.stage1_result, self.lottery_type
        )
        self.timing["stage2"] = time.time() - t0

        # 阶段3
        t0 = time.time()
        self.stage3_result = run_stage3(
            self.stage0_result, self.stage1_result,
            self.stage2_result, self.lottery_type
        )
        self.timing["stage3"] = time.time() - t0

        # 阶段4
        t0 = time.time()
        self.stage4_result = run_stage4(
            self.stage0_result, self.stage2_result,
            self.stage3_result, self.lottery_type, self.budget
        )
        self.timing["stage4"] = time.time() - t0

        self.timing["total"] = time.time() - total_start

        print(f"\n{'='*50}")
        print(f"  管道完成")
        for stage, t in self.timing.items():
            print(f"  {stage}: {t:.2f}s")
        print(f"{'='*50}")

        return self.get_report()

    def get_report(self) -> Dict[str, Any]:
        """获取完整报告数据（可直接传给可视化）"""
        return {
            "metadata": self.stage0_result["metadata"] if self.stage0_result else {},
            "stage0": self.stage0_result or {},
            "stage1": self.stage1_result or {},
            "stage2": self.stage2_result or {},
            "stage3": self.stage3_result or {},
            "stage4": self.stage4_result or {},
            "timing": self.timing,
        }

    def save_report(self, output_path: Optional[str] = None) -> str:
        """保存报告到 JSON 文件"""
        if output_path is None:
            output_dir = Path(__file__).resolve().parent.parent.parent / "data"
            output_path = str(output_dir / f"{self.lottery_type}_report.json")

        report = self.get_report()

        # 清理不可序列化的大数据（共现矩阵、转移矩阵太大）
        clean = json.loads(json.dumps(report, default=str))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)

        print(f"报告已保存: {output_path}")
        return output_path
