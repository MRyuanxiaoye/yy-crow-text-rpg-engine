# -*- coding: utf-8 -*-
"""
方法链 CLI 入口

用法：
  python3 src/pipeline/runner.py                    # 默认大乐透，预算20元
  python3 src/pipeline/runner.py --type shuangseqiu # 双色球
  python3 src/pipeline/runner.py --budget 50        # 预算50元
  python3 src/pipeline/runner.py --save             # 保存报告到文件
"""

import sys
import argparse
from pathlib import Path

# 确保 src 目录在 sys.path 中
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from pipeline.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="彩票方法链分析")
    parser.add_argument(
        "--type", "-t",
        choices=["daletou", "shuangseqiu"],
        default="daletou",
        help="彩种类型（默认: daletou）",
    )
    parser.add_argument(
        "--budget", "-b",
        type=int, default=20,
        help="购买预算，单位元（默认: 20）",
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="保存报告到 data/ 目录",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="指定报告输出路径",
    )

    args = parser.parse_args()

    # 运行管道
    pipe = Pipeline(args.type, budget=args.budget)
    report = pipe.run()

    # 打印购买方案摘要
    s4 = report.get("stage4", {})
    tickets = s4.get("tickets", [])

    print(f"\n{'='*50}")
    print(f"  购买方案")
    print(f"{'='*50}")
    for i, t in enumerate(tickets):
        if t["type"] == "胆拖":
            print(f"  第{i+1}张 [{t['type']}] {t['cost']}元 {t['combinations']}注")
            print(f"    胆码(红): {t['dan_red']}")
            print(f"    拖码(红): {t['tuo_red']}")
            if t.get("dan_blue"):
                print(f"    胆码(蓝): {t['dan_blue']}")
            if t.get("tuo_blue"):
                print(f"    拖码(蓝): {t['tuo_blue']}")
        else:
            red = t.get("red_balls", [])
            blue = t.get("blue_balls", [])
            red_str = " ".join(f"{n:02d}" for n in red)
            blue_str = " ".join(f"{n:02d}" for n in blue)
            print(f"  第{i+1}张 [{t['type']}] {red_str} | {blue_str}")

    print(f"\n  总花费: {s4.get('total_cost', 0)}元")
    print(f"  总注数: {s4.get('total_combinations', 0)}注")
    print(f"  覆盖率: {s4.get('coverage_pct', 0):.1%}")
    print(f"{'='*50}")

    # 保存报告
    if args.save or args.output:
        pipe.save_report(args.output)


if __name__ == "__main__":
    main()
