"""规则适配学习实验 - 主入口

全自动运行，无需人工干预。
用法: python3 -m src.research.experiment.run_experiment
"""

import sys
import time
import traceback
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.experiment.utils import setup_logging, log, Timer, save_json, EXPERIMENT_DIR
from research.experiment.phase1_compress import run_phase1
from research.experiment.phase2_replay import run_phase2
from research.experiment.phase3_train import run_phase3
from research.experiment.phase4_analyze import run_phase4


def run_single_lottery(lottery_type):
    """对单个彩种执行完整实验流水线"""
    log(f"\n{'═'*50}")
    log(f"  彩种: {lottery_type}")
    log(f"{'═'*50}")

    # 阶段一：规则压缩
    with Timer(f"阶段一 [{lottery_type}]"):
        clusters, a1_filtered = run_phase1(lottery_type)

    # 阶段二：历史回放
    with Timer(f"阶段二 [{lottery_type}]"):
        (train_X, train_Y, test_X, test_Y,
         train_Y_val, test_Y_val,
         feature_names, train_indices, test_indices, data) = run_phase2(
            lottery_type, clusters, a1_filtered)

    # 阶段三：模型训练
    with Timer(f"阶段三 [{lottery_type}]"):
        results_3a, results_3b, results_3c = run_phase3(
            train_X, train_Y, test_X, test_Y,
            feature_names, data, test_indices)

    # 阶段四：结果分析
    with Timer(f"阶段四 [{lottery_type}]"):
        final_report = run_phase4(lottery_type, results_3a, results_3b, results_3c)

    return final_report


def main():
    logger = setup_logging()
    start_time = time.time()

    log("=" * 60)
    log("  规则适配学习实验")
    log("  全自动运行，无需人工干预")
    log("=" * 60)

    reports = {}

    for lottery_type in ['daletou', 'shuangseqiu']:
        try:
            report = run_single_lottery(lottery_type)
            reports[lottery_type] = report
        except Exception as e:
            log(f"\n[错误] {lottery_type} 实验失败: {e}")
            log(traceback.format_exc())
            reports[lottery_type] = {'error': str(e)}

    # 汇总报告
    total_time = time.time() - start_time
    summary = {
        'total_time_seconds': round(total_time, 1),
        'total_time_minutes': round(total_time / 60, 1),
        'lottery_types': list(reports.keys()),
        'results': {},
    }

    for lt, report in reports.items():
        if 'error' in report:
            summary['results'][lt] = {'status': 'FAILED', 'error': report['error']}
        elif report.get('conclusion'):
            c = report['conclusion']
            summary['results'][lt] = {
                'status': 'OK',
                'verdict': c.get('verdict', 'N/A'),
                'best_method': c.get('best_method', 'N/A'),
                'direction_accuracy': c.get('direction_accuracy', 0),
                'avg_reduction': c.get('avg_reduction', 0),
                'avg_survival': c.get('avg_survival', 0),
                'efficiency': c.get('best_efficiency', 0),
            }

    save_json(summary, EXPERIMENT_DIR / "experiment_summary.json")

    log(f"\n{'═'*60}")
    log(f"  实验完成！总耗时: {summary['total_time_minutes']:.1f} 分钟")
    log(f"{'═'*60}")

    for lt, res in summary['results'].items():
        if res['status'] == 'OK':
            log(f"  {lt}: {res['verdict']} | "
                f"方向准确率={res['direction_accuracy']:.4f} | "
                f"缩减率={res['avg_reduction']:.4f} | "
                f"存活率={res['avg_survival']:.4f} | "
                f"效率={res['efficiency']:.4f}")
        else:
            log(f"  {lt}: FAILED - {res['error']}")

    log(f"\n  结果目录: {EXPERIMENT_DIR}")
    log(f"  最终报告: phase4_final_report_*.json")
    log(f"  实验日志: experiment.log")


if __name__ == "__main__":
    main()
