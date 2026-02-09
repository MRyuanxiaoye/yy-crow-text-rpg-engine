#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
彩票历史数据拉取与更新工具

数据源优先级：
  1. 500.com datachart history.php（大乐透全量 + 双色球早期数据）
  2. cwl.gov.cn 福彩官方API（双色球，数据最全最准）
  3. 78500.cn 移动端页面（增量更新备用，仅最近10期）
  4. 手动CSV导入（兜底）

用法：
  python3 src/data/crawler.py --full          # 拉取全部历史数据
  python3 src/data/crawler.py --update        # 更新最新数据
  python3 src/data/crawler.py --import-csv    # 从CSV导入
  python3 src/data/crawler.py --status        # 查看当前数据状态
"""

import json
import os
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 确保requests可用
try:
    import requests
except ImportError:
    print("错误：需要安装 requests 库")
    print("运行：pip3 install requests")
    sys.exit(1)

# ============================================================
# 通用工具函数
# ============================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}


def fetch_with_retry(url, max_retries=3, timeout=30):
    """带重试的HTTP GET请求"""
    for attempt in range(max_retries):
        try:
            print(f"  请求: {url} (第{attempt+1}次)")
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
            print(f"  状态码: {r.status_code}, 内容长度: {len(r.text)}")
        except requests.exceptions.Timeout:
            print(f"  超时，等待重试...")
        except requests.exceptions.ConnectionError:
            print(f"  连接失败，等待重试...")
        except Exception as e:
            print(f"  异常: {e}")
        if attempt < max_retries - 1:
            wait = (attempt + 1) * 5
            print(f"  等待{wait}秒后重试...")
            time.sleep(wait)
    return None


def load_existing_data(filepath):
    """加载已有的JSON数据文件"""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_data(filepath, data):
    """保存数据到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {filepath} ({data['metadata']['total_draws']}期数据)")


# ============================================================
# 数据源1：500.com datachart (history.php)
# 实际可用的全量数据接口，需分段请求避免超时和限流
# ============================================================

# 500.com 分段请求间隔（秒），避免被限流
FETCH_INTERVAL = 3

def parse_500_dlt(html):
    """解析500.com大乐透HTML表格
    表格结构：td[0]=序号, td[1]=期号(5位), td[2-6]=红球, td[7-8]=蓝球, td[15]=日期
    """
    draws = []
    rows = re.findall(r'<tr class="t_tr1">(.*?)</tr>', html, re.DOTALL)
    for row in rows:
        tds = re.findall(r'<td[^>]*>(.*?)</td>', row)
        cleaned = [re.sub(r'<[^>]+>', '', td).strip() for td in tds]
        if len(cleaned) >= 16 and cleaned[1].isdigit() and len(cleaned[1]) == 5:
            try:
                period = cleaned[1]
                red_balls = sorted([int(cleaned[i]) for i in range(2, 7)])
                blue_balls = sorted([int(cleaned[i]) for i in range(7, 9)])
                date_str = cleaned[15]
                if all(1 <= b <= 35 for b in red_balls) and all(1 <= b <= 12 for b in blue_balls):
                    draws.append({
                        "period": period,
                        "date": date_str,
                        "red_balls": red_balls,
                        "blue_balls": blue_balls,
                    })
            except (ValueError, IndexError):
                continue
    return draws


def parse_500_ssq(html):
    """解析500.com双色球HTML表格
    表格结构：td[0]=序号, td[1]=期号(5位), td[2-7]=红球, td[8]=蓝球
    """
    draws = []
    rows = re.findall(r'<tr class="t_tr1">(.*?)</tr>', html, re.DOTALL)
    for row in rows:
        tds = re.findall(r'<td[^>]*>(.*?)</td>', row)
        cleaned = [re.sub(r'<[^>]+>', '', td).strip() for td in tds]
        if len(cleaned) >= 10 and cleaned[1].isdigit() and len(cleaned[1]) == 5:
            try:
                period = cleaned[1]
                red_balls = sorted([int(cleaned[i]) for i in range(2, 8)])
                blue_ball = int(cleaned[8])
                if all(1 <= b <= 33 for b in red_balls) and 1 <= blue_ball <= 16:
                    draws.append({
                        "period": period,
                        "date": "",
                        "red_balls": red_balls,
                        "blue_ball": blue_ball,
                    })
            except (ValueError, IndexError):
                continue
    return draws


def fetch_500_dlt_full():
    """从500.com分段获取大乐透全部历史数据（2007年至今）"""
    print("\n[数据源] 500.com datachart - 大乐透（分段拉取）")
    segments = [
        ('07001', '08160'), ('09001', '10160'), ('11001', '12160'),
        ('13001', '14160'), ('15001', '16160'), ('17001', '18160'),
        ('19001', '20160'), ('21001', '22160'), ('23001', '24160'),
        ('25001', '26160'),
    ]
    all_draws = []
    for start, end in segments:
        url = f"https://datachart.500.com/dlt/history/newinc/history.php?start={start}&end={end}"
        html = fetch_with_retry(url, timeout=45)
        if html:
            draws = parse_500_dlt(html)
            print(f"  {start}-{end}: {len(draws)}期")
            all_draws.extend(draws)
        else:
            print(f"  {start}-{end}: 获取失败")
        time.sleep(FETCH_INTERVAL)
    print(f"  合计: {len(all_draws)}期")
    return all_draws if all_draws else None


def fetch_500_ssq_early():
    """从500.com获取双色球早期数据（2003-2012），补充福彩API缺失部分"""
    print("\n[数据源] 500.com datachart - 双色球早期数据")
    segments = [
        ('03001', '04160'), ('05001', '06160'), ('07001', '08160'),
        ('09001', '10160'), ('11001', '12160'),
    ]
    all_draws = []
    for start, end in segments:
        url = f"https://datachart.500.com/ssq/history/newinc/history.php?start={start}&end={end}"
        html = fetch_with_retry(url, timeout=45)
        if html:
            draws = parse_500_ssq(html)
            print(f"  {start}-{end}: {len(draws)}期")
            all_draws.extend(draws)
        else:
            print(f"  {start}-{end}: 获取失败")
        time.sleep(FETCH_INTERVAL)
    print(f"  合计: {len(all_draws)}期")
    return all_draws if all_draws else None


# ============================================================
# 数据源2：cwl.gov.cn 福彩官方API（双色球）
# ============================================================

CWL_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.cwl.gov.cn/',
    'Accept': 'application/json',
}


def fetch_cwl_ssq():
    """从福彩官方API获取双色球全部历史数据（约2013年至今）"""
    print("\n[数据源] cwl.gov.cn 福彩官方API - 双色球")
    all_draws = []
    page = 1
    max_pages = 100  # 安全上限

    while page <= max_pages:
        url = (f"https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
               f"?name=ssq&issueCount=&issueStart=&issueEnd=&dayStart=&dayEnd="
               f"&pageNo={page}&pageSize=30&week=&systemType=PC")
        for attempt in range(3):
            try:
                print(f"  请求第{page}页 (第{attempt+1}次)")
                r = requests.get(url, headers=CWL_HEADERS, timeout=20)
                data = r.json()
                items = data.get('result', [])
                if not items:
                    print(f"  第{page}页无数据，结束")
                    print(f"  合计: {len(all_draws)}期")
                    return all_draws if all_draws else None
                for item in items:
                    reds = [int(x) for x in item['red'].split(',')]
                    blue = int(item['blue'])
                    date_str = item['date'].split('(')[0]
                    all_draws.append({
                        "period": item['code'],
                        "date": date_str,
                        "red_balls": sorted(reds),
                        "blue_ball": blue,
                    })
                if page % 10 == 0:
                    print(f"  已拉取{page}页，累计{len(all_draws)}期")
                time.sleep(0.8)
                break
            except Exception as e:
                print(f"  第{page}页第{attempt+1}次失败: {e}")
                time.sleep(3)
        page += 1

    print(f"  合计: {len(all_draws)}期")
    return all_draws if all_draws else None


# ============================================================
# 数据源3：78500.cn（增量更新备用，仅最近10期）
# ============================================================

def parse_78500_dlt(html):
    """解析78500.cn大乐透HTML页面"""
    draws = []
    # 匹配: <strong>2026015期</strong><span>开奖时间 2026-02-04</span>
    # 匹配: <i>01</i><i>04</i><i>10</i><i>13</i><i>17</i><b>03</b><b>11</b>
    sections = re.findall(r'<section class="item">(.*?)</section>', html, re.DOTALL)
    for sec in sections:
        period_match = re.search(r'<strong>(\d+)期</strong>', sec)
        date_match = re.search(r'开奖时间\s*([\d-]+)', sec)
        red_matches = re.findall(r'<i>(\d+)</i>', sec)
        blue_matches = re.findall(r'<b>(\d+)</b>', sec)
        if period_match and len(red_matches) == 5 and len(blue_matches) == 2:
            draws.append({
                "period": period_match.group(1),
                "date": date_match.group(1) if date_match else "",
                "red_balls": sorted([int(x) for x in red_matches]),
                "blue_balls": sorted([int(x) for x in blue_matches]),
            })
    return draws


def parse_78500_ssq(html):
    """解析78500.cn双色球HTML页面"""
    draws = []
    sections = re.findall(r'<section class="item">(.*?)</section>', html, re.DOTALL)
    for sec in sections:
        period_match = re.search(r'<strong>(\d+)期</strong>', sec)
        date_match = re.search(r'开奖时间\s*([\d-]+)', sec)
        red_matches = re.findall(r'<i>(\d+)</i>', sec)
        blue_matches = re.findall(r'<b>(\d+)</b>', sec)
        if period_match and len(red_matches) == 6 and len(blue_matches) == 1:
            draws.append({
                "period": period_match.group(1),
                "date": date_match.group(1) if date_match else "",
                "red_balls": sorted([int(x) for x in red_matches]),
                "blue_ball": int(blue_matches[0]),
            })
    return draws


def fetch_78500_dlt():
    """从78500.cn获取大乐透最近数据"""
    print("\n[数据源] 78500.cn - 大乐透")
    html = fetch_with_retry("https://m.78500.cn/kaijiang/dlt/")
    if html is None:
        print("  78500.cn 大乐透数据获取失败")
        return None
    draws = parse_78500_dlt(html)
    print(f"  解析到 {len(draws)} 期数据")
    return draws


def fetch_78500_ssq():
    """从78500.cn获取双色球最近数据"""
    print("\n[数据源] 78500.cn - 双色球")
    html = fetch_with_retry("https://m.78500.cn/kaijiang/ssq/")
    if html is None:
        print("  78500.cn 双色球数据获取失败")
        return None
    draws = parse_78500_ssq(html)
    print(f"  解析到 {len(draws)} 期数据")
    return draws


# ============================================================
# CSV导入
# ============================================================

def import_csv_dlt(csv_path):
    """从CSV导入大乐透数据
    CSV格式：期号,日期,红1,红2,红3,红4,红5,蓝1,蓝2
    """
    draws = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('期'):
                continue
            parts = line.split(',')
            if len(parts) >= 9:
                try:
                    draws.append({
                        "period": parts[0].strip(),
                        "date": parts[1].strip(),
                        "red_balls": sorted([int(parts[i].strip()) for i in range(2, 7)]),
                        "blue_balls": sorted([int(parts[i].strip()) for i in range(7, 9)]),
                    })
                except ValueError:
                    continue
    return draws


def import_csv_ssq(csv_path):
    """从CSV导入双色球数据
    CSV格式：期号,日期,红1,红2,红3,红4,红5,红6,蓝1
    """
    draws = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('期'):
                continue
            parts = line.split(',')
            if len(parts) >= 9:
                try:
                    draws.append({
                        "period": parts[0].strip(),
                        "date": parts[1].strip(),
                        "red_balls": sorted([int(parts[i].strip()) for i in range(2, 8)]),
                        "blue_ball": int(parts[8].strip()),
                    })
                except ValueError:
                    continue
    return draws


# ============================================================
# 数据合并与构建
# ============================================================

def merge_draws(existing_draws, new_draws):
    """合并新旧数据，按期号去重，按期号排序"""
    draw_map = {}
    for d in existing_draws:
        draw_map[d["period"]] = d
    new_count = 0
    for d in new_draws:
        if d["period"] not in draw_map:
            new_count += 1
        draw_map[d["period"]] = d
    all_draws = sorted(draw_map.values(), key=lambda x: x["period"])
    return all_draws, new_count


def build_dlt_data(draws):
    """构建大乐透完整数据结构"""
    return {
        "lottery_type": "daletou",
        "description": "大乐透历史开奖数据",
        "rules": {
            "red_balls": {"count": 5, "range": [1, 35]},
            "blue_balls": {"count": 2, "range": [1, 12]}
        },
        "draws": draws,
        "metadata": {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_draws": len(draws),
            "first_period": draws[0]["period"] if draws else "",
            "last_period": draws[-1]["period"] if draws else "",
            "data_source": "500.com datachart"
        }
    }


def build_ssq_data(draws):
    """构建双色球完整数据结构"""
    return {
        "lottery_type": "shuangseqiu",
        "description": "双色球历史开奖数据",
        "rules": {
            "red_balls": {"count": 6, "range": [1, 33]},
            "blue_ball": {"count": 1, "range": [1, 16]}
        },
        "draws": draws,
        "metadata": {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_draws": len(draws),
            "first_period": draws[0]["period"] if draws else "",
            "last_period": draws[-1]["period"] if draws else "",
            "data_source": "cwl.gov.cn + 500.com"
        }
    }


# ============================================================
# 主命令
# ============================================================

def cmd_full():
    """拉取全部历史数据"""
    print("=" * 50)
    print("开始拉取全部历史数据")
    print("=" * 50)

    # --- 大乐透 ---
    # 主数据源：500.com datachart 分段拉取
    dlt_draws = fetch_500_dlt_full()
    if not dlt_draws:
        print("500.com失败，尝试78500.cn（仅最近数据）...")
        dlt_draws = fetch_78500_dlt()
    if dlt_draws:
        data = build_dlt_data(dlt_draws)
        save_data(DATA_DIR / "daletou_history.json", data)
    else:
        print("警告：大乐透数据拉取失败，请检查网络或使用 --import-csv 手动导入")

    # --- 双色球 ---
    # 主数据源：福彩官方API（2013至今） + 500.com（2003-2012早期数据）
    ssq_draws_cwl = fetch_cwl_ssq()
    ssq_draws_500 = fetch_500_ssq_early()
    ssq_draws = []
    if ssq_draws_cwl:
        ssq_draws.extend(ssq_draws_cwl)
    if ssq_draws_500:
        ssq_draws.extend(ssq_draws_500)
    if not ssq_draws:
        print("所有数据源失败，尝试78500.cn（仅最近数据）...")
        ssq_draws = fetch_78500_ssq()
    if ssq_draws:
        # 去重排序
        draw_map = {}
        for d in ssq_draws:
            draw_map[d["period"]] = d
        ssq_draws = sorted(draw_map.values(), key=lambda x: x["period"])
        data = build_ssq_data(ssq_draws)
        save_data(DATA_DIR / "shuangseqiu_history.json", data)
    else:
        print("警告：双色球数据拉取失败，请检查网络或使用 --import-csv 手动导入")

    print("\n完成！")


def cmd_update():
    """更新最新数据（增量合并）"""
    print("=" * 50)
    print("开始更新最新数据")
    print("=" * 50)

    # --- 大乐透 ---
    existing_dlt = load_existing_data(DATA_DIR / "daletou_history.json")
    existing_dlt_draws = existing_dlt["draws"] if existing_dlt else []
    print(f"\n大乐透已有 {len(existing_dlt_draws)} 期数据")

    # 优先用78500（最新数据更快），备用500.com最近一段
    new_dlt = fetch_78500_dlt()
    if not new_dlt:
        # 拉取最近一年的数据作为增量
        current_year = datetime.now().strftime("%y")
        url = f"https://datachart.500.com/dlt/history/newinc/history.php?start={current_year}001&end={current_year}160"
        html = fetch_with_retry(url, timeout=45)
        if html:
            new_dlt = parse_500_dlt(html)
            print(f"  500.com增量: {len(new_dlt)}期")
    if new_dlt:
        merged, new_count = merge_draws(existing_dlt_draws, new_dlt)
        print(f"  新增 {new_count} 期数据")
        if new_count > 0 or not existing_dlt:
            data = build_dlt_data(merged)
            save_data(DATA_DIR / "daletou_history.json", data)
        else:
            print("  数据已是最新，无需更新")
    else:
        print("  更新失败，保留现有数据")

    # --- 双色球 ---
    existing_ssq = load_existing_data(DATA_DIR / "shuangseqiu_history.json")
    existing_ssq_draws = existing_ssq["draws"] if existing_ssq else []
    print(f"\n双色球已有 {len(existing_ssq_draws)} 期数据")

    # 优先用福彩官方API第1页（最新30期）
    new_ssq = None
    try:
        url = ("https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
               "?name=ssq&issueCount=30&pageNo=1&pageSize=30&systemType=PC")
        r = requests.get(url, headers=CWL_HEADERS, timeout=20)
        data = r.json()
        items = data.get('result', [])
        if items:
            new_ssq = []
            for item in items:
                reds = [int(x) for x in item['red'].split(',')]
                blue = int(item['blue'])
                date_str = item['date'].split('(')[0]
                new_ssq.append({
                    "period": item['code'],
                    "date": date_str,
                    "red_balls": sorted(reds),
                    "blue_ball": blue,
                })
            print(f"  福彩API增量: {len(new_ssq)}期")
    except Exception as e:
        print(f"  福彩API失败: {e}")

    if not new_ssq:
        new_ssq = fetch_78500_ssq()
    if new_ssq:
        merged, new_count = merge_draws(existing_ssq_draws, new_ssq)
        print(f"  新增 {new_count} 期数据")
        if new_count > 0 or not existing_ssq:
            data = build_ssq_data(merged)
            save_data(DATA_DIR / "shuangseqiu_history.json", data)
        else:
            print("  数据已是最新，无需更新")
    else:
        print("  更新失败，保留现有数据")

    print("\n完成！")


def cmd_import_csv(dlt_csv=None, ssq_csv=None):
    """从CSV文件导入数据"""
    print("=" * 50)
    print("从CSV导入数据")
    print("=" * 50)

    if dlt_csv:
        print(f"\n导入大乐透: {dlt_csv}")
        existing = load_existing_data(DATA_DIR / "daletou_history.json")
        existing_draws = existing["draws"] if existing else []
        new_draws = import_csv_dlt(dlt_csv)
        print(f"  CSV中有 {len(new_draws)} 期数据")
        merged, new_count = merge_draws(existing_draws, new_draws)
        print(f"  新增 {new_count} 期")
        data = build_dlt_data(merged)
        save_data(DATA_DIR / "daletou_history.json", data)

    if ssq_csv:
        print(f"\n导入双色球: {ssq_csv}")
        existing = load_existing_data(DATA_DIR / "shuangseqiu_history.json")
        existing_draws = existing["draws"] if existing else []
        new_draws = import_csv_ssq(ssq_csv)
        print(f"  CSV中有 {len(new_draws)} 期数据")
        merged, new_count = merge_draws(existing_draws, new_draws)
        print(f"  新增 {new_count} 期")
        data = build_ssq_data(merged)
        save_data(DATA_DIR / "shuangseqiu_history.json", data)

    print("\n完成！")


def cmd_status():
    """查看当前数据状态"""
    print("=" * 50)
    print("当前数据状态")
    print("=" * 50)

    for name, filename in [("大乐透", "daletou_history.json"), ("双色球", "shuangseqiu_history.json")]:
        filepath = DATA_DIR / filename
        data = load_existing_data(filepath)
        if data:
            meta = data["metadata"]
            print(f"\n{name}:")
            print(f"  总期数: {meta['total_draws']}")
            print(f"  首期: {meta.get('first_period', 'N/A')}")
            print(f"  末期: {meta.get('last_period', 'N/A')}")
            print(f"  更新时间: {meta['last_updated']}")
            print(f"  数据源: {meta.get('data_source', 'N/A')}")
        else:
            print(f"\n{name}: 无数据")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="彩票历史数据拉取与更新工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full', action='store_true', help='拉取全部历史数据')
    group.add_argument('--update', action='store_true', help='更新最新数据')
    group.add_argument('--import-csv', nargs='*', metavar='FILE', help='从CSV导入（可指定文件路径）')
    group.add_argument('--status', action='store_true', help='查看当前数据状态')

    args = parser.parse_args()

    if args.full:
        cmd_full()
    elif args.update:
        cmd_update()
    elif args.import_csv is not None:
        dlt_csv = None
        ssq_csv = None
        for f in args.import_csv:
            if 'dlt' in f.lower() or '大乐透' in f:
                dlt_csv = f
            elif 'ssq' in f.lower() or '双色球' in f:
                ssq_csv = f
            else:
                print(f"无法识别文件类型: {f}（文件名需包含 dlt/ssq/大乐透/双色球）")
        if dlt_csv or ssq_csv:
            cmd_import_csv(dlt_csv, ssq_csv)
        else:
            print("请提供CSV文件路径，文件名需包含 dlt 或 ssq 以区分类型")
            print("CSV格式：")
            print("  大乐透: 期号,日期,红1,红2,红3,红4,红5,蓝1,蓝2")
            print("  双色球: 期号,日期,红1,红2,红3,红4,红5,红6,蓝1")
    elif args.status:
        cmd_status()


if __name__ == "__main__":
    main()
