#!/usr/bin/env python3
"""批注解析工具：提取修改稿中的批注，生成结构化报告，支持增强diff。

批注语法（两种格式均支持）：
  带类型：<!--[理由：...]-->  <!--[不满：...]-->  <!--[求助：...]-->
  普通：  <!--...-->  自动归类为"批注"

使用方式：
  python3 scripts/annotation_parser.py parse <修改稿>              # 输出批注报告
  python3 scripts/annotation_parser.py strip <修改稿>              # 生成去批注的 _clean.md
  python3 scripts/annotation_parser.py enrich <原稿> <修改稿>      # 增强diff报告（diff+批注关联）
"""

import os
import re
import argparse
import difflib
from dataclasses import dataclass

# 复用 diff_learn 的文件读取和diff分析
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from diff_learn import read_file, analyze_diff

ANNOTATION_PATTERN = re.compile(r'<!--\[(理由|不满|求助)：(.*?)\]-->', re.DOTALL)
# 普通注释格式：<!--...-->（排除已被上面匹配的带类型批注）
PLAIN_COMMENT_PATTERN = re.compile(r'<!--(.*?)-->', re.DOTALL)


@dataclass
class Annotation:
    type: str       # "理由" | "不满" | "求助"
    content: str    # 批注正文
    line_no: int    # 行号（从1开始）
    context: str    # 关联段落/句子
    raw: str        # 原始匹配串


def extract_context(lines, line_idx, max_chars=200):
    """从批注所在行向前回溯到空行，收集段落上下文"""
    # 用两种正则都清理掉批注
    clean_re = lambda s: PLAIN_COMMENT_PATTERN.sub('', ANNOTATION_PATTERN.sub('', s))
    parts = []
    current = clean_re(lines[line_idx]).strip()
    if current:
        parts.append(current)
    for i in range(line_idx - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            break
        parts.insert(0, line)
    text = '\n'.join(parts).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def parse_annotations(text):
    """正则提取所有批注，计算行号和关联上下文。支持带类型和普通注释两种格式。"""
    lines = text.splitlines()
    annotations = []
    # 记录已匹配的位置区间，避免普通注释重复匹配带类型批注
    matched_spans = set()

    # 第一轮：带类型的批注 <!--[理由：...]-->
    for match in ANNOTATION_PATTERN.finditer(text):
        raw = match.group(0)
        ann_type = match.group(1)
        content = match.group(2).strip()
        line_no = text[:match.start()].count('\n') + 1
        line_idx = min(line_no - 1, len(lines) - 1)
        context = extract_context(lines, line_idx)
        annotations.append(Annotation(
            type=ann_type, content=content, line_no=line_no,
            context=context, raw=raw,
        ))
        matched_spans.add((match.start(), match.end()))

    # 第二轮：普通注释 <!--...-->，归类为"批注"
    for match in PLAIN_COMMENT_PATTERN.finditer(text):
        if (match.start(), match.end()) in matched_spans:
            continue
        raw = match.group(0)
        content = match.group(1).strip()
        if not content:
            continue
        line_no = text[:match.start()].count('\n') + 1
        line_idx = min(line_no - 1, len(lines) - 1)
        context = extract_context(lines, line_idx)
        annotations.append(Annotation(
            type='批注', content=content, line_no=line_no,
            context=context, raw=raw,
        ))

    # 按行号排序
    annotations.sort(key=lambda a: a.line_no)
    return annotations


def strip_annotations(text):
    """去除所有批注标记（带类型+普通注释），清理多余空行"""
    cleaned = ANNOTATION_PATTERN.sub('', text)
    cleaned = PLAIN_COMMENT_PATTERN.sub('', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned


def print_annotation_report(annotations):
    """按类型分组输出批注报告"""
    if not annotations:
        print('[信息] 未发现批注')
        return

    print('\n' + '=' * 50)
    print(' 批注解析报告')
    print('=' * 50)

    # 按类型分组
    grouped = {}
    for ann in annotations:
        grouped.setdefault(ann.type, []).append(ann)

    type_labels = {'理由': '修改理由', '不满': '不满/待改进', '求助': '求助/拿不准', '批注': '普通批注'}

    for t in ['理由', '不满', '求助', '批注']:
        items = grouped.get(t, [])
        if not items:
            continue
        label = type_labels[t]
        print(f'\n## {label} ({len(items)} 处)')
        for ann in items:
            print(f'  [L{ann.line_no}] {ann.content}')
            if ann.context:
                ctx = ann.context[:80].replace('\n', ' ')
                print(f'       上下文: "{ctx}"')

    print(f'\n[统计] 共 {len(annotations)} 条批注'
          f'（理由 {len(grouped.get("理由", []))},'
          f' 不满 {len(grouped.get("不满", []))},'
          f' 求助 {len(grouped.get("求助", []))},'
          f' 普通 {len(grouped.get("批注", []))}）')


def enrich_diff(original, modified):
    """增强diff报告：diff条目旁附上用户批注"""
    annotations = parse_annotations(modified)
    clean = strip_annotations(modified)
    result = analyze_diff(original, clean)

    print('\n' + '=' * 50)
    print(' 增强Diff报告（含批注关联）')
    print('=' * 50)

    # 收集所有 diff 条目（替换+重构）
    diff_items = []
    for orig, mod in result['replacements']:
        diff_items.append(('替换', orig, mod))
    for orig, mod in result['restructures']:
        diff_items.append(('重构', orig, mod))
    for d in result['deletions']:
        diff_items.append(('删除', d, ''))
    for a in result['additions']:
        diff_items.append(('新增', '', a))

    if not diff_items:
        print('\n[信息] 无差异')
        return

    for kind, orig, mod in diff_items:
        print(f'\n--- [{kind}] ---')
        if orig:
            print(f'  原: "{orig[:80].replace(chr(10), " ")}"')
        if mod:
            print(f'  改: "{mod[:80].replace(chr(10), " ")}"')

        # 模糊匹配关联批注
        target = mod if mod else orig
        matched = []
        for ann in annotations:
            ratio = difflib.SequenceMatcher(
                None, target, ann.context).ratio()
            if ratio > 0.6:
                matched.append(ann)

        if matched:
            for ann in matched:
                print(f'  📝 [{ann.type}] {ann.content}')
        else:
            print(f'  （无关联批注）')

    total = len(diff_items)
    linked = sum(1 for kind, orig, mod in diff_items
                 if any(difflib.SequenceMatcher(
                     None, mod if mod else orig, a.context).ratio() > 0.6
                        for a in annotations))
    print(f'\n[统计] {total} 处差异，{linked} 处有批注关联，'
          f'{len(annotations)} 条批注')


def main():
    parser = argparse.ArgumentParser(description='批注解析工具')
    sub = parser.add_subparsers(dest='command')

    # parse 命令
    p_parse = sub.add_parser('parse', help='解析批注并输出报告')
    p_parse.add_argument('modified', help='带批注的修改稿')

    # strip 命令
    p_strip = sub.add_parser('strip', help='去除批注生成干净版本')
    p_strip.add_argument('modified', help='带批注的修改稿')

    # enrich 命令
    p_enrich = sub.add_parser('enrich', help='增强diff报告（diff+批注）')
    p_enrich.add_argument('original', help='原稿文件路径')
    p_enrich.add_argument('modified', help='带批注的修改稿')

    args = parser.parse_args()

    if args.command == 'parse':
        text = read_file(args.modified)
        annotations = parse_annotations(text)
        print_annotation_report(annotations)

    elif args.command == 'strip':
        text = read_file(args.modified)
        clean = strip_annotations(text)
        # 生成 _clean.md 文件
        base, ext = os.path.splitext(args.modified)
        out_path = base + '_clean' + ext
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(clean)
        print(f'[完成] 已生成: {out_path}')

    elif args.command == 'enrich':
        orig = read_file(args.original)
        mod = read_file(args.modified)
        enrich_diff(orig, mod)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
