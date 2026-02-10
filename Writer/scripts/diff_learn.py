#!/usr/bin/env python3
"""文风学习工具：对比原稿与修改稿，提取修改模式，更新修正规则库。

使用方式：
  python3 scripts/diff_learn.py diff <原稿> <修改稿>     # 生成diff分析报告
  python3 scripts/diff_learn.py learn <原稿> <修改稿>    # 分析并写入修正规则库
  python3 scripts/diff_learn.py consult <修改稿>         # 风格顾问：与参考库比对给建议
"""

import os
import sys
import re
import json
import difflib
import argparse
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ROLE_DIR = os.path.join(PROJECT_ROOT, 'roles', 'black_crow')
CORRECTIONS_FILE = os.path.join(ROLE_DIR, 'corrections.md')
STYLE_GUIDE_FILE = os.path.join(ROLE_DIR, 'style_guide.md')
REF_DIR = os.path.join(PROJECT_ROOT, 'references', 'black_crow')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def read_file(path):
    """读取文本文件"""
    for enc in ['utf-8', 'gbk', 'gb18030']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    print(f'[错误] 无法读取: {path}')
    return ''


def split_sentences(text):
    """按句子切分文本"""
    parts = re.split(r'([。！？…\n])', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = (parts[i] + parts[i + 1]).strip()
        if s:
            sentences.append(s)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences


def split_words(text):
    """简易中文分词（按字符+标点切分，用于词频统计）"""
    # 提取连续中文字符块（2-6字词组）和标点
    words = re.findall(r'[\u4e00-\u9fff]{2,6}|[a-zA-Z]+', text)
    return words


def analyze_diff(original, modified):
    """对比原稿与修改稿，提取修改模式"""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)

    replacements = []   # 词汇替换
    deletions = []      # 删除内容
    additions = []      # 新增内容
    restructures = []   # 句式重构

    matcher = difflib.SequenceMatcher(None, orig_lines, mod_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        orig_chunk = ''.join(orig_lines[i1:i2]).strip()
        mod_chunk = ''.join(mod_lines[j1:j2]).strip()

        if tag == 'delete':
            if orig_chunk:
                deletions.append(orig_chunk)
        elif tag == 'insert':
            if mod_chunk:
                additions.append(mod_chunk)
        elif tag == 'replace':
            if not orig_chunk and mod_chunk:
                additions.append(mod_chunk)
            elif orig_chunk and not mod_chunk:
                deletions.append(orig_chunk)
            else:
                # 判断是词汇替换还是句式重构
                ratio = difflib.SequenceMatcher(
                    None, orig_chunk, mod_chunk).ratio()
                if ratio > 0.5:
                    replacements.append((orig_chunk, mod_chunk))
                else:
                    restructures.append((orig_chunk, mod_chunk))

    return {
        'replacements': replacements,
        'deletions': deletions,
        'additions': additions,
        'restructures': restructures,
    }


def print_report(result):
    """打印diff分析报告"""
    print('\n' + '=' * 50)
    print(' 修改模式分析报告')
    print('=' * 50)

    if result['replacements']:
        print(f'\n## 词汇/短语替换 ({len(result["replacements"])} 处)')
        for orig, mod in result['replacements']:
            orig_s = orig[:60].replace('\n', ' ')
            mod_s = mod[:60].replace('\n', ' ')
            print(f'  - "{orig_s}" → "{mod_s}"')

    if result['deletions']:
        print(f'\n## 删除内容 ({len(result["deletions"])} 处)')
        for d in result['deletions']:
            print(f'  - 删除: "{d[:80].replace(chr(10), " ")}"')

    if result['additions']:
        print(f'\n## 新增内容 ({len(result["additions"])} 处)')
        for a in result['additions']:
            print(f'  + 新增: "{a[:80].replace(chr(10), " ")}"')

    if result['restructures']:
        print(f'\n## 句式重构 ({len(result["restructures"])} 处)')
        for orig, mod in result['restructures']:
            print(f'  原: "{orig[:60].replace(chr(10), " ")}"')
            print(f'  改: "{mod[:60].replace(chr(10), " ")}"')
            print()

    total = sum(len(v) for v in result.values())
    print(f'\n[统计] 共 {total} 处修改')
    return total


def append_corrections(result):
    """将分析结果追加到修正规则库"""
    if not os.path.exists(CORRECTIONS_FILE):
        print(f'[错误] 修正规则库不存在: {CORRECTIONS_FILE}')
        return

    lines = []

    # 词汇替换 → 追加到黑名单表格
    for orig, mod in result['replacements']:
        orig_s = orig[:40].replace('\n', ' ').replace('|', '/')
        mod_s = mod[:40].replace('\n', ' ').replace('|', '/')
        lines.append(f'| {orig_s} | {mod_s} | diff_learn |')

    # 删除模式
    del_lines = []
    for d in result['deletions']:
        d_s = d[:60].replace('\n', ' ')
        del_lines.append(f'- 删除类型: "{d_s}"')

    # 新增偏好
    add_lines = []
    for a in result['additions']:
        a_s = a[:60].replace('\n', ' ')
        add_lines.append(f'- 用户偏好新增: "{a_s}"')

    content = read_file(CORRECTIONS_FILE)

    # 追加词汇替换
    if lines:
        marker = '| （待学习积累） | | |'
        if marker in content:
            content = content.replace(marker, '\n'.join(lines))
        else:
            # 在黑名单表格末尾追加
            content = content.replace(
                '\n## 二、句式规则',
                '\n' + '\n'.join(lines) + '\n\n## 二、句式规则')

    # 追加删除模式
    if del_lines:
        marker = '- （待学习积累）\n\n## 四'
        if marker in content:
            content = content.replace(marker,
                                      '\n'.join(del_lines) + '\n\n## 四')

    # 追加新增偏好
    if add_lines:
        marker = '- （待学习积累）\n\n## 五'
        if marker in content:
            content = content.replace(marker,
                                      '\n'.join(add_lines) + '\n\n## 五')

    with open(CORRECTIONS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    total = len(lines) + len(del_lines) + len(add_lines)
    print(f'[学习] 已写入 {total} 条规则到 corrections.md')


def load_reference_samples(author=None, max_chars=5000):
    """从参考库加载文本样本片段"""
    samples = {}
    for dirname in os.listdir(REF_DIR):
        dirpath = os.path.join(REF_DIR, dirname)
        if not os.path.isdir(dirpath):
            continue
        if author and dirname != author:
            continue
        texts = []
        # 支持直接放md或放在md子目录
        search_dirs = [dirpath]
        md_sub = os.path.join(dirpath, 'md')
        if os.path.isdir(md_sub):
            search_dirs.append(md_sub)
        for sd in search_dirs:
            for fname in os.listdir(sd):
                if not fname.endswith('.md'):
                    continue
                content = read_file(os.path.join(sd, fname))
                if content:
                    # 取正文片段，跳过开头元信息
                    body = content[200:max_chars + 200]
                    texts.append(body)
        if texts:
            samples[dirname] = texts
    return samples


def style_stats(text):
    """计算文本的风格统计指标"""
    sentences = split_sentences(text)
    if not sentences:
        return {}
    lengths = [len(s) for s in sentences]
    words = split_words(text)
    word_freq = Counter(words)

    return {
        'avg_sentence_len': sum(lengths) / len(lengths),
        'max_sentence_len': max(lengths),
        'min_sentence_len': min(lengths),
        'sentence_count': len(sentences),
        'top_words': word_freq.most_common(20),
        'dialogue_ratio': sum(1 for s in sentences
                              if '"' in s or '"' in s or '「' in s)
                          / len(sentences),
    }


def consult_style(modified_text):
    """风格顾问：将修改稿与参考库做风格统计对比"""
    mod_stats = style_stats(modified_text)
    samples = load_reference_samples()

    print('\n' + '=' * 50)
    print(' 风格顾问分析报告')
    print('=' * 50)

    print(f'\n### 修改稿统计')
    print(f'  平均句长: {mod_stats["avg_sentence_len"]:.1f} 字')
    print(f'  最长句: {mod_stats["max_sentence_len"]} 字')
    print(f'  最短句: {mod_stats["min_sentence_len"]} 字')
    print(f'  对话占比: {mod_stats["dialogue_ratio"]:.1%}')
    print(f'  高频词: {", ".join(w for w, _ in mod_stats["top_words"][:10])}')

    for author, texts in samples.items():
        combined = '\n'.join(texts)
        ref_stats = style_stats(combined)
        if not ref_stats:
            continue
        print(f'\n### 参考: {author}')
        print(f'  平均句长: {ref_stats["avg_sentence_len"]:.1f} 字')
        print(f'  对话占比: {ref_stats["dialogue_ratio"]:.1%}')
        print(f'  高频词: {", ".join(w for w, _ in ref_stats["top_words"][:10])}')

        # 给出差异建议
        diff_len = mod_stats['avg_sentence_len'] - ref_stats['avg_sentence_len']
        if abs(diff_len) > 5:
            direction = '偏长' if diff_len > 0 else '偏短'
            print(f'  ⚠ 句长{direction} {abs(diff_len):.0f} 字，'
                  f'建议向{author}靠拢')

        diff_dial = mod_stats['dialogue_ratio'] - ref_stats['dialogue_ratio']
        if abs(diff_dial) > 0.1:
            direction = '偏多' if diff_dial > 0 else '偏少'
            print(f'  ⚠ 对话占比{direction}，'
                  f'{author}为 {ref_stats["dialogue_ratio"]:.0%}')

    print('\n[提示] 以上为统计层面的对比，'
          '深层风格建议请在 Claude 对话中使用 /role 黑乌鸦 后请求审阅')


def main():
    parser = argparse.ArgumentParser(description='文风学习工具')
    sub = parser.add_subparsers(dest='command')

    # diff 命令
    p_diff = sub.add_parser('diff', help='对比原稿与修改稿')
    p_diff.add_argument('original', help='原稿文件路径')
    p_diff.add_argument('modified', help='修改稿文件路径')

    # learn 命令
    p_learn = sub.add_parser('learn', help='分析差异并写入修正规则库')
    p_learn.add_argument('original', help='原稿文件路径')
    p_learn.add_argument('modified', help='修改稿文件路径')

    # consult 命令
    p_consult = sub.add_parser('consult', help='风格顾问：与参考库比对')
    p_consult.add_argument('modified', help='修改稿文件路径')

    args = parser.parse_args()

    if args.command == 'diff':
        orig = read_file(args.original)
        mod = read_file(args.modified)
        result = analyze_diff(orig, mod)
        print_report(result)

    elif args.command == 'learn':
        orig = read_file(args.original)
        mod = read_file(args.modified)
        result = analyze_diff(orig, mod)
        print_report(result)
        append_corrections(result)

    elif args.command == 'consult':
        mod = read_file(args.modified)
        consult_style(mod)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
