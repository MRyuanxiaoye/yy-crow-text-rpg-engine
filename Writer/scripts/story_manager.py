#!/usr/bin/env python3
"""故事参考库管理脚本 - 索引管理、按需拉取、自动清理"""

import argparse
import json
import os
import shutil
import sys
import time

# 项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STORIES_DIR = os.path.join(PROJECT_ROOT, 'stories')
INDEX_FILE = os.path.join(STORIES_DIR, 'index.json')
CACHE_DIR = os.path.join(STORIES_DIR, 'cache')
CUSTOM_DIR = os.path.join(STORIES_DIR, 'custom')


def load_index():
    """加载故事索引"""
    if not os.path.exists(INDEX_FILE):
        return {'stories': []}
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_index(data):
    """保存故事索引"""
    os.makedirs(STORIES_DIR, exist_ok=True)
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[完成] 索引已更新: {INDEX_FILE}")


def next_id(data):
    """生成下一个故事ID"""
    if not data['stories']:
        return 1
    return max(s['id'] for s in data['stories']) + 1


def cmd_search(keyword):
    """按关键词搜索故事"""
    data = load_index()
    results = []
    for s in data['stories']:
        # 在标题、标签、摘要中搜索
        searchable = s['title'] + ' '.join(s.get('tags', [])) + s.get('summary', '')
        if keyword in searchable:
            results.append(s)

    if not results:
        print(f"[搜索] 未找到与「{keyword}」相关的故事")
        return

    print(f"[搜索] 找到 {len(results)} 个相关故事:\n")
    for s in results:
        tags = ', '.join(s.get('tags', []))
        print(f"  [{s['id']}] {s['title']}")
        print(f"      标签: {tags}")
        print(f"      摘要: {s.get('summary', '无')[:60]}")
        if s.get('url'):
            print(f"      URL: {s['url']}")
        print()


def cmd_add(title, url, tags, summary, source=''):
    """添加故事到索引"""
    data = load_index()
    story = {
        'id': next_id(data),
        'title': title,
        'tags': [t.strip() for t in tags.split(',') if t.strip()],
        'url': url,
        'source': source,
        'summary': summary,
    }
    data['stories'].append(story)
    save_index(data)
    print(f"[添加] 已添加故事: [{story['id']}] {title}")


def cmd_fetch(story_id):
    """按需拉取故事全文到缓存"""
    # 延迟导入fetcher，避免循环依赖
    from fetcher import fetch_page, extract_article, html_to_text, sanitize_filename

    data = load_index()
    story = None
    for s in data['stories']:
        if s['id'] == story_id:
            story = s
            break

    if not story:
        print(f"[错误] 未找到ID为 {story_id} 的故事")
        return None

    if not story.get('url'):
        # 检查是否有本地自定义文件
        local_path = os.path.join(CUSTOM_DIR, sanitize_filename(story['title']) + '.md')
        if os.path.exists(local_path):
            print(f"[信息] 本地故事: {local_path}")
            return local_path
        print(f"[错误] 故事「{story['title']}」没有URL，也没有本地文件")
        return None

    print(f"[拉取] 正在获取: {story['title']}")
    html = fetch_page(story['url'])
    if not html:
        return None

    title, content = extract_article(html, story['url'])
    if not content.strip():
        content = html_to_text(html)
    if not content.strip():
        print(f"[错误] 无法提取内容")
        return None

    os.makedirs(CACHE_DIR, exist_ok=True)
    filename = sanitize_filename(story['title']) + '.md'
    filepath = os.path.join(CACHE_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {story['title']}\n\n")
        f.write(f"> 来源: {story.get('source', '')} | {story['url']}\n\n")
        f.write(content)

    print(f"[完成] 已缓存到: {filepath}")
    return filepath


def cmd_clean():
    """清理缓存目录"""
    if not os.path.exists(CACHE_DIR):
        print("[信息] 缓存目录为空，无需清理")
        return
    files = os.listdir(CACHE_DIR)
    if not files:
        print("[信息] 缓存目录为空，无需清理")
        return
    shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[清理] 已删除 {len(files)} 个缓存文件")


def cmd_tags():
    """列出所有标签及其故事数量"""
    data = load_index()
    tag_count = {}
    for s in data['stories']:
        for tag in s.get('tags', []):
            tag_count[tag] = tag_count.get(tag, 0) + 1

    if not tag_count:
        print("[信息] 索引中暂无标签")
        return

    print("[标签列表]\n")
    for tag, count in sorted(tag_count.items(), key=lambda x: -x[1]):
        print(f"  {tag} ({count}篇)")


def cmd_list():
    """列出所有故事"""
    data = load_index()
    if not data['stories']:
        print("[信息] 索引为空，请先使用 add 命令添加故事")
        return

    print(f"[故事列表] 共 {len(data['stories'])} 篇\n")
    for s in data['stories']:
        tags = ', '.join(s.get('tags', []))
        print(f"  [{s['id']}] {s['title']}  ({tags})")


def cmd_remove(story_id):
    """从索引中删除故事"""
    data = load_index()
    original_len = len(data['stories'])
    data['stories'] = [s for s in data['stories'] if s['id'] != story_id]
    if len(data['stories']) == original_len:
        print(f"[错误] 未找到ID为 {story_id} 的故事")
        return
    save_index(data)
    print(f"[删除] 已移除故事 ID={story_id}")


def main():
    parser = argparse.ArgumentParser(description='故事参考库管理工具')
    sub = parser.add_subparsers(dest='command', help='可用命令')

    # search 命令
    p_search = sub.add_parser('search', help='按关键词搜索故事')
    p_search.add_argument('keyword', help='搜索关键词')

    # add 命令
    p_add = sub.add_parser('add', help='添加故事到索引')
    p_add.add_argument('--title', required=True, help='故事标题')
    p_add.add_argument('--url', default='', help='故事URL')
    p_add.add_argument('--tags', default='', help='标签，逗号分隔')
    p_add.add_argument('--summary', default='', help='故事摘要')
    p_add.add_argument('--source', default='', help='来源（如：史记）')

    # fetch 命令
    p_fetch = sub.add_parser('fetch', help='按需拉取故事全文到缓存')
    p_fetch.add_argument('id', type=int, help='故事ID')

    # clean 命令
    sub.add_parser('clean', help='清理缓存目录')

    # tags 命令
    sub.add_parser('tags', help='列出所有标签')

    # list 命令
    sub.add_parser('list', help='列出所有故事')

    # remove 命令
    p_rm = sub.add_parser('remove', help='从索引中删除故事')
    p_rm.add_argument('id', type=int, help='故事ID')

    args = parser.parse_args()

    if args.command == 'search':
        cmd_search(args.keyword)
    elif args.command == 'add':
        cmd_add(args.title, args.url, args.tags, args.summary, args.source)
    elif args.command == 'fetch':
        cmd_fetch(args.id)
    elif args.command == 'clean':
        cmd_clean()
    elif args.command == 'tags':
        cmd_tags()
    elif args.command == 'list':
        cmd_list()
    elif args.command == 'remove':
        cmd_remove(args.id)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
