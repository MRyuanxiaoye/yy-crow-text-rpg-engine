#!/usr/bin/env python3
"""通用作品拉取脚本 - 从URL抓取文章内容并保存为Markdown"""

import argparse
import os
import re
import time
import json
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from readability import Document


# 默认请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

# 请求间隔（秒），避免被封
REQUEST_DELAY = 1.5


def fetch_page(url, encoding=None):
    """拉取页面HTML内容"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if encoding:
            resp.encoding = encoding
        else:
            resp.encoding = resp.apparent_encoding
        return resp.text
    except requests.RequestException as e:
        print(f"[错误] 请求失败: {url} - {e}")
        return None


def extract_article(html, url):
    """使用readability提取正文内容"""
    doc = Document(html, url=url)
    title = doc.title()
    # 提取正文HTML，再转为纯文本
    content_html = doc.summary()
    soup = BeautifulSoup(content_html, 'lxml')
    content = soup.get_text(separator='\n', strip=True)
    return title, content


def html_to_text(html):
    """将HTML转为清洁文本"""
    soup = BeautifulSoup(html, 'lxml')
    # 移除script和style标签
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)


def sanitize_filename(name):
    """清理文件名中的非法字符"""
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = name.strip()[:80]
    return name if name else 'untitled'


def save_article(title, content, output_dir, source_url=None):
    """将文章保存为Markdown文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = sanitize_filename(title) + '.md'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'# {title}\n\n')
        if source_url:
            f.write(f'> 来源: {source_url}\n\n')
        f.write(content)

    print(f"[完成] 已保存: {filepath}")
    return filepath


def extract_links(html, base_url, link_selector=None):
    """从目录页提取文章链接列表"""
    soup = BeautifulSoup(html, 'lxml')
    links = []

    if link_selector:
        elements = soup.select(link_selector)
    else:
        # 默认提取正文区域内的所有链接
        content_area = soup.find('main') or soup.find('article') or soup.find('body')
        elements = content_area.find_all('a', href=True) if content_area else []

    for a in elements:
        href = a.get('href', '')
        if not href or href.startswith('#') or href.startswith('javascript:'):
            continue
        full_url = urljoin(base_url, href)
        title = a.get_text(strip=True) or '未命名'
        links.append({'title': title, 'url': full_url})

    return links


def fetch_single(url, output_dir, encoding=None):
    """抓取单篇文章"""
    print(f"[拉取] 正在抓取: {url}")
    html = fetch_page(url, encoding)
    if not html:
        return None
    title, content = extract_article(html, url)
    if not content.strip():
        print(f"[警告] 未提取到正文内容，尝试备用方式...")
        content = html_to_text(html)
    if not content.strip():
        print(f"[错误] 无法提取内容: {url}")
        return None
    return save_article(title, content, output_dir, source_url=url)


def fetch_list(url, output_dir, encoding=None, selector=None, limit=0):
    """抓取目录页下的所有文章"""
    print(f"[拉取] 正在解析目录页: {url}")
    html = fetch_page(url, encoding)
    if not html:
        return []

    links = extract_links(html, url, link_selector=selector)
    if not links:
        print("[警告] 未找到文章链接，请尝试指定 --selector 参数")
        return []

    print(f"[信息] 发现 {len(links)} 个链接")
    if limit > 0:
        links = links[:limit]
        print(f"[信息] 限制抓取前 {limit} 篇")

    saved = []
    for i, link in enumerate(links, 1):
        print(f"\n[{i}/{len(links)}] {link['title']}")
        result = fetch_single(link['url'], output_dir, encoding)
        if result:
            saved.append(result)
        time.sleep(REQUEST_DELAY)

    print(f"\n[完成] 共保存 {len(saved)}/{len(links)} 篇文章")
    return saved


def main():
    parser = argparse.ArgumentParser(description='通用作品拉取脚本')
    parser.add_argument('--url', required=True, help='目标URL（单篇文章或目录页）')
    parser.add_argument('--output', default='references/black_crow',
                        help='输出目录（默认: references/black_crow）')
    parser.add_argument('--mode', choices=['single', 'list'], default='single',
                        help='抓取模式: single=单篇, list=目录页批量（默认: single）')
    parser.add_argument('--encoding', default=None,
                        help='指定页面编码（如 utf-8, gbk），默认自动检测')
    parser.add_argument('--selector', default=None,
                        help='CSS选择器，用于从目录页提取文章链接（list模式）')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制最大抓取篇数（0=不限制）')

    args = parser.parse_args()

    # 将相对路径转为基于项目根目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, args.output)

    if args.mode == 'single':
        fetch_single(args.url, output_dir, args.encoding)
    else:
        fetch_list(args.url, output_dir, args.encoding, args.selector, args.limit)


if __name__ == '__main__':
    main()
