#!/usr/bin/env python3
"""网文分析器 - 多模型分层压缩网文为结构化参考资料"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径常量
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
NOVELS_DIR = PROJECT_DIR / "novels"
INDEX_FILE = NOVELS_DIR / "index.json"
ENV_FILE = Path("/Users/yuanye/Documents/深度搜索/.env")

# 卷大小（每卷包含的章节数）
VOLUME_SIZE = 50
# 并发批次大小
BATCH_SIZE = 20

# ============================================================
# 模型路由配置
# ============================================================
MODEL_ROUTES = {
    "chapter_summary":  {"provider": "deepseek", "model": "deepseek-chat"},
    "chapter_detail":   {"provider": "deepseek", "model": "deepseek-chat"},
    "volume_summary":   {"provider": "deepseek", "model": "deepseek-chat"},
    "worldbuilding":    {"provider": "deepseek", "model": "deepseek-chat"},
    "power_system":     {"provider": "deepseek", "model": "deepseek-chat"},
    "characters":       {"provider": "openai",   "model": "gpt-4.1"},
    "side_characters":  {"provider": "openai",   "model": "gpt-4.1"},
    "book_summary":     {"provider": "openai",   "model": "gpt-4.1"},
    "plot_main":        {"provider": "openai",   "model": "gpt-4.1"},
    "plot_sub":         {"provider": "openai",   "model": "gpt-4.1"},
    "pacing":           {"provider": "openai",   "model": "gpt-4.1"},
}

# ============================================================
# 环境变量 & API 客户端
# ============================================================
def load_env():
    """从 .env 文件加载环境变量"""
    if not ENV_FILE.exists():
        print(f"[错误] 找不到 .env 文件: {ENV_FILE}")
        sys.exit(1)
    with open(ENV_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def get_client(provider: str):
    """获取 OpenAI 兼容客户端"""
    from openai import OpenAI
    if provider == "deepseek":
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            timeout=300.0,
        )
    else:
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=300.0)


_clients = {}

def llm_call(task_type: str, prompt: str, max_retries: int = 3) -> str:
    """统一 LLM 调用，根据任务类型自动路由模型"""
    route = MODEL_ROUTES[task_type]
    provider = route["provider"]
    model = route["model"]

    if provider not in _clients:
        _clients[provider] = get_client(provider)
    client = _clients[provider]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [重试] {provider}/{model} 调用失败: {e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  [错误] {provider}/{model} 调用失败: {e}")
                raise


# ============================================================
# 文件读取（多编码）
# ============================================================
def read_file(path):
    """多编码尝试读取文本文件"""
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    print(f"[错误] 无法读取: {path}")
    return ""


# ============================================================
# 索引 CRUD
# ============================================================
def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"novels": []}


def save_index(data):
    NOVELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def next_id(data):
    if not data["novels"]:
        return 1
    return max(n["id"] for n in data["novels"]) + 1


def find_novel(data, novel_id: int):
    for n in data["novels"]:
        if n["id"] == novel_id:
            return n
    return None


def get_novel_dir(novel_id: int, slug: str) -> Path:
    return NOVELS_DIR / f"{novel_id:03d}_{slug}"


def load_meta(novel_dir: Path) -> dict:
    meta_file = novel_dir / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(novel_dir: Path, meta: dict):
    meta_file = novel_dir / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ============================================================
# 章节切分
# ============================================================
# 章节标题正则
CHAPTER_RE = re.compile(
    r"^\s*(?:"
    r"第[零一二三四五六七八九十百千万\d]+[章节回]"
    r"|[Cc]hapter\s+\d+"
    r"|楔子|序章|序|前言|引子|尾声|番外"
    r")",
    re.MULTILINE,
)


def split_txt(file_path: str) -> list[dict]:
    """切分 txt 文件为章节列表，返回 [{"title": ..., "content": ...}, ...]"""
    text = read_file(file_path)
    if not text:
        return []

    matches = list(CHAPTER_RE.finditer(text))
    if not matches:
        # 无章节标题，按 5000 字切分
        print("  [提示] 未检测到章节标题，按 5000 字自动切分")
        chunks = []
        for i in range(0, len(text), 5000):
            chunk = text[i : i + 5000].strip()
            if chunk:
                chunks.append({"title": f"第{len(chunks)+1}段", "content": chunk})
        return chunks

    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if not block:
            continue
        # 提取标题（第一行）
        first_line = block.split("\n", 1)[0].strip()
        content = block[len(first_line):].strip() if len(block) > len(first_line) else ""
        chapters.append({"title": first_line, "content": content})

    # 如果第一个匹配之前有内容（序章等），作为第 0 章
    if matches and matches[0].start() > 200:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            chapters.insert(0, {"title": "序", "content": preamble})

    return chapters


def split_epub(file_path: str) -> list[dict]:
    """切分 epub 文件为章节列表"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        print("[错误] 需要安装 ebooklib 和 beautifulsoup4: pip install ebooklib beautifulsoup4")
        sys.exit(1)

    book = epub.read_epub(file_path)
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html = item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        # 移除 script/style
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if not text or len(text) < 50:
            continue

        # 尝试在单个 spine item 中二次切分
        sub_matches = list(CHAPTER_RE.finditer(text))
        if len(sub_matches) >= 2:
            for i, m in enumerate(sub_matches):
                start = m.start()
                end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(text)
                block = text[start:end].strip()
                if block:
                    first_line = block.split("\n", 1)[0].strip()
                    content = block[len(first_line):].strip()
                    chapters.append({"title": first_line, "content": content})
        else:
            # 整个 item 作为一章
            first_line = text.split("\n", 1)[0].strip()[:80]
            content = text
            chapters.append({"title": first_line, "content": content})

    return chapters


def to_slug(title: str) -> str:
    """中文标题转 slug（取拼音首字母或简化）"""
    # 简单处理：去除特殊字符，保留字母数字和中文
    slug = re.sub(r"[^\w\u4e00-\u9fff]", "", title)
    return slug[:20] if slug else "novel"


# ============================================================
# import 命令
# ============================================================
def cmd_import(args):
    """导入小说文件，切分章节，创建目录结构"""
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        print(f"正在解析 txt 文件...")
        chapters = split_txt(file_path)
    elif ext == ".epub":
        print(f"正在解析 epub 文件...")
        chapters = split_epub(file_path)
    else:
        print(f"[错误] 不支持的格式: {ext}（支持 .txt / .epub）")
        return

    if not chapters:
        print("[错误] 未能切分出任何章节")
        return

    print(f"切分完成，共 {len(chapters)} 章")

    # 创建索引条目
    data = load_index()
    novel_id = next_id(data)
    slug = to_slug(args.title)
    novel_dir = get_novel_dir(novel_id, slug)

    # 创建目录结构
    for sub in ["chapters", "summaries/chapter", "summaries/chapter_detail",
                "summaries/volume", "indexes"]:
        (novel_dir / sub).mkdir(parents=True, exist_ok=True)

    # 写入章节文件
    for i, ch in enumerate(chapters, 1):
        ch_file = novel_dir / "chapters" / f"{i:04d}.txt"
        with open(ch_file, "w", encoding="utf-8") as f:
            f.write(f"# {ch['title']}\n\n{ch['content']}")

    # 创建 meta.json
    total = len(chapters)
    meta = {
        "title": args.title,
        "author": args.author or "未知",
        "source_format": ext.lstrip("."),
        "total_chapters": total,
        "volume_size": VOLUME_SIZE,
        "progress": {
            "split": "done",
            "chapter_summaries": {"done": [], "total": total},
            "chapter_details": {"done": [], "ranges_requested": []},
            "volume_summaries": {"done": [], "total": (total + VOLUME_SIZE - 1) // VOLUME_SIZE},
            "book_summary": "pending",
            "indexes": {
                "worldbuilding": "pending",
                "characters": "pending",
                "power_system": "pending",
                "plot_main": "pending",
                "plot_sub": "pending",
                "side_characters": "pending",
                "pacing": "pending",
            },
        },
    }
    save_meta(novel_dir, meta)

    # 更新索引
    data["novels"].append({
        "id": novel_id,
        "title": args.title,
        "author": args.author or "未知",
        "slug": slug,
        "total_chapters": total,
    })
    save_index(data)

    print(f"导入成功！ID: {novel_id}，目录: {novel_dir}")
    print(f"下一步: python3 {__file__} analyze --id {novel_id}")


# ============================================================
# list / status / remove 命令
# ============================================================
def cmd_list(args):
    """列出所有已导入小说"""
    data = load_index()
    if not data["novels"]:
        print("暂无导入的小说")
        return
    print(f"{'ID':<5} {'标题':<20} {'作者':<10} {'章节数':<8}")
    print("-" * 50)
    for n in data["novels"]:
        print(f"{n['id']:<5} {n['title']:<20} {n.get('author',''):<10} {n.get('total_chapters',''):<8}")


def cmd_status(args):
    """查看分析进度"""
    data = load_index()
    novel = find_novel(data, args.id)
    if not novel:
        print(f"[错误] 找不到 ID={args.id} 的小说")
        return

    novel_dir = get_novel_dir(novel["id"], novel["slug"])
    meta = load_meta(novel_dir)
    p = meta["progress"]
    total = meta["total_chapters"]

    print(f"《{meta['title']}》 作者: {meta['author']}")
    print(f"格式: {meta['source_format']}  总章数: {total}")
    print()

    # 章节摘要
    ch_done = len(p["chapter_summaries"]["done"])
    print(f"章节摘要:   {ch_done}/{total} ({ch_done*100//total if total else 0}%)")

    # 详细拆解
    det_done = len(p["chapter_details"]["done"])
    det_ranges = p["chapter_details"]["ranges_requested"]
    print(f"详细拆解:   {det_done} 章已完成  请求范围: {det_ranges or '无'}")

    # 卷摘要
    vol_done = len(p["volume_summaries"]["done"])
    vol_total = p["volume_summaries"]["total"]
    print(f"卷摘要:     {vol_done}/{vol_total}")

    # 全书摘要
    print(f"全书摘要:   {p['book_summary']}")

    # 索引
    print(f"索引:")
    for k, v in p["indexes"].items():
        label = {"worldbuilding": "世界观", "characters": "人物图谱",
                 "power_system": "金手指", "plot_main": "主线",
                 "plot_sub": "支线", "side_characters": "配角",
                 "pacing": "故事节奏"}.get(k, k)
        print(f"  {label:<10} {v}")


def cmd_remove(args):
    """删除小说分析数据"""
    data = load_index()
    novel = find_novel(data, args.id)
    if not novel:
        print(f"[错误] 找不到 ID={args.id} 的小说")
        return

    novel_dir = get_novel_dir(novel["id"], novel["slug"])
    if novel_dir.exists():
        shutil.rmtree(novel_dir)

    data["novels"] = [n for n in data["novels"] if n["id"] != args.id]
    save_index(data)
    print(f"已删除《{novel['title']}》(ID={args.id})")


# ============================================================
# Prompt 模板
# ============================================================
PROMPT_CHAPTER_SUMMARY = """请为以下章节生成约500字的结构化摘要，格式如下：

## 事件
（本章主要事件与结果）

## 人物
（出场人物及关键行为）

## 场景
（按顺序列出每个场景：地点、参与者、核心冲突）

## 节奏
（标记本章节奏类型：铺垫/推进/高潮/过渡）

## 钩子
（章末悬念或下章引子）

---
{chapter_text}"""

PROMPT_CHAPTER_DETAIL = """请为以下章节生成约1500字的详细场景拆解，格式如下：

## 事件
（本章主要事件与结果）

## 人物
（出场人物及关键行为）

## 场景序列
按顺序拆解每个场景：
### 场景1：[地点]
- 参与者：
- 展开方式：（如何进入这个场景）
- 核心冲突/推进：
- 关键对话目的：
- 情绪走向：从___到___
- 转出方式：（如何过渡到下一场景）

### 场景2：[地点]
...

## 悬念与伏笔
- 本章埋设的伏笔：
- 本章回收的伏笔：

## 情绪曲线
（整章情绪走向描述）

## 章节结构
- 开头钩子手法：
- 中段推进手法：
- 结尾悬念手法：

---
{chapter_text}"""

PROMPT_VOLUME_SUMMARY = """以下是第{volume_num}卷（第{start}-{end}章）的各章摘要。
请合并生成约2000字的卷摘要，包含：本卷核心事件线、人物关系变化、世界观新增信息、与前后卷的衔接。

{chapter_summaries}"""

PROMPT_BOOK_SUMMARY = """以下是全书各卷摘要。请生成约5000字的全书摘要，包含：整体故事脉络、核心冲突与转折、主题演变。

{volume_summaries}"""

PROMPT_WORLDBUILDING = """以下是一部网文的各卷摘要。请提取并整理世界观设定，包含：
- 地理/空间结构
- 势力/阵营划分
- 修炼/魔法/力量体系规则
- 重要物品/资源
- 世界运行规则

用结构化的 Markdown 格式输出。

{volume_summaries}"""

PROMPT_POWER_SYSTEM = """以下是一部网文的各卷摘要。请提取主角的金手指/力量体系，包含：
- 主角初始能力与获得的特殊能力
- 力量等级体系与进阶路线
- 关键道具/功法/技能
- 能力的限制与代价

用结构化的 Markdown 格式输出。

{volume_summaries}"""

PROMPT_CHARACTERS = """以下是一部网文的各卷摘要。请提取并整理人物图谱，包含：
- 主角：性格特征、成长变化、核心动机
- 主要配角：与主角的关系、各自动机、关系变化
- 阵营归属
- 重要的人物关系转折点

用结构化的 Markdown 格式输出。

{volume_summaries}"""

PROMPT_SIDE_CHARACTERS = """以下是一部网文的各卷摘要。请从中剥离出重要配角的独立故事线，包含：
- 每个重要配角的个人经历线
- 与主线的交汇点
- 配角的结局或最后状态

用结构化的 Markdown 格式输出。

{volume_summaries}"""

PROMPT_PLOT_MAIN = """以下是一部网文的各卷摘要。请梳理主线剧情，包含：
- 核心冲突与目标
- 主线推进的关键转折点（按时间顺序）
- 伏笔的埋设与回收
- 因果链条

用结构化的 Markdown 格式输出。

{volume_summaries}"""

PROMPT_PLOT_SUB = """以下是一部网文的各卷摘要。请梳理支线剧情，包含：
- 各条支线的起止范围
- 支线与主线的关系
- 支线的独立价值（世界观扩展/人物塑造/主题深化）

用结构化的 Markdown 格式输出。

{volume_summaries}"""

PROMPT_PACING = """以下是一部2025年最火网文的各卷摘要和章节摘要数据。请从以下五个维度深入分析这部小说是如何抓住读者的，每个维度都要具体到卷和章节举例：

## 一、节奏把控
分析每个卷/故事弧的起承转合结构：
- 每卷的节奏模式（铺垫多少章→推进多少章→高潮多少章→收尾多少章）
- 爽点密度：平均多少章出现一次爽点/高潮
- 慢节奏段落（铺垫/日常）的处理技巧：如何让铺垫不无聊
- 快节奏段落的加速手法
- 总结出这部小说的节奏公式

## 二、钩子设计
分析悬念和钩子的设置技巧：
- 章末钩子的常用手法（列举具体类型和频率）
- 卷末钩子的设计模式
- 长线悬念（跨越多卷的伏笔）如何埋设和回收
- 短线悬念（1-3章内解决）的密度和节奏
- 信息差的运用：读者知道但角色不知道 / 角色知道但读者不知道

## 三、升级节奏
分析主角变强的频率和方式：
- 每卷主角的实力提升幅度
- 升级的触发方式（战斗突破/机缘/顿悟/危机逼迫等）
- 升级后的即时验证（如何让读者感受到变强的爽感）
- 实力压制与反杀的交替节奏
- 金手指/外挂的释放节奏

## 四、冲突递进
分析敌人和困难的升级模式：
- 每卷的核心冲突类型（个人恩怨/势力对抗/生死危机/阴谋揭露等）
- 反派的升级曲线：从小角色到大Boss的递进
- 冲突的嵌套结构：大冲突中套小冲突
- "打脸"节奏：被轻视→展示实力→震惊旁观者的频率和变化
- 危机感的维持：如何让读者始终觉得主角有危险

## 五、情绪曲线
分析读者情绪的调控：
- 紧张→释放→紧张的波动规律
- 低谷期（主角受挫/失败）的处理：持续多久，如何转折
- 高光时刻的铺垫手法：如何让爽点更爽
- 搞笑/日常/温情段落的插入时机和作用
- 整体情绪走向：是持续上扬还是波浪式

## 六、综合结论
- 这部小说最核心的"留住读者"技巧是什么（提炼3-5条）
- 可以直接借鉴的写作公式/模板
- 与其他热门网文相比的独特之处

请结合具体卷和章节举例分析，不要泛泛而谈。

{volume_summaries}"""


# ============================================================
# analyze 命令 — 各步骤实现
# ============================================================
def _resolve_novel(args):
    """从 args.id 解析出 novel_dir 和 meta"""
    data = load_index()
    novel = find_novel(data, args.id)
    if not novel:
        print(f"[错误] 找不到 ID={args.id} 的小说")
        sys.exit(1)
    novel_dir = get_novel_dir(novel["id"], novel["slug"])
    meta = load_meta(novel_dir)
    return novel_dir, meta


def _read_chapter(novel_dir: Path, ch_num: int) -> str:
    """读取章节原文"""
    ch_file = novel_dir / "chapters" / f"{ch_num:04d}.txt"
    if not ch_file.exists():
        return ""
    return read_file(str(ch_file))


def _process_chapter_batch(novel_dir, meta, ch_nums, task_type, prompt_tpl, out_subdir):
    """并发处理一批章节的 LLM 调用"""
    results = {}

    def _do_one(ch_num):
        text = _read_chapter(novel_dir, ch_num)
        if not text:
            return ch_num, None
        prompt = prompt_tpl.format(chapter_text=text)
        result = llm_call(task_type, prompt)
        return ch_num, result

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_do_one, n): n for n in ch_nums}
        for fut in as_completed(futures):
            ch_num, result = fut.result()
            if result:
                out_file = novel_dir / "summaries" / out_subdir / f"{ch_num:04d}.md"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(result)
                results[ch_num] = True
    return results


def step_chapter_summaries(novel_dir: Path, meta: dict):
    """生成标准章节摘要（全部章节）"""
    total = meta["total_chapters"]
    done = set(meta["progress"]["chapter_summaries"]["done"])
    todo = [i for i in range(1, total + 1) if i not in done]

    if not todo:
        print("[章节摘要] 已全部完成")
        return

    print(f"[章节摘要] 待处理: {len(todo)} 章")

    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start : batch_start + BATCH_SIZE]
        results = _process_chapter_batch(
            novel_dir, meta, batch,
            "chapter_summary", PROMPT_CHAPTER_SUMMARY, "chapter",
        )
        # 更新进度
        for ch_num in results:
            if ch_num not in done:
                done.add(ch_num)
                meta["progress"]["chapter_summaries"]["done"].append(ch_num)
        save_meta(novel_dir, meta)

        finished = len(done)
        print(f"  [章节摘要] {finished}/{total} ({finished*100//total}%)")


def step_chapter_details(novel_dir: Path, meta: dict, range_str: str):
    """生成详细拆解（指定范围）"""
    start, end = map(int, range_str.split("-"))
    total = meta["total_chapters"]
    start = max(1, start)
    end = min(total, end)

    done = set(meta["progress"]["chapter_details"]["done"])
    todo = [i for i in range(start, end + 1) if i not in done]

    if not todo:
        print(f"[详细拆解] {range_str} 已全部完成")
        return

    # 记录请求范围
    if range_str not in meta["progress"]["chapter_details"]["ranges_requested"]:
        meta["progress"]["chapter_details"]["ranges_requested"].append(range_str)

    print(f"[详细拆解] 范围 {range_str}，待处理: {len(todo)} 章")

    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start : batch_start + BATCH_SIZE]
        results = _process_chapter_batch(
            novel_dir, meta, batch,
            "chapter_detail", PROMPT_CHAPTER_DETAIL, "chapter_detail",
        )
        for ch_num in results:
            if ch_num not in done:
                done.add(ch_num)
                meta["progress"]["chapter_details"]["done"].append(ch_num)
        save_meta(novel_dir, meta)

        finished = len([x for x in done if start <= x <= end])
        target = end - start + 1
        print(f"  [详细拆解] {finished}/{target}")


def step_volume_summaries(novel_dir: Path, meta: dict):
    """生成卷摘要"""
    total = meta["total_chapters"]
    vol_size = meta["volume_size"]
    vol_total = meta["progress"]["volume_summaries"]["total"]
    done = set(meta["progress"]["volume_summaries"]["done"])

    todo = [v for v in range(1, vol_total + 1) if v not in done]
    if not todo:
        print("[卷摘要] 已全部完成")
        return

    # 检查章节摘要是否完成
    ch_done = set(meta["progress"]["chapter_summaries"]["done"])
    if len(ch_done) < total:
        print(f"[卷摘要] 需要先完成章节摘要（当前 {len(ch_done)}/{total}）")
        return

    print(f"[卷摘要] 待处理: {len(todo)} 卷")

    for vol_num in todo:
        start = (vol_num - 1) * vol_size + 1
        end = min(vol_num * vol_size, total)

        # 读取该卷所有章摘要
        summaries = []
        for ch in range(start, end + 1):
            ch_file = novel_dir / "summaries" / "chapter" / f"{ch:04d}.md"
            if ch_file.exists():
                summaries.append(f"### 第{ch}章\n{read_file(str(ch_file))}")

        prompt = PROMPT_VOLUME_SUMMARY.format(
            volume_num=vol_num, start=start, end=end,
            chapter_summaries="\n\n".join(summaries),
        )
        result = llm_call("volume_summary", prompt)

        out_file = novel_dir / "summaries" / "volume" / f"v{vol_num:02d}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result)

        done.add(vol_num)
        meta["progress"]["volume_summaries"]["done"].append(vol_num)
        save_meta(novel_dir, meta)
        print(f"  [卷摘要] {len(done)}/{vol_total}")


def _read_all_volume_summaries(novel_dir: Path, meta: dict) -> str:
    """读取全部卷摘要，拼接为文本"""
    vol_total = meta["progress"]["volume_summaries"]["total"]
    parts = []
    for v in range(1, vol_total + 1):
        vf = novel_dir / "summaries" / "volume" / f"v{v:02d}.md"
        if vf.exists():
            parts.append(f"## 第{v}卷\n{read_file(str(vf))}")
    return "\n\n".join(parts)


PROMPT_CONDENSE = """以下是一部网文第{start}卷到第{end}卷的卷摘要。请将这些内容压缩为约2000字的综合摘要，保留：核心事件、人物变化、世界观要素、力量体系进展。

{text}"""

# 每组卷数（用于分批压缩）
VOLUME_GROUP_SIZE = 10


def _get_condensed_volume_text(novel_dir: Path, meta: dict) -> str:
    """分批压缩卷摘要，返回适合 GPT-4o 处理的精简文本"""
    vol_total = meta["progress"]["volume_summaries"]["total"]

    # 如果卷数不多（<=10），直接返回原文
    if vol_total <= VOLUME_GROUP_SIZE:
        return _read_all_volume_summaries(novel_dir, meta)

    # 分组压缩
    condensed_parts = []
    for group_start in range(1, vol_total + 1, VOLUME_GROUP_SIZE):
        group_end = min(group_start + VOLUME_GROUP_SIZE - 1, vol_total)
        parts = []
        for v in range(group_start, group_end + 1):
            vf = novel_dir / "summaries" / "volume" / f"v{v:02d}.md"
            if vf.exists():
                parts.append(f"## 第{v}卷\n{read_file(str(vf))}")
        group_text = "\n\n".join(parts)

        print(f"  [压缩] 第{group_start}-{group_end}卷...")
        prompt = PROMPT_CONDENSE.format(start=group_start, end=group_end, text=group_text)
        condensed = llm_call("volume_summary", prompt)  # 用 DeepSeek
        condensed_parts.append(f"## 第{group_start}-{group_end}卷综合\n{condensed}")

    return "\n\n".join(condensed_parts)


def step_book_summary(novel_dir: Path, meta: dict):
    """生成全书摘要"""
    if meta["progress"]["book_summary"] == "done":
        print("[全书摘要] 已完成")
        return

    vol_done = len(meta["progress"]["volume_summaries"]["done"])
    vol_total = meta["progress"]["volume_summaries"]["total"]
    if vol_done < vol_total:
        print(f"[全书摘要] 需要先完成卷摘要（当前 {vol_done}/{vol_total}）")
        return

    print("[全书摘要] 生成中...")
    vol_text = _get_condensed_volume_text(novel_dir, meta)
    prompt = PROMPT_BOOK_SUMMARY.format(volume_summaries=vol_text)
    result = llm_call("book_summary", prompt)

    out_file = novel_dir / "summaries" / "book.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result)

    meta["progress"]["book_summary"] = "done"
    save_meta(novel_dir, meta)
    print("  [全书摘要] 完成")


def step_indexes(novel_dir: Path, meta: dict):
    """生成结构化索引"""
    vol_done = len(meta["progress"]["volume_summaries"]["done"])
    vol_total = meta["progress"]["volume_summaries"]["total"]
    if vol_done < vol_total:
        print(f"[索引] 需要先完成卷摘要（当前 {vol_done}/{vol_total}）")
        return

    vol_text = _get_condensed_volume_text(novel_dir, meta)

    index_tasks = {
        "worldbuilding":    (PROMPT_WORLDBUILDING, "worldbuilding.md"),
        "power_system":     (PROMPT_POWER_SYSTEM, "power_system.md"),
        "characters":       (PROMPT_CHARACTERS, "characters.md"),
        "side_characters":  (PROMPT_SIDE_CHARACTERS, "side_characters.md"),
        "plot_main":        (PROMPT_PLOT_MAIN, "plot_main.md"),
        "plot_sub":         (PROMPT_PLOT_SUB, "plot_sub.md"),
        "pacing":           (PROMPT_PACING, "pacing.md"),
    }

    for task_key, (prompt_tpl, filename) in index_tasks.items():
        if meta["progress"]["indexes"][task_key] == "done":
            continue

        print(f"  [索引] 生成 {filename}...")
        prompt = prompt_tpl.format(volume_summaries=vol_text)
        result = llm_call(task_key, prompt)

        out_file = novel_dir / "indexes" / filename
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result)

        meta["progress"]["indexes"][task_key] = "done"
        save_meta(novel_dir, meta)

    print("  [索引] 全部完成")


# ============================================================
# analyze 命令入口
# ============================================================
def cmd_analyze(args):
    """运行分析流程"""
    load_env()
    novel_dir, meta = _resolve_novel(args)

    step = args.step
    detail = args.detail
    range_str = args.range

    if step:
        if step == "chapter":
            if detail:
                if not range_str:
                    print("[错误] --detail 需要配合 --range 使用，如 --range 51-100")
                    return
                step_chapter_details(novel_dir, meta, range_str)
            else:
                step_chapter_summaries(novel_dir, meta)
        elif step == "volume":
            step_volume_summaries(novel_dir, meta)
        elif step == "book":
            step_book_summary(novel_dir, meta)
        elif step == "indexes":
            step_indexes(novel_dir, meta)
        else:
            print(f"[错误] 未知步骤: {step}（可选: chapter, volume, book, indexes）")
    else:
        # 全自动流程（不含详细拆解）
        print(f"开始全自动分析《{meta['title']}》...")
        step_chapter_summaries(novel_dir, meta)
        step_volume_summaries(novel_dir, meta)
        step_book_summary(novel_dir, meta)
        step_indexes(novel_dir, meta)
        print(f"\n分析完成！使用以下命令查看进度:")
        print(f"  python3 {__file__} status --id {args.id}")


# ============================================================
# argparse 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="网文分析器 - 多模型分层压缩")
    sub = parser.add_subparsers(dest="command")

    # import
    p_import = sub.add_parser("import", help="导入小说文件")
    p_import.add_argument("--file", required=True, help="小说文件路径（.txt / .epub）")
    p_import.add_argument("--title", required=True, help="书名")
    p_import.add_argument("--author", default="", help="作者")
    p_import.set_defaults(func=cmd_import)

    # list
    p_list = sub.add_parser("list", help="列出所有已导入小说")
    p_list.set_defaults(func=cmd_list)

    # status
    p_status = sub.add_parser("status", help="查看分析进度")
    p_status.add_argument("--id", type=int, required=True, help="小说 ID")
    p_status.set_defaults(func=cmd_status)

    # analyze
    p_analyze = sub.add_parser("analyze", help="运行分析流程")
    p_analyze.add_argument("--id", type=int, required=True, help="小说 ID")
    p_analyze.add_argument("--step", choices=["chapter", "volume", "book", "indexes"],
                           help="只运行指定步骤")
    p_analyze.add_argument("--detail", action="store_true",
                           help="生成详细拆解（需配合 --range）")
    p_analyze.add_argument("--range", help="章节范围，如 51-100")
    p_analyze.set_defaults(func=cmd_analyze)

    # remove
    p_remove = sub.add_parser("remove", help="删除小说分析数据")
    p_remove.add_argument("--id", type=int, required=True, help="小说 ID")
    p_remove.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
