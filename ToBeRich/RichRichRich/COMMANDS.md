# 快捷指令手册

## 数据管理

### 拉取全部历史数据
```bash
python3 src/data/crawler.py --full
```
从 500.com / 78500.cn 拉取大乐透和双色球的全部历史开奖数据，保存到 `data/` 目录。

### 更新最新数据
```bash
python3 src/data/crawler.py --update
```
增量更新，只拉取新开奖的数据并合并到已有数据中。

### 查看数据状态
```bash
python3 src/data/crawler.py --status
```
显示当前已有数据的期数、时间范围、更新时间等信息。

### 从CSV手动导入
```bash
# 导入大乐透（文件名需包含 dlt 或 大乐透）
python3 src/data/crawler.py --import-csv data/dlt_import.csv

# 导入双色球（文件名需包含 ssq 或 双色球）
python3 src/data/crawler.py --import-csv data/ssq_import.csv

# 同时导入两种
python3 src/data/crawler.py --import-csv data/dlt_import.csv data/ssq_import.csv
```

CSV格式：
- 大乐透：`期号,日期,红1,红2,红3,红4,红5,蓝1,蓝2`
- 双色球：`期号,日期,红1,红2,红3,红4,红5,红6,蓝1`

---

## 环境准备

### 安装依赖
```bash
pip3 install requests
```

### 检查 Python 版本
```bash
python3 --version
```

---

## 数据文件说明

| 文件 | 说明 |
|------|------|
| `data/daletou_history.json` | 大乐透历史开奖数据 |
| `data/shuangseqiu_history.json` | 双色球历史开奖数据 |
