# 开发指南

本文档说明如何在开发和生产环境中使用 Docker。

## 📋 目录

- [两种模式对比](#两种模式对比)
- [开发模式（推荐日常开发）](#开发模式推荐日常开发)
- [生产模式](#生产模式)
- [常见问题](#常见问题)

---

## 🔄 两种模式对比

| 特性 | 开发模式 | 生产模式 |
|------|---------|---------|
| **配置文件** | `docker-compose.dev.yml` | `docker-compose.yml` |
| **代码修改** | ✅ 即时生效，无需重新构建 | ❌ 需要重新构建 |
| **热重载** | ✅ 支持（前端/后端） | ❌ 不支持 |
| **端口暴露** | ✅ 直接访问各个服务 | ❌ 仅通过 Nginx 访问 |
| **镜像大小** | 较大（包含开发工具） | 较小（优化过的生产镜像） |
| **适用场景** | 本地开发、调试 | 部署到服务器 |
| **构建时间（首次）** | 5-15 分钟 | 10-30 分钟 |
| **修改代码后启动** | 秒级 | 需要重新构建（分钟级） |

---

## 🛠️ 开发模式（推荐日常开发）

### 特点

- ✅ **代码即时生效**：修改代码后，刷新浏览器即可看到效果
- ✅ **热重载**：前端和后端都支持自动重载
- ✅ **快速迭代**：无需每次都重新构建镜像
- ✅ **便于调试**：直接访问各个服务的端口

### 使用方法

#### 1. 首次启动（构建镜像）

```bash
# 构建并启动所有服务
docker compose -f docker-compose.dev.yml up --build

# 或者后台运行
docker compose -f docker-compose.dev.yml up --build -d
```

**首次构建时间**：约 5-15 分钟（取决于网络速度）

#### 2. 日常开发（代码修改后）

```bash
# 直接启动，无需重新构建！
docker compose -f docker-compose.dev.yml up

# 或者后台运行
docker compose -f docker-compose.dev.yml up -d
```

**启动时间**：约 10-30 秒

#### 3. 修改代码

**前端代码**：
```bash
# 修改 frontend-world-weaver/app/page.tsx
# 保存后，浏览器自动刷新，即可看到效果！✨
```

**后端代码**：
```bash
# 修改 backend-deep-search/main.py
# 保存后，uvicorn 自动重载，几秒后生效！✨
```

#### 4. 访问服务

开发模式下，可以直接访问各个服务：

```
前端服务：
- Hub:              http://localhost:3001
- Deep Search:      http://localhost:3002
- World Weaver:     http://localhost:3003

后端服务：
- Deep Search API:  http://localhost:8000
- World Weaver API: http://localhost:8001

完整服务（通过 Nginx）：
- HTTPS:            https://localhost
- HTTP:             http://localhost
```

#### 5. 查看日志

```bash
# 查看所有服务日志
docker compose -f docker-compose.dev.yml logs -f

# 查看特定服务日志
docker compose -f docker-compose.dev.yml logs -f backend-deep-search
docker compose -f docker-compose.dev.yml logs -f frontend-world-weaver
```

#### 6. 停止服务

```bash
# 停止所有服务
docker compose -f docker-compose.dev.yml down

# 停止并删除数据卷
docker compose -f docker-compose.dev.yml down -v
```

### 工作流示例

```bash
# 周一上午：启动开发环境
docker compose -f docker-compose.dev.yml up -d

# 开始开发...
# 修改 frontend-world-weaver/app/page.tsx
# 保存，刷新浏览器，立即看到效果！✨

# 修改 backend-deep-search/main.py  
# 保存，等待 2 秒，API 自动重载！✨

# 午休：保持运行（不需要停止）

# 下午继续开发...
# 修改代码，保存，即时生效！

# 下班：停止环境
docker compose -f docker-compose.dev.yml down

# 周二上午：快速启动（无需重新构建）
docker compose -f docker-compose.dev.yml up -d
# 10 秒后就可以继续开发了！
```

---

## 🚀 生产模式

### 特点

- ✅ **优化的镜像**：体积更小，启动更快
- ✅ **安全性高**：通过 Nginx 统一入口
- ✅ **适合部署**：配置符合生产环境标准

### 使用方法

#### 1. 构建并启动

```bash
# 构建并启动
docker compose up --build -d

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f
```

#### 2. 更新代码（需要重新构建）

```bash
# 修改代码后，必须重新构建
docker compose up --build -d
```

**构建时间**：约 10-30 分钟（首次或依赖变化时）

#### 3. 访问服务

生产模式下，通过 Nginx 统一访问：

```
HTTPS:  https://your-domain.com
HTTP:   http://your-domain.com (自动重定向到 HTTPS)
```

---

## ❓ 常见问题

### Q1: 什么时候需要重新构建？

**开发模式 (`docker-compose.dev.yml`)**：
- ✅ **不需要重新构建**：修改 Python/JavaScript/TypeScript 代码
- ❌ **需要重新构建**：
  - 修改 `requirements.txt`（Python 依赖）
  - 修改 `package.json`（Node.js 依赖）
  - 修改 `Dockerfile`

**生产模式 (`docker-compose.yml`)**：
- ❌ **需要重新构建**：任何代码或配置修改

### Q2: 如何只重新构建某个服务？

```bash
# 开发模式
docker compose -f docker-compose.dev.yml up --build backend-deep-search -d

# 生产模式
docker compose up --build backend-deep-search -d
```

### Q3: volumes 挂载的原理是什么？

```yaml
volumes:
  - ./backend-deep-search:/app  # 本地目录 -> 容器目录
  - /app/__pycache__             # 排除（使用容器内的）
```

**工作原理**：
1. 容器启动时，将本地的 `./backend-deep-search` 目录挂载到容器的 `/app`
2. 你在本地修改文件，容器内立即可见
3. `/app/__pycache__` 不挂载，使用容器内生成的缓存

### Q4: 依赖变化了怎么办？

```bash
# 如果修改了 requirements.txt 或 package.json
# 需要重新构建镜像

# 开发模式
docker compose -f docker-compose.dev.yml up --build -d

# 生产模式
docker compose up --build -d
```

### Q5: 如何在开发和生产模式之间切换？

```bash
# 停止开发模式
docker compose -f docker-compose.dev.yml down

# 启动生产模式
docker compose up --build -d

# 或反之
docker compose down
docker compose -f docker-compose.dev.yml up --build -d
```

### Q6: volumes 占用了很多空间怎么办？

```bash
# 查看磁盘使用
docker system df

# 清理未使用的数据
docker system prune

# 清理所有未使用的数据（包括未使用的镜像）
docker system prune -a
```

### Q7: 热重载不生效怎么办？

**前端**：
1. 确保使用了 `npm run dev`
2. 检查 Next.js 是否启用了 Fast Refresh
3. 查看容器日志：`docker compose -f docker-compose.dev.yml logs -f frontend-world-weaver`

**后端**：
1. 确保使用了 `uvicorn --reload`
2. 检查 volumes 是否正确挂载
3. 查看容器日志：`docker compose -f docker-compose.dev.yml logs -f backend-deep-search`

---

## 🎯 最佳实践

### 开发时

1. ✅ 使用 `docker-compose.dev.yml`
2. ✅ 保持容器运行，频繁修改代码
3. ✅ 只在依赖变化时重新构建
4. ✅ 定期清理 Docker 缓存（`docker system prune`）

### 部署时

1. ✅ 使用 `docker-compose.yml`
2. ✅ 测试生产构建（确保 Dockerfile 正确）
3. ✅ 使用 Git 管理代码，避免直接在服务器上修改
4. ✅ 配置自动化部署（CI/CD）

---

## 📚 相关资源

- [Docker Compose 文档](https://docs.docker.com/compose/)
- [Docker Volumes 文档](https://docs.docker.com/storage/volumes/)
- [Next.js 开发模式文档](https://nextjs.org/docs/getting-started/installation)
- [FastAPI 热重载文档](https://fastapi.tiangolo.com/tutorial/first-steps/)
