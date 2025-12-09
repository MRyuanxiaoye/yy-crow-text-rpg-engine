# YY-Crow Tools - 项目架构说明

## 项目结构

```
yy-crow-tools/
├── frontend-hub/               # 工具集入口前端（端口 3000）
│   ├── app/
│   │   ├── page.tsx           # 首页（工具列表）
│   │   ├── layout.tsx         # 全局布局
│   │   └── globals.css        # 全局样式
│   ├── Dockerfile
│   ├── package.json
│   └── next.config.ts
│
├── frontend-deep-search/       # 深度搜索前端（端口 3001）
│   ├── app/
│   │   ├── page.js            # 深度搜索界面
│   │   └── ...
│   ├── Dockerfile
│   └── package.json
│
├── backend-deep-search/        # 深度搜索后端（端口 8000）
│   ├── app/
│   │   ├── agents/
│   │   ├── tools/
│   │   └── ...
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── nginx/                      # Nginx 反向代理（可选）
│   ├── nginx.conf
│   └── Dockerfile
│
├── docker-compose.yml          # Docker 编排配置
└── PROJECT_STRUCTURE.md        # 本文件
```

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| frontend-hub | 3000 | 工具集入口首页 |
| frontend-deep-search | 3001 | 深度搜索工具前端 |
| backend-deep-search | 8000 | 深度搜索 API |

## 本地开发

### 方式 1：Docker Compose（推荐）

```bash
# 构建并启动所有服务
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

### 方式 2：单独运行

**入口前端**：
```bash
cd frontend-hub
npm install
npm run dev  # http://localhost:3000
```

**深度搜索前端**：
```bash
cd frontend-deep-search
npm install
npm run dev  # http://localhost:3001
```

**深度搜索后端**：
```bash
cd backend-deep-search
pip install -r requirements.txt
python main.py  # http://localhost:8000
```

## 部署到服务器

### 步骤 1：同步代码

```bash
rsync -avz --exclude 'node_modules' --exclude '__pycache__' --exclude '.next' --exclude '.git' \
  /Users/yuanye/Documents/深度搜索/ root@47.84.19.89:/root/yy-crow-tools/
```

### 步骤 2：构建并启动

```bash
ssh root@47.84.19.89 << 'EOF'
cd /root/yy-crow-tools
docker compose down
docker compose up -d --build
EOF
```

### 步骤 3：查看日志

```bash
ssh root@47.84.19.89 "docker compose logs -f"
```

## 添加新工具

### 1. 创建工具页面

在 `frontend-hub/app/page.tsx` 的 `tools` 数组中添加：

```typescript
{
  id: 'new-tool',
  name: '新工具',
  description: '工具描述',
  icon: <YourIcon className="w-8 h-8" />,
  path: 'http://yy-crow.com:3002',  // 或子路径
  tags: ['标签1', '标签2'],
  status: 'active'  // 或 'coming'
}
```

### 2. 创建工具前端（如果需要）

```bash
cd /Users/yuanye/Documents/深度搜索
npx create-next-app@latest frontend-new-tool
```

### 3. 创建工具后端（如果需要）

```bash
mkdir backend-new-tool
# 添加你的后端代码
```

### 4. 更新 docker-compose.yml

```yaml
  frontend-new-tool:
    build: ./frontend-new-tool
    container_name: yy-crow-new-tool-ui
    ports:
      - "3002:3000"
    restart: always

  backend-new-tool:
    build: ./backend-new-tool
    container_name: yy-crow-new-tool-api
    ports:
      - "8001:8000"
    restart: always
```

### 5. 重新部署

```bash
docker compose up -d --build
```

## 访问地址

- **本地**:
  - 工具集首页: http://localhost:3000
  - 深度搜索: http://localhost:3001
  - 深度搜索 API: http://localhost:8000

- **线上**:
  - 工具集首页: http://yy-crow.com
  - 深度搜索: http://yy-crow.com:3001
  - 深度搜索 API: http://yy-crow.com:8000

## 注意事项

1. **环境变量**: 确保每个服务的环境变量配置正确（特别是 API URL）
2. **端口冲突**: 确保服务器上没有其他服务占用相同端口
3. **Nginx 配置**: 如果需要统一入口，可以启用 Nginx 服务（目前未在 docker-compose.yml 中启用）
4. **安全性**: 生产环境建议配置 HTTPS 和防火墙规则

## 故障排查

### 容器无法启动

```bash
# 查看容器状态
docker compose ps

# 查看详细日志
docker logs <container_name>

# 重新构建（不使用缓存）
docker compose build --no-cache
```

### 端口被占用

```bash
# 查看端口占用
netstat -tlnp | grep <port>

# 或
ss -tlnp | grep <port>
```

### 清理磁盘空间

```bash
# 清理未使用的镜像和容器
docker system prune -a -f

# 清理构建缓存
docker builder prune -a -f
```

## 维护建议

- **定期备份**: 定期备份代码和数据库（如果有）
- **日志管理**: 定期清理 Docker 日志，防止占用过多磁盘空间
- **更新依赖**: 定期更新 npm 和 pip 依赖包
- **监控**: 考虑使用 Docker monitoring 工具监控容器状态

## 技术栈

- **前端**: Next.js 14 + React + TypeScript + Tailwind CSS
- **后端**: Python FastAPI + LangChain + LangGraph
- **容器化**: Docker + Docker Compose
- **反向代理**: Nginx (可选)

