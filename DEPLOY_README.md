# 深度搜索 Agent 部署指南 (Docker Compose)

本指南将帮助您将 Deep Search Agent 部署到任何 Linux 服务器（如 Ubuntu 22.04/24.04）。

## 前置要求

服务器需要安装 Docker 和 Docker Compose。
如果未安装，请运行：
```bash
curl -fsSL https://get.docker.com | sh
```

## 部署步骤

### 1. 上传代码
将整个项目文件夹上传到服务器，例如上传到 `/opt/deep-search`。

### 2. 配置环境变量
确保 `backend/` 目录下有 `.env` 文件，并填入您的 API Keys。
**关键**：如果是生产环境，建议在 `docker-compose.yml` 中把 `frontend` 服务的 `NEXT_PUBLIC_API_URL` 修改为服务器的公网 IP 或域名。

### 3. 启动服务
在项目根目录下运行：
```bash
# 构建并后台启动
docker compose up -d --build
```

### 4. 验证
- **前端**：访问 `http://服务器IP:3000`
- **后端 API**：`http://服务器IP:8000/docs` (Swagger UI)

## 常用维护命令

```bash
# 查看日志
docker compose logs -f

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 更新代码后重新部署
git pull
docker compose up -d --build
```

## 生产环境建议
对于生产环境，建议使用 Nginx作为反向代理，并配置 SSL 证书。

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        # ... proxy headers
    }

    location /api {
        proxy_pass http://localhost:8000;
        # ... proxy headers
    }
}
```

