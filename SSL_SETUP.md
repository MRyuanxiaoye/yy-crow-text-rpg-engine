# SSL/HTTPS 配置说明

## ✅ 已完成的配置

### 步骤 2：修改 Nginx Dockerfile ✅
- 已添加证书复制命令
- 已暴露 443 端口

### 步骤 3：修改 nginx.conf ✅
- 已添加 `listen 443 ssl;`
- 已配置 SSL 证书路径
- 已添加 SSL 优化配置（TLS 1.2/1.3）

### 步骤 4：修改 docker-compose.yml ✅
- 已添加 `443:443` 端口映射

---

## 📋 部署前的准备工作

### 1. 放置证书文件

请将从阿里云下载的 SSL 证书文件放入 `nginx/certs/` 目录，并**重命名**为：

- `yy-crow.com.pem` - SSL 证书文件
- `yy-crow.com.key` - SSL 私钥文件

**示例：**
```bash
cd /Users/yuanye/Documents/深度搜索/nginx/certs

# 如果你的文件名是 1234567_yy-crow.com.pem，重命名为：
mv 1234567_yy-crow.com.pem yy-crow.com.pem
mv 1234567_yy-crow.com.key yy-crow.com.key
```

### 2. 验证文件是否存在

```bash
ls -la /Users/yuanye/Documents/深度搜索/nginx/certs/
```

应该看到：
- `yy-crow.com.pem`
- `yy-crow.com.key`
- `README.md`

---

## 🚀 步骤 5：部署到服务器

### 完整部署命令

```bash
# 1. 同步所有文件到服务器（包括证书）
rsync -avz --exclude 'node_modules' --exclude '__pycache__' --exclude '.next' --exclude '.git' --exclude 'venv' \
  /Users/yuanye/Documents/深度搜索/ root@47.84.19.89:/root/yy-crow-tools/

# 2. 在服务器上重新构建并启动 Nginx
ssh root@47.84.19.89 "cd /root/yy-crow-tools && \
  docker compose build nginx && \
  docker compose up -d nginx"

# 3. 验证 HTTPS 是否生效
curl -I https://yy-crow.com
```

### 分步部署（推荐）

```bash
# 步骤 1：同步代码
rsync -avz --exclude 'node_modules' --exclude '__pycache__' --exclude '.next' --exclude '.git' --exclude 'venv' \
  /Users/yuanye/Documents/深度搜索/ root@47.84.19.89:/root/yy-crow-tools/

# 步骤 2：检查证书文件是否同步成功
ssh root@47.84.19.89 "ls -la /root/yy-crow-tools/nginx/certs/"

# 步骤 3：重新构建 Nginx（如果空间不够，先清理）
ssh root@47.84.19.89 "cd /root/yy-crow-tools && \
  docker image prune -f && \
  docker compose build nginx"

# 步骤 4：重启 Nginx
ssh root@47.84.19.89 "cd /root/yy-crow-tools && docker compose up -d nginx"

# 步骤 5：查看 Nginx 日志（确认启动成功）
ssh root@47.84.19.89 "docker logs yy-crow-nginx --tail 30"
```

---

## ✅ 验证 HTTPS

部署完成后，访问以下 URL 验证：

- ✅ HTTP: `http://yy-crow.com`
- ✅ HTTPS: `https://yy-crow.com`
- ✅ 深度搜索: `https://yy-crow.com/tools/deep-search`

---

## ⚠️ 常见问题

### 1. Nginx 启动失败

**错误信息：** `cannot load certificate key`

**解决方法：**
- 检查证书文件是否存在：`ssh root@47.84.19.89 "ls -la /root/yy-crow-tools/nginx/certs/"`
- 检查文件名是否正确：必须是 `yy-crow.com.pem` 和 `yy-crow.com.key`
- 检查文件权限：`ssh root@47.84.19.89 "chmod 600 /root/yy-crow-tools/nginx/certs/*.key"`

### 2. HTTPS 无法访问但 HTTP 正常

**可能原因：**
- 服务器防火墙没有开放 443 端口

**解决方法：**
```bash
# 检查 443 端口是否监听
ssh root@47.84.19.89 "netstat -tulpn | grep :443"

# 如果没有，检查阿里云安全组规则，确保 443 端口已开放
```

### 3. 证书不受信任

**可能原因：**
- 使用的是测试证书
- 证书未正确绑定域名

**解决方法：**
- 测试证书：浏览器会显示警告，点击"继续访问"即可
- 生产环境：使用正式的 SSL 证书

---

## 📝 安全提示

1. ❌ **不要**将证书文件提交到 Git（已添加到 `.gitignore`）
2. ✅ **定期更新**证书（通常证书有效期为 1 年）
3. ✅ **备份**证书文件到安全的地方
4. ✅ 生产环境建议启用 **HTTP 自动跳转 HTTPS**（可选）

---

## 🔄 可选：强制 HTTPS（HTTP 自动跳转）

如果想让所有 HTTP 请求自动跳转到 HTTPS，可以修改 `nginx/nginx.conf`：

```nginx
# 添加一个新的 server 块，专门处理 HTTP 到 HTTPS 的跳转
server {
    listen 80;
    server_name yy-crow.com;
    return 301 https://$server_name$request_uri;
}

# 原来的 server 块改为只监听 HTTPS
server {
    listen 443 ssl;
    server_name yy-crow.com;
    
    # ... 其他配置保持不变 ...
}
```

