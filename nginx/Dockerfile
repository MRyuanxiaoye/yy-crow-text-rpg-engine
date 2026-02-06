FROM nginx:alpine

# 复制配置文件
COPY nginx.conf /etc/nginx/nginx.conf

# 创建证书目录并复制 SSL 证书
RUN mkdir -p /etc/nginx/certs
COPY certs/ /etc/nginx/certs/

# 暴露端口（HTTP 和 HTTPS）
EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]

