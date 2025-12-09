FROM nginx:alpine

# 复制配置文件
COPY nginx.conf /etc/nginx/nginx.conf

# 暴露端口
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

