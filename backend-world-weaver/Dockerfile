FROM python:3.11-slim

WORKDIR /app

# Configure pip to use Aliyun mirror for faster downloads in China
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

COPY . .

# Expose port 8001 (distinct from deep-search 8000)
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

