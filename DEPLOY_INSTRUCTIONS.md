# Deep Search Agent - Quick Deployment Guide

## 1. Update Code & Dependencies (Local -> Server)
We updated `requirements.txt` again to fix the `lxml` error.

Run this on your **local computer**:
```bash
# Replace 47.84.19.89 with your Server IP
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '.git' ./ root@47.84.19.89:/root/deep-search-agent
```

## 2. Rebuild & Restart (On Server)
SSH into your server:
```bash
ssh root@47.84.19.89
```

Then run:
```bash
cd /root/deep-search-agent
# Force rebuild to install the new lxml_html_clean package
docker compose up -d --build
```

## 3. Access
Open browser: `http://47.84.19.89:8501`
