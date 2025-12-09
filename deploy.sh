#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Deep Search Agent Deployment...${NC}"

# 1. Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}✅ Docker installed successfully.${NC}"
else
    echo -e "${GREEN}✅ Docker is already installed.${NC}"
fi

# 2. Check for .env file
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your API keys before deploying."
    exit 1
fi

# 3. Build and Run
echo -e "${GREEN}📦 Building and starting containers...${NC}"
# Using docker compose plugin (v2) or standalone docker-compose (v1)
if docker compose version &> /dev/null; then
    docker compose up -d --build
else
    docker-compose up -d --build
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 Deployment Successful!${NC}"
    echo -e "Access your agent at: http://<YOUR_SERVER_IP>:8501"
else
    echo "❌ Deployment failed."
    exit 1
fi

