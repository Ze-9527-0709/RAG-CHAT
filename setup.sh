#!/bin/bash

# RAG Chat App - 快速安装脚本
# Quick Setup Script for RAG Chat App

set -e

echo "🚀 RAG Chat App 快速安装向导"
echo "=================================="

# 检查系统要求
echo "📋 检查系统要求..."

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装。请访问 https://nodejs.org/ 安装 Node.js 18+"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js 版本过低。当前版本: $(node -v)，需要 18+"
    exit 1
fi
echo "✅ Node.js $(node -v)"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装。请访问 https://python.org/ 安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION"

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未安装。请安装 pip3"
    exit 1
fi
echo "✅ pip3 已安装"

echo ""
echo "🔧 开始安装依赖..."

# 创建并配置环境变量
if [ ! -f ".env" ]; then
    echo "📝 创建环境配置文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件，填入您的API密钥"
    echo "   OpenAI API Key: https://platform.openai.com/api-keys"
    echo "   Pinecone API Key: https://app.pinecone.io/"
fi

# 安装后端依赖
echo "📦 安装后端依赖 (Python)..."
cd backend
pip3 install -r requirements.txt
cd ..

# 安装前端依赖
echo "📦 安装前端依赖 (Node.js)..."
cd frontend
npm install
cd ..

# 创建必要的目录
echo "📁 创建必要目录..."
mkdir -p uploads
mkdir -p docs

# 设置执行权限
echo "🔐 设置脚本执行权限..."
chmod +x start.sh
chmod +x stop.sh

echo ""
echo "🎉 安装完成！"
echo "=================================="
echo ""
echo "📋 下一步操作："
echo "1. 编辑 .env 文件，配置您的API密钥"
echo "   nano .env"
echo ""
echo "2. 启动应用："
echo "   ./start.sh"
echo ""
echo "3. 访问应用："
echo "   http://localhost:5173"
echo ""
echo "📚 详细说明请查看 README.md"
echo ""
echo "❓ 如遇问题，请访问："
echo "   https://github.com/Ze-9527-0709/RAG-CHAT/issues"