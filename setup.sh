#!/bin/bash

# RAG Chat App - 快速安装脚本
# Quick Setup Script for RAG Chat App

set -e

echo "🚀 RAG Chat App 快速安装向导"
echo "=================================="

# 检查虚拟环境状态
echo "🔍 检查Python环境..."
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ 检测到激活的虚拟环境: $VIRTUAL_ENV"
    PYTHON_CMD="python"
else
    echo "⚠️  未检测到激活的虚拟环境"
    PYTHON_CMD="python3"
fi

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
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ Python 未安装。"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_PATH=$(which $PYTHON_CMD)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "🐍 当前Python: $PYTHON_VERSION (路径: $PYTHON_PATH)"

# 检查Python版本范围 (3.8 - 3.11)
COMPATIBLE=false
if [ "$PYTHON_MAJOR" -eq 3 ]; then
    if [ "$PYTHON_MINOR" -ge 8 ] && [ "$PYTHON_MINOR" -le 11 ]; then
        COMPATIBLE=true
    fi
fi

if [ "$COMPATIBLE" = true ]; then
    echo "✅ Python $PYTHON_VERSION (兼容LangChain)"
else
    echo "❌ Python版本不兼容！"
    echo "   当前版本: $PYTHON_VERSION"
    echo "   需要版本: 3.8-3.11"
    echo "   LangChain等AI依赖在Python 3.12+上无法正常工作"
    echo ""
    echo "🛠️  解决方案:"
    
    # 检查系统中是否有兼容版本
    echo "   正在检查系统中的其他Python版本..."
    FOUND_COMPATIBLE=false
    
    for py_ver in python3.11 python3.10 python3.9 python3.8; do
        if command -v $py_ver &> /dev/null; then
            VER=$($py_ver -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            echo "   ✅ 发现兼容版本: $py_ver ($VER)"
            PYTHON_CMD=$py_ver
            FOUND_COMPATIBLE=true
            break
        fi
    done
    
    if [ "$FOUND_COMPATIBLE" = false ]; then
        echo "   📦 建议安装方案:"
        echo "   1. 使用 Homebrew: brew install python@3.11"
        echo "   2. 使用 pyenv: 查看 PYTHON_SETUP.md 详细指南"
        echo "   3. 创建虚拟环境: python3.11 -m venv venv"
        echo ""
        read -p "   是否继续安装（可能失败）？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ 已取消安装"
            echo "💡 请安装兼容的Python版本后重新运行 ./setup.sh"
            exit 1
        fi
    else
        echo "   🔄 将使用兼容版本: $PYTHON_CMD"
        PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        echo "   ✅ 切换到 Python $PYTHON_VERSION"
    fi
fi

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

# 根据Python版本选择pip命令
if [ "$PYTHON_CMD" = "python" ]; then
    PIP_CMD="pip"
elif [[ "$PYTHON_CMD" == python3.* ]]; then
    PIP_CMD="${PYTHON_CMD/python/pip}"
else
    PIP_CMD="pip3"
fi

echo "   使用命令: $PIP_CMD install -r requirements.txt"

# 尝试安装，如果失败提供建议
if ! $PIP_CMD install -r requirements.txt; then
    echo ""
    echo "❌ 依赖安装失败！"
    echo "💡 这通常是Python版本兼容性问题"
    echo ""
    echo "🛠️  建议解决方案:"
    echo "1. 创建虚拟环境:"
    echo "   $PYTHON_CMD -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "2. 或安装兼容Python版本:"
    echo "   brew install python@3.11"
    echo ""
    echo "详细指南请查看: PYTHON_SETUP.md"
    exit 1
fi

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
echo "⚠️  重要提醒："
echo "   如遇到 LangChain 安装问题，请查看 PYTHON_SETUP.md"
echo "   确保使用 Python 3.8-3.11 版本"
echo ""
echo "❓ 如遇问题，请访问："
echo "   https://github.com/Ze-9527-0709/RAG-CHAT/issues"