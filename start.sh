#!/bin/bash

# RAG Chat App 一键启动脚本
# 使用方法: ./start.sh

echo "🚀 启动 RAG Chat App..."

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查端口是否被占用
check_port() {
    lsof -i :$1 -t >/dev/null 2>&1
}

# 终止旧进程
echo "📋 清理旧进程..."
pkill -f "vite" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
sleep 2

# 启动后端
echo "🔧 启动后端服务 (端口 8000)..."
cd backend

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行: cd backend && python3.13 -m venv venv && venv/bin/pip install -q fastapi 'uvicorn[standard]' python-dotenv openai"
    exit 1
fi

# 选择后端模式
if [ "$1" = "full" ] || [ "$1" = "rag" ]; then
    echo "🧠 启动完整RAG后端 (app.py)..."
    venv/bin/python -m uvicorn app:app --port 8000 > ../backend.log 2>&1 &
    BACKEND_MODE="完整RAG模式"
else
    echo "🚀 启动轻量测试后端 (mock_app.py)..."
    venv/bin/python -m uvicorn mock_app:app --port 8000 > ../backend.log 2>&1 &
    BACKEND_MODE="测试模式"
fi
BACKEND_PID=$!
echo "✅ 后端已启动 (PID: $BACKEND_PID) - $BACKEND_MODE"

# 等待后端启动
sleep 3

# 启动前端
echo "🎨 启动前端服务 (端口 5173)..."
cd ../frontend

# 检查 node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install
fi

# 启动前端（后台运行）
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✅ 前端已启动 (PID: $FRONTEND_PID)"

cd ..

# 等待服务启动
sleep 3

# 检查服务状态
echo ""
echo "🔍 检查服务状态..."
if check_port 8000; then
    echo "✅ 后端运行中: http://localhost:8000"
else
    echo "❌ 后端启动失败，查看日志: tail -f backend.log"
fi

if check_port 5173; then
    echo "✅ 前端运行中: http://localhost:5173"
else
    echo "❌ 前端启动失败，查看日志: tail -f frontend.log"
fi

echo ""
echo "🎉 启动完成！"
echo "📱 在浏览器访问: http://localhost:5173"
echo "🔧 当前后端模式: $BACKEND_MODE"
echo ""
echo "💡 启动模式:"
echo "   测试模式: ./start.sh (默认，快速启动)"
echo "   完整RAG: ./start.sh full (需要API密钥)"
echo ""
echo "📝 查看日志:"
echo "   后端: tail -f backend.log"
echo "   前端: tail -f frontend.log"
echo ""
echo "🛑 停止服务: ./stop.sh"
