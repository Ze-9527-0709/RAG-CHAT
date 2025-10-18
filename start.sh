#!/bin/bash

# RAG Chat App One-Click Startup Script
# Usage: ./start.sh

echo "🚀 Starting RAG Chat App..."

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查端口是否被占用
check_port() {
    lsof -i :$1 -t >/dev/null 2>&1
}

# 终止旧进程
echo "📋 Cleaning up old processes..."
pkill -f "vite" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
sleep 2

# 启动后端
echo "🔧 Starting backend service (port 8000)..."
cd backend

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment does not exist, please run first: cd backend && python3.13 -m venv venv && venv/bin/pip install -q fastapi 'uvicorn[standard]' python-dotenv openai"
    exit 1
fi

# 选择后端模式
if [ "$1" = "full" ] || [ "$1" = "rag" ]; then
    echo "🧠 Starting full RAG backend (app.py)..."
    venv/bin/python -m uvicorn app:app --port 8000 > ../backend.log 2>&1 &
    BACKEND_MODE="Full RAG Mode"
else
    echo "🚀 Starting lightweight test backend (mock_app.py)..."
    venv/bin/python -m uvicorn mock_app:app --port 8000 > ../backend.log 2>&1 &
    BACKEND_MODE="Test Mode"
fi
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID) - $BACKEND_MODE"

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend service (port 5173)..."
cd ../frontend

# Check node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start frontend (run in background)
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

cd ..

# Wait for services to start
sleep 3

# Check service status
echo ""
echo "🔍 Checking service status..."
if check_port 8000; then
    echo "✅ Backend running: http://localhost:8000"
else
    echo "❌ Backend failed to start, check logs: tail -f backend.log"
fi

if check_port 5173; then
    echo "✅ Frontend running: http://localhost:5173"
else
    echo "❌ Frontend failed to start, check logs: tail -f frontend.log"
fi

echo ""
echo "🎉 Startup complete!"
echo "📱 Access in browser: http://localhost:5173"
echo "🔧 Current backend mode: $BACKEND_MODE"
echo ""
echo "💡 Startup modes:"
echo "   Test mode: ./start.sh (default, quick start)"
echo "   Full RAG: ./start.sh full (requires API keys)"
echo ""
echo "📝 View logs:"
echo "   Backend: tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 Stop services: ./stop.sh"
