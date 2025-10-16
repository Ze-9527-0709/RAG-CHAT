#!/bin/bash

# RAG Chat App 停止脚本
# 使用方法: ./stop.sh

echo "🛑 停止 RAG Chat App..."

# 停止前端
echo "停止前端服务..."
pkill -f "vite"

# 停止后端
echo "停止后端服务..."
pkill -f "uvicorn"

sleep 2

# 检查是否成功停止
lsof -i :5173 -t >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "⚠️  前端进程仍在运行"
else
    echo "✅ 前端已停止"
fi

lsof -i :8000 -t >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "⚠️  后端进程仍在运行"
else
    echo "✅ 后端已停止"
fi

echo ""
echo "🎯 所有服务已停止"
