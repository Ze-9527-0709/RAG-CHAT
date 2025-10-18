#!/bin/bash

# RAG Chat App Stop Script
# Usage: ./stop.sh

echo "🛑 Stopping RAG Chat App..."

# Stop frontend
echo "Stopping frontend service..."
pkill -f "vite"

# Stop backend
echo "Stopping backend service..."
pkill -f "uvicorn"

sleep 2

# Check if successfully stopped
lsof -i :5173 -t >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "⚠️  Frontend process still running"
else
    echo "✅ Frontend stopped"
fi

lsof -i :8000 -t >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "⚠️  Backend process still running"
else
    echo "✅ Backend stopped"
fi

echo ""
echo "🎯 All services stopped"
