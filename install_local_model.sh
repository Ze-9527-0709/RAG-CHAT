#!/bin/bash

# Ollama 本地模型安装脚本
# 用于在 OpenAI token 不足时的备用方案

echo "🦙 Ollama 本地模型安装向导"
echo "=================================="

# 检查是否已安装 Ollama
if command -v ollama &> /dev/null; then
    echo "✅ Ollama 已安装"
    ollama --version
else
    echo "📦 安装 Ollama..."
    
    # macOS 安装
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "检测到 macOS，使用 Homebrew 安装..."
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "请先安装 Homebrew 或从 https://ollama.ai 手动下载"
            exit 1
        fi
    # Linux 安装
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "检测到 Linux，下载安装脚本..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "❌ 不支持的操作系统: $OSTYPE"
        echo "请访问 https://ollama.ai 手动安装"
        exit 1
    fi
fi

echo ""
echo "🚀 启动 Ollama 服务..."
ollama serve &
OLLAMA_PID=$!
echo "Ollama 服务已启动 (PID: $OLLAMA_PID)"

sleep 3

echo ""
echo "📥 下载推荐的模型..."

# 下载轻量级模型 (约 4GB)
echo "下载 Llama 2 7B Chat (适合一般对话)..."
ollama pull llama2:7b-chat

echo ""
echo "可选模型 (可根据需要下载):"
echo "  - ollama pull codellama:7b     # 代码生成专用 (约 4GB)"
echo "  - ollama pull mistral:7b       # 更快速的替代方案 (约 4GB)" 
echo "  - ollama pull llama2:13b       # 更高质量但更慢 (约 7GB)"

echo ""
echo "🧪 测试本地模型..."
echo "正在测试 llama2:7b-chat..."

TEST_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate -d '{
  "model": "llama2:7b-chat",
  "prompt": "Hello, please respond with: Local model is working!",
  "stream": false
}')

if [[ $? -eq 0 ]] && [[ $TEST_RESPONSE == *"working"* ]]; then
    echo "✅ 本地模型测试成功！"
else
    echo "⚠️  本地模型测试可能有问题，请检查："
    echo "   1. Ollama 服务是否正常运行"
    echo "   2. 模型是否下载完成"
    echo "   3. 网络连接是否正常"
fi

echo ""
echo "📋 安装完成！"
echo ""
echo "🔧 配置说明："
echo "1. 在 .env 文件中设置:"
echo "   LOCAL_MODEL_ENDPOINT=http://localhost:11434/api/generate"
echo "   LOCAL_MODEL_NAME=llama2:7b-chat"
echo ""
echo "2. RAG Chat App 会在以下情况自动切换到本地模型:"
echo "   - OpenAI API 余额不足 (< $5)"
echo "   - OpenAI API 调用失败"
echo "   - 网络问题无法访问 OpenAI"
echo ""
echo "🎮 手动测试本地模型："
echo "   curl -X POST http://localhost:11434/api/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"llama2:7b-chat\", \"prompt\": \"Hello!\"}'"
echo ""
echo "🛑 停止 Ollama 服务: kill $OLLAMA_PID"