#!/bin/bash

# Ollama 本地模型安装脚本
# 用于在 OpenAI token 不足时的备用方案

echo "🦙 Ollama Local Model Installation Wizard"
echo "=================================="

# 检查是否已安装 Ollama
if command -v ollama &> /dev/null; then
    echo "✅ Ollama already installed"
    ollama --version
else
    echo "📦 Installing Ollama..."
    
    # macOS 安装
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS detected, installing with Homebrew..."
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Please install Homebrew first or download manually from https://ollama.ai"
            exit 1
        fi
    # Linux 安装
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Linux detected, downloading installation script..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "❌ Unsupported operating system: $OSTYPE"
        echo "Please visit https://ollama.ai for manual installation"
        exit 1
    fi
fi

echo ""
echo "🚀 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
echo "Ollama service started (PID: $OLLAMA_PID)"

sleep 3

echo ""
echo "📥 Downloading recommended models..."

# 下载轻量级模型 (约 4GB)
echo "Downloading Llama 2 7B Chat (suitable for general conversations)..."
ollama pull llama2:7b-chat

echo ""
echo "Optional models (download as needed):"
echo "  - ollama pull codellama:7b     # Code generation specialized (~4GB)"
echo "  - ollama pull mistral:7b       # Faster alternative (~4GB)" 
echo "  - ollama pull llama2:13b       # 更高质量但更慢 (约 7GB)"

echo ""
echo "🧪 Testing local model..."
echo "Testing llama2:7b-chat..."

TEST_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate -d '{
  "model": "llama2:7b-chat",
  "prompt": "Hello, please respond with: Local model is working!",
  "stream": false
}')

if [[ $? -eq 0 ]] && [[ $TEST_RESPONSE == *"working"* ]]; then
    echo "✅ Local model test successful!"
else
    echo "⚠️  Local model test may have issues, please check:"
    echo "   1. Is Ollama service running properly"
    echo "   2. Is model download complete"
    echo "   3. Is network connection normal"
fi

echo ""
echo "📋 Installation complete!"
echo ""
echo "🔧 Configuration instructions:"
echo "1. Set in .env file:"
echo "   LOCAL_MODEL_ENDPOINT=http://localhost:11434/api/generate"
echo "   LOCAL_MODEL_NAME=llama2:7b-chat"
echo ""
echo "2. RAG Chat App will automatically switch to local model in:"
echo "   - OpenAI API balance insufficient (< $5)"
echo "   - OpenAI API calls fail"
echo "   - Network issues preventing OpenAI access"
echo ""
echo "🎮 Manual test local model:"
echo "   curl -X POST http://localhost:11434/api/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"llama2:7b-chat\", \"prompt\": \"Hello!\"}'"
echo ""
echo "🛑 Stop Ollama service: kill $OLLAMA_PID"