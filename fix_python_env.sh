#!/bin/bash

# 针对当前系统的Python环境快速修复脚本
# 专门处理 Python 3.14/3.13 与 LangChain 不兼容的问题

set -e

echo "🔧 RAG Chat App - Python环境快速修复"
echo "=================================="
echo "检测到系统Python版本过高，正在设置兼容环境..."
echo ""

# 检查当前Python版本
CURRENT_PY=$(python3 --version)
echo "当前系统Python: $CURRENT_PY"

# 方案1: 检查是否已有兼容版本
echo ""
echo "🔍 检查系统中的兼容Python版本..."

COMPATIBLE_PYTHON=""
for py_cmd in python3.11 python3.10 python3.9 python3.8; do
    if command -v $py_cmd &> /dev/null; then
        version=$($py_cmd --version)
        echo "✅ 发现: $py_cmd ($version)"
        COMPATIBLE_PYTHON=$py_cmd
        break
    fi
done

if [ -n "$COMPATIBLE_PYTHON" ]; then
    echo ""
    echo "🎉 找到兼容版本: $COMPATIBLE_PYTHON"
    echo "正在创建项目虚拟环境..."
    
    # 创建虚拟环境
    $COMPATIBLE_PYTHON -m venv venv
    source venv/bin/activate
    
    echo "✅ 虚拟环境已创建并激活"
    echo "Python版本: $(python --version)"
    
    # 安装依赖
    echo "📦 安装项目依赖..."
    cd backend
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
    
    echo ""
    echo "🎉 修复完成！"
    echo ""
    echo "📋 使用说明:"
    echo "1. 每次使用前激活虚拟环境:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. 启动应用:"
    echo "   ./start.sh"
    echo ""
    echo "3. 停用虚拟环境:"
    echo "   deactivate"
    
else
    echo ""
    echo "❌ 未找到兼容的Python版本"
    echo ""
    echo "🛠️  推荐解决方案:"
    echo ""
    echo "方案1: 使用Homebrew安装Python 3.11"
    echo "--------------------------------------"
    
    if command -v brew &> /dev/null; then
        echo "✅ Homebrew已安装"
        echo "运行以下命令："
        echo "   brew install python@3.11"
        echo "   echo 'export PATH=\"/opt/homebrew/opt/python@3.11/bin:\$PATH\"' >> ~/.zshrc"
        echo "   source ~/.zshrc"
        echo ""
        read -p "是否现在安装Python 3.11？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "正在安装Python 3.11..."
            brew install python@3.11
            echo "设置PATH..."
            echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
            
            # 重新检查
            export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
            if command -v python3.11 &> /dev/null; then
                echo "✅ Python 3.11安装成功！"
                echo "重新运行修复脚本..."
                exec "$0"
            fi
        fi
    else
        echo "❌ 未安装Homebrew"
        echo "请先安装Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    fi
    
    echo ""
    echo "方案2: 使用pyenv管理Python版本"
    echo "--------------------------------"
    echo "1. 安装pyenv:"
    echo "   curl https://pyenv.run | bash"
    echo ""
    echo "2. 重启终端并安装Python 3.11:"
    echo "   pyenv install 3.11.0"
    echo "   pyenv local 3.11.0"
    echo ""
    echo "详细指南请查看: PYTHON_SETUP.md"
fi