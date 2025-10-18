# 🐍 Python 3.14+ 用户指南

**语言**: [English](../PYTHON_3.14_USERS.md) | [中文](PYTHON_3.14_USERS.md)

---

> **Python 3.14+ 用户的特殊设置指南**

如果您的系统安装了Python 3.14或更新版本，本指南将帮助您为RAG聊天应用设置兼容的环境。

## 🚨 为什么Python 3.14+不能工作

### 兼容性问题

RAG聊天应用依赖几个尚未更新到Python 3.14+的AI/ML库：

- **LangChain**: 核心RAG功能 - 需要Python ≤ 3.11
- **Sentence Transformers**: 文本嵌入 - 与3.12+存在兼容性问题
- **各种ML依赖**: NumPy、SciPy编译的二进制文件可能不可用

### 错误症状

您可能会看到如下错误：
```bash
ERROR: Could not find a version that satisfies the requirement langchain
ERROR: No matching distribution found for sentence-transformers
ImportError: No module named '_ctypes'
```

## 🛠️ 解决方案选项

### 选项1: 使用pyenv (推荐)

**安装pyenv：**

**macOS (Homebrew):**
```bash
# 安装pyenv
brew install pyenv

# 添加到shell配置
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# 重新加载shell
source ~/.zshrc
```

**Linux (curl):**
```bash
# 安装pyenv
curl https://pyenv.run | bash

# 添加到shell配置
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# 重新加载shell
source ~/.bashrc
```

**安装兼容的Python：**
```bash
# 列出可用的Python版本
pyenv install --list | grep "3.11"

# 安装Python 3.11 (最新稳定版)
pyenv install 3.11.9

# 设为项目特定版本
cd RAG-Chat-App
pyenv local 3.11.9

# 验证
python --version  # 应显示Python 3.11.9
```

### 选项2: 使用Conda/Miniconda

**安装Miniconda：**
```bash
# 下载Miniconda安装程序
# 访问: https://docs.conda.io/en/latest/miniconda.html

# 对于Linux/macOS:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 按照安装提示操作
```

**创建兼容环境：**
```bash
# 使用Python 3.11创建环境
conda create -n rag-chat python=3.11

# 激活环境
conda activate rag-chat

# 验证Python版本
python --version  # 应显示Python 3.11.x
```

### 选项3: 使用Docker (隔离)

**创建带兼容Python的Dockerfile：**

创建 `Dockerfile.python311`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 复制依赖文件
COPY backend/requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY backend/ .

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["python", "app.py"]
```

**构建并运行：**
```bash
# 构建Docker镜像
docker build -f Dockerfile.python311 -t rag-chat-backend .

# 运行容器
docker run -p 8000:8000 rag-chat-backend
```

### 选项4: 系统级Python安装

**⚠️ 警告**: 这可能会影响其他应用程序。

**Ubuntu/Debian:**
```bash
# 添加deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 安装Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-pip

# 使用特定版本
python3.11 -m venv venv
source venv/bin/activate
```

**CentOS/RHEL/Fedora:**
```bash
# 从EPEL安装或从源码构建
sudo dnf install python3.11 python3.11-pip python3.11-venv
```

## 🚀 自动化设置脚本

### fix_python_env.sh脚本

项目包含一个自动修复脚本：

```bash
# 使其可执行
chmod +x fix_python_env.sh

# 运行修复程序
./fix_python_env.sh
```

**脚本功能：**

1. **检测您当前的Python版本**
2. **检查pyenv可用性**
3. **如需要则安装pyenv**
4. **下载并安装Python 3.11**
5. **使用兼容Python创建虚拟环境**
6. **安装所有必需依赖**

### 手动脚本内容

如果您想了解脚本的作用：

```bash
#!/bin/bash
set -e

echo "🔧 RAG聊天应用Python 3.14+环境修复器"

# 检查当前Python版本
PYTHON_VERSION=$(python3 --version 2>/dev/null | grep -o "3\.[0-9][0-9]*" || echo "not found")
echo "当前Python版本: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" > "3.11" ]] || [[ "$PYTHON_VERSION" == "not found" ]]; then
    echo "⚠️  检测到Python 3.12+或未找到Python。设置兼容环境..."
    
    # 检查是否安装了pyenv
    if ! command -v pyenv &> /dev/null; then
        echo "📦 安装pyenv..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install pyenv
            else
                echo "请先安装Homebrew: https://brew.sh"
                exit 1
            fi
        else
            # Linux
            curl https://pyenv.run | bash
        fi
        
        # 添加到PATH
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi
    
    # 安装Python 3.11
    echo "🐍 安装Python 3.11..."
    pyenv install 3.11.9 || echo "Python 3.11.9已安装"
    
    # 设置本地版本
    pyenv local 3.11.9
    
    echo "✅ 已为此项目设置Python 3.11.9"
else
    echo "✅ 检测到兼容的Python版本"
fi

# 创建虚拟环境
echo "📦 创建虚拟环境..."
python -m venv venv

# 激活并安装依赖
echo "🔌 安装依赖..."
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt

echo "🎉 设置完成！您的环境已准备就绪。"
echo "激活环境: source venv/bin/activate"
```

## ✅ 验证步骤

### 检查环境设置

设置后，验证一切正常工作：

```bash
# 1. 检查Python版本
python --version
# 应显示: Python 3.11.x

# 2. 激活环境
source venv/bin/activate  # 或 conda activate rag-chat

# 3. 测试关键导入
python -c "import langchain; print('✅ LangChain:', langchain.__version__)"
python -c "import openai; print('✅ OpenAI:', openai.__version__)"
python -c "import sentence_transformers; print('✅ Sentence Transformers: OK')"

# 4. 测试后端启动
cd backend
python app.py
# 应无错误启动
```

### 运行完整应用

```bash
# 启动后端
cd backend
source venv/bin/activate  # 如果使用venv
# 或: conda activate rag-chat  # 如果使用conda
python app.py

# 在另一个终端启动前端
cd frontend
npm install
npm run dev
```

## 🐛 故障排除

### 设置后的常见问题

**问题: 安装后找不到pyenv命令**

```bash
# 添加到您的shell配置文件
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# 重新加载shell
source ~/.bashrc
```

**问题: Python 3.11安装失败**

```bash
# 安装构建依赖 (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
liblzma-dev python3-openssl git

# 然后重试
pyenv install 3.11.9
```

**问题: 虚拟环境激活失败**

```bash
# 重新创建虚拟环境
rm -rf venv
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**问题: 仍然出现导入错误**

```bash
# 清理所有缓存
pip cache purge
rm -rf ~/.cache/pip

# 从头重新安装
pip uninstall -y -r backend/requirements.txt
pip install -r backend/requirements.txt
```

## 📋 替代方法

### 仅使用虚拟环境

如果您无法安装替代Python版本：

```bash
# 尝试使用特定约束安装
pip install --constraint https://raw.githubusercontent.com/langchain-ai/langchain/master/constraints.txt langchain

# 或使用较旧的包版本
pip install langchain==0.0.350  # 示例: 较旧的兼容版本
```

### 使用开发/预览版本

**⚠️ 实验性 - 可能不稳定**

```bash
# 尝试安装可能支持Python 3.14的预览版本
pip install --pre langchain
pip install --pre sentence-transformers
```

### 仅容器开发

如果其他方法都失败，完全在Docker中开发：

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.python311
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    command: python app.py
    
  frontend:
    image: node:18
    volumes:
      - ./frontend:/app
    working_dir: /app
    ports:
      - "5173:5173"
    command: bash -c "npm install && npm run dev"
```

使用以下命令运行：
```bash
docker-compose -f docker-compose.dev.yml up
```

## 🎯 总结

对于Python 3.14+用户，推荐的方法是：

1. **使用pyenv** 在系统Python旁边安装Python 3.11
2. **创建项目特定的虚拟环境** 使用兼容的Python
3. **使用自动化`fix_python_env.sh`脚本** 以便于操作
4. **使用提供的测试命令验证设置**

这种方法在为RAG聊天应用提供兼容性的同时保持系统Python不变。

---

**🚀 准备继续了吗？** 返回 [快速开始指南](QUICK_START.md) 启动您的应用程序！