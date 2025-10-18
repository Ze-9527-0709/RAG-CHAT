# 🐍 Python环境配置指南

**语言**: [English](../PYTHON_SETUP.md) | [中文](PYTHON_SETUP.md)

---

> **RAG聊天应用的完整Python环境配置**

本指南提供了设置与RAG聊天应用兼容的Python环境的详细说明，特别关注LangChain依赖要求。

## 🚨 关键版本要求

### 支持的Python版本
- ✅ **Python 3.8** - 完全支持
- ✅ **Python 3.9** - 完全支持  
- ✅ **Python 3.10** - 推荐使用
- ✅ **Python 3.11** - 推荐使用
- ❌ **Python 3.12** - 不支持 (LangChain兼容性问题)
- ❌ **Python 3.13** - 不支持
- ❌ **Python 3.14** - 不支持

> **⚠️ 重要提醒**: LangChain和相关AI库对版本要求严格。使用不支持的版本会导致安装失败。

## 🔍 检查当前Python版本

```bash
# 检查Python版本
python --version
python3 --version

# 检查pip是否可用
pip --version
pip3 --version
```

## 🛠️ 安装方法

### 方法1: 官方Python安装程序 (推荐新手使用)

1. **访问官方网站**
   - 前往: https://www.python.org/downloads/
   - 下载Python 3.10或3.11 (最新稳定版)

2. **安装步骤**
   ```bash
   # macOS: 下载.pkg安装包
   # Windows: 下载.exe安装包  
   # Linux: 使用包管理器或从源码编译
   ```

3. **验证安装**
   ```bash
   python3 --version
   # 应显示: Python 3.10.x 或 Python 3.11.x
   ```

### 方法2: 包管理器

#### macOS (Homebrew)
```bash
# 如果未安装Homebrew，先安装
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装Python 3.11
brew install python@3.11

# 设为默认 (可选)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Ubuntu/Debian
```bash
# 更新包列表
sudo apt update

# 安装Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-pip

# 安装其他工具
sudo apt install python3.11-dev python3.11-distutils
```

#### CentOS/RHEL/Fedora
```bash
# 安装Python 3.11 (Fedora)
sudo dnf install python3.11 python3.11-pip python3.11-venv

# 对于CentOS/RHEL，首先启用EPEL仓库
sudo yum install epel-release
sudo yum install python311 python311-pip
```

### 方法3: Python版本管理器 (pyenv) - 高级用户

pyenv允许您安装和管理多个Python版本。

#### 安装pyenv

**macOS:**
```bash
# 使用Homebrew安装
brew install pyenv

# 添加到shell配置
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

**Linux:**
```bash
# 使用curl安装
curl https://pyenv.run | bash

# 添加到shell配置
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

#### 使用pyenv安装Python

```bash
# 列出可用的Python版本
pyenv install --list | grep 3.11

# 安装Python 3.11.x (最新版)
pyenv install 3.11.5

# 设为全局默认
pyenv global 3.11.5

# 验证
python --version
```

## 🏠 虚拟环境设置

虚拟环境隔离项目依赖并防止冲突。

### 方法1: venv (内置，推荐)

```bash
# 导航到项目目录
cd RAG-Chat-App

# 创建虚拟环境
python3 -m venv venv

# 激活环境
# macOS/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

# 验证激活 (应显示venv路径)
which python
python --version

# 安装项目依赖
pip install -r backend/requirements.txt
```

### 方法2: conda (如果使用Anaconda/Miniconda)

```bash
# 使用Python 3.11创建conda环境
conda create -n rag-chat python=3.11

# 激活环境
conda activate rag-chat

# 安装pip包
pip install -r backend/requirements.txt

# 或通过conda安装可用包
conda install numpy pandas
pip install -r backend/requirements.txt
```

### 方法3: virtualenv (第三方)

```bash
# 安装virtualenv
pip install virtualenv

# 创建环境
virtualenv -p python3.11 venv

# 激活
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r backend/requirements.txt
```

## 🐍 Conda环境设置 (高级)

### 安装Miniconda

1. **下载Miniconda**
   - 访问: https://docs.conda.io/en/latest/miniconda.html
   - 选择适合您操作系统的安装程序

2. **安装和设置**
   ```bash
   # macOS/Linux
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # 按照提示操作，重启终端
   conda --version
   ```

### 创建项目环境

```bash
# 创建指定Python版本的环境
conda create -n rag-chat python=3.11 pip

# 激活环境
conda activate rag-chat

# 通过conda安装核心包 (更快)
conda install numpy pandas scipy

# 通过pip安装AI包
pip install -r backend/requirements.txt

# 列出已安装包
conda list
pip list
```

### 环境管理

```bash
# 列出所有环境
conda env list

# 删除环境
conda env remove -n rag-chat

# 导出环境
conda env export > environment.yml

# 从导出文件创建
conda env create -f environment.yml
```

## 🔧 依赖安装与故障排除

### 核心依赖

RAG聊天应用需要这些关键包：

```bash
# AI和ML包
pip install openai langchain langchain-huggingface
pip install pinecone-client sentence-transformers

# Web框架
pip install fastapi uvicorn

# 数据处理
pip install pandas numpy

# 文件处理
pip install python-multipart
```

### 常见安装问题

#### 问题1: LangChain兼容性

**问题**: `ERROR: Cannot install langchain with Python 3.12+`

**解决方案**:
```bash
# 检查Python版本
python --version

# 如果是3.12+，安装兼容的Python版本
pyenv install 3.11.5
pyenv local 3.11.5

# 重新创建虚拟环境
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

#### 问题2: 编译错误

**问题**: `error: Microsoft Visual C++ 14.0 is required` (Windows)

**解决方案**:
```bash
# 选项1: 安装Visual Studio构建工具
# 从此处下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 选项2: 使用预编译wheels
pip install --only-binary=all -r backend/requirements.txt

# 选项3: 对有问题的包使用conda
conda install numpy scipy pandas
pip install -r backend/requirements.txt
```

#### 问题3: 权限错误

**问题**: 安装时出现 `Permission denied`

**解决方案**:
```bash
# 使用用户安装 (在venv中不推荐)
pip install --user package_name

# 修复权限 (macOS/Linux)
sudo chown -R $(whoami) ~/.local

# 使用虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

#### 问题4: 网络/SSL错误

**问题**: SSL证书验证错误

**解决方案**:
```bash
# 升级pip和证书
pip install --upgrade pip

# 临时解决方案 (生产环境不推荐)
pip install --trusted-host pypi.org --trusted-host pypi.python.org package_name

# 使用公司代理设置
pip install --proxy http://user:pass@proxy.company.com:port package_name
```

### 验证命令

```bash
# 测试Python安装
python -c "import sys; print(sys.version)"

# 测试关键依赖
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import langchain; print('LangChain:', langchain.__version__)"
python -c "import pinecone; print('Pinecone: OK')"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"

# 测试后端启动
cd backend
python app.py
# 应该无错误启动
```

## 🔄 环境切换与管理

### 激活/停用环境

```bash
# 激活venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 停用任何环境
deactivate

# 激活conda环境
conda activate rag-chat

# 停用conda环境
conda deactivate
```

### 多项目管理

```bash
# 项目1: RAG聊天
cd ~/projects/rag-chat
source venv/bin/activate

# 项目2: 其他AI项目
cd ~/projects/other-ai
source other-venv/bin/activate

# 为多个项目使用conda
conda create -n project1 python=3.11
conda create -n project2 python=3.10
```

## 🚀 快速设置脚本

### 自动化环境设置

创建 `setup_python.sh`:

```bash
#!/bin/bash
set -e

echo "🐍 为RAG聊天应用设置Python环境..."

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | grep -o "3\.[0-9][0-9]*" | head -1)
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ] && [ "$MINOR" -le 11 ]; then
    echo "✅ Python $PYTHON_VERSION 兼容"
else
    echo "❌ Python $PYTHON_VERSION 不兼容"
    echo "请安装Python 3.8-3.11"
    exit 1
fi

# 创建虚拟环境
echo "📦 创建虚拟环境..."
python3 -m venv venv

# 激活环境
echo "🔌 激活环境..."
source venv/bin/activate

# 升级pip
echo "⬆️  升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装依赖..."
pip install -r backend/requirements.txt

echo "🎉 设置完成！使用以下命令激活: source venv/bin/activate"
```

使其可执行并运行:
```bash
chmod +x setup_python.sh
./setup_python.sh
```

### 环境健康检查

创建 `check_env.py`:

```python
#!/usr/bin/env python3
import sys
import importlib

def check_python_version():
    version = sys.version_info
    if 3.8 <= version.major == 3 <= version.minor <= 3.11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - 兼容")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - 不兼容")
        return False

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False

if __name__ == "__main__":
    print("🔍 RAG聊天应用环境检查\n")
    
    # 检查Python版本
    python_ok = check_python_version()
    print()
    
    # 检查必需包
    packages = [
        ('OpenAI', 'openai'),
        ('LangChain', 'langchain'),
        ('FastAPI', 'fastapi'),
        ('Uvicorn', 'uvicorn'),
        ('Pinecone', 'pinecone'),
        ('Sentence Transformers', 'sentence_transformers'),
        ('HuggingFace Transformers', 'transformers'),
    ]
    
    all_packages_ok = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_packages_ok = False
    
    print()
    if python_ok and all_packages_ok:
        print("🎉 环境已准备好运行RAG聊天应用！")
    else:
        print("⚠️  环境需要注意。请安装缺失组件。")
        if not python_ok:
            print("   - 安装兼容的Python版本 (3.8-3.11)")
        if not all_packages_ok:
            print("   - 安装缺失包: pip install -r backend/requirements.txt")
```

运行检查:
```bash
python check_env.py
```

## 🆘 紧急修复

### 完全环境重置

如果一切都出现问题:

```bash
# 删除现有环境
rm -rf venv

# 清理pip缓存
pip cache purge

# 从头重新安装
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 替代Python安装 (Linux)

如果系统Python有问题:

```bash
# 从deadsnakes PPA安装 (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip

# 使用特定Python版本
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

## 📋 最佳实践

### 开发工作流程

1. **始终使用虚拟环境**
2. **在requirements.txt中固定依赖版本**
3. **为团队成员记录环境设置**
4. **定期更新依赖并测试**
5. **开发/生产环境保持一致**

### 依赖管理

```bash
# 生成当前依赖
pip freeze > requirements.txt

# 安装精确版本
pip install -r requirements.txt

# 更新特定包
pip install --upgrade package_name
pip freeze > requirements.txt

# 检查过时包
pip list --outdated
```

### 安全考虑

```bash
# 检查安全漏洞
pip install safety
safety check

# 更新有安全修复的包
pip install --upgrade pip setuptools wheel
pip install --upgrade -r requirements.txt
```

---

**🎯 准备开始了吗？** 返回 [快速开始指南](QUICK_START.md) 启动您的RAG聊天应用！