# 🐍 Python Environment Setup Guide | Python 环境配置指南

**Language / 语言**: [English](#english) | [中文](#中文)

---

## English

> **Important Notice**: RAG Chat App's AI dependencies (LangChain, Transformers, etc.) are sensitive to Python versions. Please use compatible versions.

---

## 中文

> **重要提醒**: RAG Chat App 的 AI 依赖（LangChain、Transformers等）对 Python 版本敏感，请务必使用兼容版本。

## ✅ Supported Python Versions | 支持的 Python 版本

**English:**

| Version Range | Status | Description |
|---------------|---------|-------------|
| Python 3.8-3.11 | ✅ Fully Supported | Recommended |
| Python 3.10-3.11 | 🌟 Best Choice | Optimal compatibility |
| Python 3.12+ | ❌ Not Supported | LangChain compatibility issues |
| Python < 3.8 | ❌ Not Supported | Missing features |

**中文:**

| 版本范围 | 状态 | 说明 |
|---------|------|------|
| Python 3.8-3.11 | ✅ 完全支持 | 推荐使用 |
| Python 3.10-3.11 | 🌟 最佳选择 | 最佳兼容性 |
| Python 3.12+ | ❌ 不支持 | LangChain 兼容性问题 |
| Python < 3.8 | ❌ 不支持 | 功能不完整 |

## 🔍 Check Current Version | 检查当前版本

**English:**
```bash
python3 --version
# or
python --version
```

**中文:**
```bash
python3 --version
# 或
python --version
```

## 🛠️ Version Solutions | 解决版本问题

### Method 1: Using pyenv (Recommended) | 方法一：使用 pyenv (推荐)

**English:**

**1. Install pyenv**
```bash
# macOS
brew install pyenv

# Ubuntu/Debian
curl https://pyenv.run | bash

# Add to shell configuration
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

**2. Install and Use Python 3.11**
```bash
# View available versions
pyenv install --list | grep 3.11

# Install Python 3.11
pyenv install 3.11.0

# Set project to use specific version
cd RAG-CHAT
pyenv local 3.11.0

# Verify version
python --version
```

**中文:**

**1. 安装 pyenv**
```bash
# macOS
brew install pyenv

# Ubuntu/Debian
curl https://pyenv.run | bash

# 添加到 shell 配置
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

**2. 安装并使用 Python 3.11**
```bash
# 查看可用版本
pyenv install --list | grep 3.11

# 安装 Python 3.11
pyenv install 3.11.0

# 设置项目使用指定版本
cd RAG-CHAT
pyenv local 3.11.0

# 验证版本
python --version
```

### Method 2: Virtual Environment | 方法二：虚拟环境

**English:**

**Create Compatible Virtual Environment**
```bash
# If system has multiple Python versions
python3.11 -m venv rag_chat_env

# Activate virtual environment
source rag_chat_env/bin/activate  # Linux/macOS
# or
rag_chat_env\Scripts\activate     # Windows

# Verify version
python --version

# Install dependencies
pip install -r backend/requirements.txt
```

**中文:**

**创建兼容的虚拟环境**
```bash
# 如果系统有多个 Python 版本
python3.11 -m venv rag_chat_env

# 激活虚拟环境
source rag_chat_env/bin/activate  # Linux/macOS
# 或
rag_chat_env\Scripts\activate     # Windows

# 验证版本
python --version

# 安装依赖
pip install -r backend/requirements.txt
```

### Method 3: Conda Environment | 方法三：Conda 环境

**English:**

**Using Anaconda/Miniconda**
```bash
# Create new environment
conda create -n rag_chat python=3.11

# Activate environment
conda activate rag_chat

# Install dependencies
pip install -r backend/requirements.txt
```

**中文:**

**使用 Anaconda/Miniconda**
```bash
# 创建新环境
conda create -n rag_chat python=3.11

# 激活环境
conda activate rag_chat

# 安装依赖
pip install -r backend/requirements.txt
```

## 🚀 Quick Fix Script | 快速修复脚本

**English:**

If you encounter version issues, use the following script for quick fixes:

**中文:**

如果遇到版本问题，可以使用以下脚本快速修复：

**English:**
```bash
#!/bin/bash
# fix_python.sh

echo "🔧 Python Environment Fix Tool"
echo "=============================="

# Check current version
CURRENT_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Current Python Version: $CURRENT_VERSION"

# Check pyenv
if command -v pyenv &> /dev/null; then
    echo "✅ Found pyenv, installing Python 3.11..."
    pyenv install 3.11.0 -s
    pyenv local 3.11.0
    echo "✅ Switched to Python 3.11"
else
    echo "⚠️  pyenv not found, creating virtual environment..."
    python3 -m venv venv --python=python3.11 2>/dev/null || {
        echo "❌ Python 3.11 not installed on system"
        echo "Please install Python 3.11 manually: https://python.org/"
        exit 1
    }
    source venv/bin/activate
    echo "✅ Created and activated virtual environment"
fi

# Verify and install dependencies
python --version
pip install -r backend/requirements.txt

echo "🎉 Python environment setup complete!"
```

**中文:**
```bash
#!/bin/bash
# fix_python.sh

echo "🔧 Python 环境修复工具"
echo "===================="

# 检查当前版本
CURRENT_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "当前 Python 版本: $CURRENT_VERSION"

# 检查 pyenv
if command -v pyenv &> /dev/null; then
    echo "✅ 发现 pyenv，安装 Python 3.11..."
    pyenv install 3.11.0 -s
    pyenv local 3.11.0
    echo "✅ 已切换到 Python 3.11"
else
    echo "⚠️  未发现 pyenv，创建虚拟环境..."
    python3 -m venv venv --python=python3.11 2>/dev/null || {
        echo "❌ 系统未安装 Python 3.11"
        echo "请手动安装 Python 3.11: https://python.org/"
        exit 1
    }
    source venv/bin/activate
    echo "✅ 已创建并激活虚拟环境"
fi

# 验证并安装依赖
python --version
pip install -r backend/requirements.txt

echo "🎉 Python 环境配置完成！"
```

## ❗ Common Errors and Solutions | 常见错误及解决

**English:**

### Error 1: `No module named 'langchain'`
```bash
# Cause: Python version incompatibility
# Solution: Switch to supported version
pyenv local 3.11.0
pip install langchain
```

### Error 2: `ERROR: Failed building wheel for xxx`
```bash
# Cause: Compilation dependency issues, usually occurs in Python 3.12+
# Solution: Downgrade Python version
pyenv install 3.11.0
pyenv local 3.11.0
pip cache purge
pip install -r backend/requirements.txt
```

### Error 3: `ImportError: cannot import name 'xxx' from 'langchain'`
```bash
# Cause: LangChain version doesn't match Python version
# Solution: Use compatible version
pip uninstall langchain -y
pip install "langchain>=0.2,<0.3"
```

**中文:**

### 错误1：`No module named 'langchain'`
```bash
# 原因：Python 版本不兼容
# 解决：切换到支持的版本
pyenv local 3.11.0
pip install langchain
```

### 错误2：`ERROR: Failed building wheel for xxx`
```bash
# 原因：编译依赖问题，通常在 Python 3.12+ 出现
# 解决：降级 Python 版本
pyenv install 3.11.0
pyenv local 3.11.0
pip cache purge
pip install -r backend/requirements.txt
```

### 错误3：`ImportError: cannot import name 'xxx' from 'langchain'`
```bash
# 原因：LangChain 版本与 Python 版本不匹配
# 解决：使用兼容版本
pip uninstall langchain -y
pip install "langchain>=0.2,<0.3"
```

## 💡 Best Practices | 最佳实践

**English:**
1. **Use project-specific environments**: Avoid global Python environment pollution
2. **Pin versions**: Use `requirements.txt` to lock versions in production environments
3. **Regular updates**: Monitor compatibility updates of dependency libraries
4. **Test installations**: Run `./setup.sh` for verification after each environment switch

**中文:**
1. **使用项目专用环境**：避免全局 Python 环境污染
2. **固定版本**：在生产环境使用 `requirements.txt` 锁定版本
3. **定期更新**：关注依赖库的兼容性更新
4. **测试安装**：每次切换环境后运行 `./setup.sh` 验证

## 🆘 Still Having Issues? | 仍然有问题？

**English:**

If the above methods still don't solve the problem, please:

1. **View detailed error information**: `pip install -v`
2. **Submit an Issue**: Include complete error logs
3. **Community help**: Seek help in GitHub Discussions

**中文:**

如果按照上述方法仍无法解决，请：

1. **查看详细错误信息**：`pip install -v`
2. **提交 Issue**：附带完整的错误日志
3. **社区求助**：在 GitHub Discussions 寻求帮助

---

**English:** Remember: A correct Python environment is the foundation for successfully running RAG Chat App! 🚀

**中文:** 记住: 正确的 Python 环境是成功运行 RAG Chat App 的基础！ 🚀