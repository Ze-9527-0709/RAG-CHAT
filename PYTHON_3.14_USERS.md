# ⚠️ Special Guide for Python 3.14/3.13 Users | Python 3.14/3.13 用户特别说明

**Language / 语言**: [English](#english) | [中文](#中文)

---

## English

> Your system has a newer Python version installed. Special setup is required to run RAG Chat App.

---

## 中文

> 您的系统安装了较新的 Python 版本，需要特殊设置才能运行 RAG Chat App

## 🔍 Current Detection Status | 当前检测情况

**English:**

- **System Python Version**: 3.14.0 (located at `/opt/homebrew/bin/python3`)
- **Compatibility Status**: ❌ Incompatible with LangChain 
- **Issue Cause**: LangChain and other AI dependencies don't support Python 3.12+ yet

**中文:**

- **系统Python版本**: 3.14.0 (位于 `/opt/homebrew/bin/python3`)
- **兼容性状态**: ❌ 不兼容 LangChain 
- **问题原因**: LangChain 等 AI 依赖库尚未支持 Python 3.12+

## 🚀 Quick Solutions | 快速解决方案

### Option 1: Auto Fix (Recommended) | 选项1: 自动修复 (推荐)

**English:**
```bash
# Run auto-fix script
./fix_python_env.sh
```
This script will:
- Detect available compatible Python versions on system
- Automatically create virtual environment
- Install required dependencies

**中文:**
```bash
# 运行自动修复脚本
./fix_python_env.sh
```
这个脚本会：
- 检测系统中可用的兼容 Python 版本
- 自动创建虚拟环境
- 安装所需依赖

### Option 2: Manual Python 3.11 Installation | 选项2: 手动安装 Python 3.11

**English:**
```bash
# Install Python 3.11 using Homebrew
brew install python@3.11

# Create project virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Verify version
python --version  # Should show 3.11.x

# Install dependencies
cd backend
pip install -r requirements.txt
cd ..
```

**中文:**
```bash
# 使用 Homebrew 安装 Python 3.11
brew install python@3.11

# 创建项目虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 验证版本
python --version  # 应该显示 3.11.x

# 安装依赖
cd backend
pip install -r requirements.txt
cd ..
```

### Option 3: Using pyenv Version Management | 选项3: 使用 pyenv 管理版本

**English:**
```bash
# Install pyenv
curl https://pyenv.run | bash

# After terminal restart, install Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0

# Verify version
python --version  # Should show 3.11.0
```

**中文:**
```bash
# 安装 pyenv
curl https://pyenv.run | bash

# 重启终端后安装 Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0

# 验证版本
python --version  # 应该显示 3.11.0
```

## 💡 Usage Recommendations | 使用建议

**English:**
1. **Use virtual environment**: Avoid affecting system Python
2. **Activate before each use**: `source venv/bin/activate`
3. **Deactivate when done**: `deactivate`

**中文:**
1. **推荐使用虚拟环境**: 避免影响系统 Python
2. **每次启动前激活**: `source venv/bin/activate`
3. **完成后可停用**: `deactivate`

## 🎯 Verify Installation | 验证安装

**English:**

After installation, verify dependencies are correctly installed:

```bash
# Activate virtual environment (if using)
source venv/bin/activate

# Test key dependencies
python -c "import langchain; print('✅ LangChain OK')"
python -c "import openai; print('✅ OpenAI OK')"
python -c "import fastapi; print('✅ FastAPI OK')"

# If all OK, start the application
./start.sh
```

**中文:**

安装完成后，验证依赖是否正确安装：

```bash
# 激活虚拟环境 (如果使用)
source venv/bin/activate

# 测试关键依赖
python -c "import langchain; print('✅ LangChain 正常')"
python -c "import openai; print('✅ OpenAI 正常')"
python -c "import fastapi; print('✅ FastAPI 正常')"

# 如果都正常，可以启动应用
./start.sh
```

## ❓ Having Issues? | 遇到问题？

**English:**
- **Dependency installation failed**: Check if correct Python version is being used
- **Module not found**: Ensure virtual environment is activated
- **Permission issues**: Use `pip install --user` or virtual environment

**中文:**
- **依赖安装失败**: 检查是否使用了正确的 Python 版本
- **模块找不到**: 确认虚拟环境已激活
- **权限问题**: 使用 `pip install --user` 或虚拟环境

---

**English:** Remember: Always activate virtual environment before each use!

**中文:** 记住: 完成设置后，每次使用都要先激活虚拟环境！