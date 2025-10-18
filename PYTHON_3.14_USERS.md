# ⚠️ Python 3.14/3.13 用户特别说明

> 您的系统安装了较新的 Python 版本，需要特殊设置才能运行 RAG Chat App

## 🔍 当前检测情况

- **系统Python版本**: 3.14.0 (位于 `/opt/homebrew/bin/python3`)
- **兼容性状态**: ❌ 不兼容 LangChain 
- **问题原因**: LangChain 等 AI 依赖库尚未支持 Python 3.12+

## 🚀 快速解决方案

### 选项1: 自动修复 (推荐)
```bash
# 运行自动修复脚本
./fix_python_env.sh
```
这个脚本会：
- 检测系统中可用的兼容 Python 版本
- 自动创建虚拟环境
- 安装所需依赖

### 选项2: 手动安装 Python 3.11
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

### 选项3: 使用 pyenv 管理版本
```bash
# 安装 pyenv
curl https://pyenv.run | bash

# 重启终端后安装 Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0

# 验证版本
python --version  # 应该显示 3.11.0
```

## 💡 使用建议

1. **推荐使用虚拟环境**: 避免影响系统 Python
2. **每次启动前激活**: `source venv/bin/activate`
3. **完成后可停用**: `deactivate`

## 🎯 验证安装

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

## ❓ 遇到问题？

- **依赖安装失败**: 检查是否使用了正确的 Python 版本
- **模块找不到**: 确认虚拟环境已激活
- **权限问题**: 使用 `pip install --user` 或虚拟环境

---

**记住**: 完成设置后，每次使用都要先激活虚拟环境！