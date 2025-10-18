# 🐍 Python 环境配置指南

> **重要提醒**: RAG Chat App 的 AI 依赖（LangChain、Transformers等）对 Python 版本敏感，请务必使用兼容版本。

## ✅ 支持的 Python 版本

| 版本范围 | 状态 | 说明 |
|---------|------|------|
| Python 3.8-3.11 | ✅ 完全支持 | 推荐使用 |
| Python 3.10-3.11 | 🌟 最佳选择 | 最佳兼容性 |
| Python 3.12+ | ❌ 不支持 | LangChain 兼容性问题 |
| Python < 3.8 | ❌ 不支持 | 功能不完整 |

## 🔍 检查当前版本

```bash
python3 --version
# 或
python --version
```

## 🛠️ 解决版本问题

### 方法一：使用 pyenv (推荐)

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

### 方法二：虚拟环境

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

### 方法三：Conda 环境

**使用 Anaconda/Miniconda**
```bash
# 创建新环境
conda create -n rag_chat python=3.11

# 激活环境
conda activate rag_chat

# 安装依赖
pip install -r backend/requirements.txt
```

## 🚀 快速修复脚本

如果遇到版本问题，可以使用以下脚本快速修复：

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

## ❗ 常见错误及解决

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

## 💡 最佳实践

1. **使用项目专用环境**：避免全局 Python 环境污染
2. **固定版本**：在生产环境使用 `requirements.txt` 锁定版本
3. **定期更新**：关注依赖库的兼容性更新
4. **测试安装**：每次切换环境后运行 `./setup.sh` 验证

## 🆘 仍然有问题？

如果按照上述方法仍无法解决，请：

1. **查看详细错误信息**：`pip install -v`
2. **提交 Issue**：附带完整的错误日志
3. **社区求助**：在 GitHub Discussions 寻求帮助

---

**记住**: 正确的 Python 环境是成功运行 RAG Chat App 的基础！ 🚀