# 🚀 RAG Chat App - 5-Minute Quick Start Guide | 5分钟快速开始指南

**Language / 语言**: [English](#english) | [中文](#中文)

---

## English

> **Beginner Friendly** | **No Complex Setup** | **One-Click Launch**

---

## 中文

> **新手友好** | **无需复杂配置** | **一键启动**

## 🎯 Preparation (5 minutes) | 开始前准备 (5分钟)

### 1️⃣ Download Code | 下载代码

**English:**
```bash
# Method 1: Using Git (Recommended)
git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
cd RAG-CHAT

# Method 2: Direct Download
# Visit https://github.com/Ze-9527-0709/RAG-CHAT
# Click "Code" -> "Download ZIP" -> Extract locally
```

**中文:**
```bash
# 方法一：使用Git (推荐)
git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
cd RAG-CHAT

# 方法二：直接下载
# 访问 https://github.com/Ze-9527-0709/RAG-CHAT
# 点击 "Code" -> "Download ZIP" -> 解压到本地
```

### 2️⃣ Install Required Software | 安装必需软件

**English:**

**📦 Node.js** (JavaScript Runtime)
- Download: https://nodejs.org/
- Choose LTS version (Recommended 18.x+)
- Verify after installation: `node -v`

**🐍 Python** (Backend Language) ⚠️ **Version Critical**
- Download: https://python.org/
- **Recommended: Python 3.10 or 3.11** (Best compatibility)
- **Supported: 3.8-3.11** 
- **Avoid: Python 3.12+** (LangChain incompatible)
- Verify after installation: `python3 --version`

> 💡 **Tip**: If Python 3.12+ already installed, consider using pyenv for version management

**中文:**

**📦 Node.js** (JavaScript运行环境)
- 下载：https://nodejs.org/
- 选择LTS版本 (推荐 18.x+)
- 安装后验证：`node -v`

**🐍 Python** (后端语言) ⚠️ **版本重要**
- 下载：https://python.org/
- **推荐版本: 3.10 或 3.11** (最佳兼容性)
- **支持范围: 3.8-3.11** 
- **避免: Python 3.12+** (LangChain不兼容)
- 安装后验证：`python3 --version`

> 💡 **提示**: 如果已安装Python 3.12+，建议使用pyenv管理多版本

### 3️⃣ Get API Keys (Free) | 获取API密钥 (免费)

**English:**

**OpenAI API Key** (Required)
1. Visit: https://platform.openai.com/
2. Register/Login account
3. Click "API Keys" -> "Create new secret key"
4. Copy and save the key

**Pinecone API Key** (Optional, for document search)
1. Visit: https://app.pinecone.io/
2. Register/Login account
3. Create free project
4. Copy API key

**中文:**

**OpenAI API Key** (必需)
1. 访问：https://platform.openai.com/
2. 注册/登录账号
3. 点击 "API Keys" -> "Create new secret key"
4. 复制并保存密钥

**English:**

**Pinecone API Key** (Optional, for document search)
1. Visit: https://app.pinecone.io/
2. Sign up/Login
3. Create free project
4. Copy API key

**中文:**

**Pinecone API Key** (可选，用于文档搜索)
1. 访问：https://app.pinecone.io/
2. 注册/登录账号
3. 创建免费项目
4. 复制API密钥

## ⚡ One-Click Installation (2 minutes) | 一键安装 (2分钟)

**English:**
```bash
# Enter project directory
cd RAG-CHAT

# Run auto-installation script
./setup.sh
```

The installation script will automatically:
- ✅ Check system environment
- ✅ Install all dependencies
- ✅ Create configuration files
- ✅ Set permissions

**中文:**
```bash
# 进入项目目录
cd RAG-CHAT

# 运行自动安装脚本
./setup.sh
```

安装脚本会自动：
- ✅ 检查系统环境
- ✅ 安装所有依赖
- ✅ 创建配置文件
- ✅ 设置权限

## 🔧 Configure API Keys (1 minute) | 配置API密钥 (1分钟)

**English:**

Edit configuration file:
```bash
nano .env
```

**Basic Configuration** (Only these two items needed to run):
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key(optional)
```

Save file: `Ctrl + X` → `Y` → `Enter`

**中文:**

编辑配置文件：
```bash
nano .env
```

**最基本配置** (只需这两项就能运行)：
```env
OPENAI_API_KEY=你的OpenAI密钥
PINECONE_API_KEY=你的Pinecone密钥(可选)
```

保存文件：`Ctrl + X` → `Y` → `回车`

## 🚀 Launch Application (30 seconds) | 启动应用 (30秒)

**English:**
```bash
# One-click start all services
./start.sh
```

Wait for startup to complete, then visit: **http://localhost:5173**

**中文:**
```bash
# 一键启动所有服务
./start.sh
```

等待启动完成，然后访问：**http://localhost:5173**

## 🎉 Start Using | 开始使用

**English:**
1. **💬 Basic Chat**: Type questions directly, press Enter
2. **📁 Upload Files**: Click 📎 button to upload PDF/images
3. **🤖 Switch Models**: Select different AI models in top-right corner

**中文:**
1. **💬 基础聊天**：直接输入问题，按回车
2. **📁 上传文件**：点击📎按钮，上传PDF/图片
3. **🤖 切换模型**：右上角选择不同AI模型

## 🛑 Stop Application | 停止应用

**English:**
```bash
./stop.sh
```

**中文:**
```bash
./stop.sh
```

## ❓ Having Issues? | 遇到问题？

### Quick Fixes for Common Issues | 常见问题快速修复

**English:**

**Issue: Port Already in Use**
```bash
# Find occupying process
lsof -i :5173
lsof -i :8000

# Kill process
kill -9 <ProcessID>
```

**Issue: API Key Error**
- Check if keys in `.env` file are correct
- Confirm keys haven't expired
- View backend logs: `tail -f backend.log`

**Issue: Dependency Installation Failed**
```bash
# Check Python version
python3 --version

# If version incompatible, install correct version
# Method 1: Use pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv local 3.11.0

# Method 2: Clear cache and reinstall
pip3 cache purge
npm cache clean --force
./setup.sh
```

**Issue: LangChain Installation Failed**
```bash
# Usually a Python version issue
python3 --version  # Confirm version is between 3.8-3.11

# If version too high, downgrade Python or use virtual environment
python3 -m venv venv --python=python3.11
source venv/bin/activate
pip install -r backend/requirements.txt
```

**中文:**

**问题：端口被占用**
```bash
# 查找占用进程
lsof -i :5173
lsof -i :8000

# 终止进程
kill -9 <进程ID>
```

**问题：API密钥错误**
- 检查 `.env` 文件中的密钥是否正确
- 确认密钥没有过期
- 查看后端日志：`tail -f backend.log`

**问题：依赖安装失败**
```bash
# 检查Python版本
python3 --version

# 如果版本不兼容，安装正确版本
# 方法1: 使用pyenv (推荐)
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv local 3.11.0

# 方法2: 清理缓存重新安装
pip3 cache purge
npm cache clean --force
./setup.sh
```

**问题：LangChain安装失败**
```bash
# 通常是Python版本问题
python3 --version  # 确认版本在3.8-3.11之间

# 如果版本过高，降级Python或使用虚拟环境
python3 -m venv venv --python=python3.11
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 🆘 Get Help | 获得帮助

**English:**
- 📖 **Detailed Documentation**: Check [README.md](README.md)
- 🐛 **Issue Reports**: https://github.com/Ze-9527-0709/RAG-CHAT/issues
- 💬 **Community Discussions**: https://github.com/Ze-9527-0709/RAG-CHAT/discussions

**中文:**
- 📖 **详细文档**：查看 [README.md](README.md)
- 🐛 **问题反馈**：https://github.com/Ze-9527-0709/RAG-CHAT/issues
- 💬 **社区讨论**：https://github.com/Ze-9527-0709/RAG-CHAT/discussions

---

**English:** 🌟 **Tip**: First startup may take a few minutes to download dependencies, please be patient!

**中文:** 🌟 **小贴士**: 第一次启动可能需要几分钟下载依赖，请耐心等待！