# 🚀 RAG Chat App - 5分钟快速开始指南

**语言**: [English](../QUICK_START.md) | [中文](QUICK_START.md)

---

> **新手友好** | **无需复杂配置** | **一键启动**

## 🎯 开始前准备 (5分钟)

### 1️⃣ 下载代码

```bash
# 方法一：使用Git (推荐)
git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
cd RAG-CHAT

# 方法二：直接下载
# 访问 https://github.com/Ze-9527-0709/RAG-CHAT
# 点击 "Code" -> "Download ZIP" -> 解压到本地
```

### 2️⃣ 安装必需软件

**📦 Node.js** (JavaScript运行环境)
- 下载地址：https://nodejs.org/
- 选择LTS版本 (推荐18.x+)
- 安装后验证：`node -v`

**🐍 Python** (AI后端语言)
- 下载地址：https://python.org/downloads/
- **重要**：选择版本3.8-3.11 (LangChain兼容性)
- ⚠️ **避免Python 3.12+** (会导致依赖冲突)
- 验证安装：`python --version` 或 `python3 --version`

> **🚨 Python版本提醒**：如果您已经安装了Python 3.12/3.13/3.14，请运行 `./fix_python_env.sh` 自动设置兼容环境。

## 🚀 一键安装脚本

### 选项1：自动化脚本 (推荐)

```bash
# 赋予执行权限
chmod +x setup.sh

# 运行自动化设置
./setup.sh

# 启动应用
./start.sh
```

**脚本功能：**
1. ✅ 检查系统依赖
2. ✅ 创建Python虚拟环境
3. ✅ 安装所有必需包
4. ✅ 设置环境配置
5. ✅ 启动后端和前端服务

### 选项2：手动安装

如果自动脚本失败，请按照手动步骤：

#### 后端设置
```bash
cd backend

# 创建虚拟环境
python -m venv venv

# 激活环境
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 前端设置
```bash
cd frontend

# 安装Node.js依赖
npm install
```

## ⚙️ API配置 (2分钟)

### 获取API密钥

1. **OpenAI API Key** (聊天功能必需)
   - 访问：https://platform.openai.com/api-keys
   - 创建新的API密钥
   - 复制密钥 (以`sk-`开头)

2. **Pinecone API Key** (可选，用于文档上传)
   - 访问：https://app.pinecone.io/
   - 创建免费账户
   - 从控制台获取API密钥

### 配置环境

```bash
# 复制配置模板
cp backend/.env.example backend/.env

# 编辑配置文件
nano backend/.env  # 或使用您偏好的编辑器
```

**最小配置 (仅聊天)：**
```env
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
```

**完整配置 (聊天 + 文档上传)：**
```env
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=rag-chat-index
```

## 🎮 启动应用 (1分钟)

### 方法1：脚本启动 (最简单)
```bash
# 启动所有服务
./start.sh

# 应用将在以下地址可用：
# 前端：http://localhost:5173
# 后端API：http://localhost:8000
```

### 方法2：手动启动
```bash
# 终端1：启动后端
cd backend
source venv/bin/activate  # 激活Python环境
python app.py

# 终端2：启动前端
cd frontend
npm run dev
```

### 方法3：Docker启动
```bash
# 确保Docker已安装并运行
docker-compose up -d

# 检查状态
docker-compose ps
```

## 💬 开始聊天！

1. **打开浏览器**：访问 http://localhost:5173
2. **基础聊天**：输入您的问题并按回车
3. **上传文件**：点击📎按钮上传PDF/图片
4. **模型选择**：点击模型下拉菜单切换AI模型

## 🔧 快速故障排除

### 常见问题

**端口被占用**
```bash
# 终止使用5173或8000端口的进程
lsof -ti:5173 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

**Python版本问题**
```bash
# 检查Python版本
python --version

# 如果是Python 3.12+，使用此修复
./fix_python_env.sh
```

**API密钥错误**
- 仔细检查您的`.env`文件配置
- 确保API密钥周围没有多余空格
- 在提供商网站上验证API密钥有效性

**前端无法加载**
```bash
# 清理Node.js缓存
cd frontend
npm cache clean --force
npm install
```

**后端崩溃**
```bash
# 检查后端日志
cd backend
python app.py

# 查找具体错误消息
```

### 性能提示

- **使用gpt-4o-mini模型** 获得更快响应
- **上传较小文件** (< 10MB) 以获得更好性能
- **重启服务** 如果响应变慢

## 🎯 下一步

**探索功能：**
- 📁 上传不同文件类型 (PDF, 图片)
- 🤖 尝试不同AI模型
- 💬 测试流式聊天响应
- 🔍 询问关于上传文档的问题

**高级配置：**
- 📖 阅读 [完整文档](../README.md)
- 🐍 查看 [Python环境指南](PYTHON_SETUP.md)
- 🛠️ 探索自定义选项

**需要帮助？**
- 📋 查看 [故障排除指南](../README.md#故障排除)
- 💡 浏览常见问题和解决方案
- 🚀 加入我们的社区讨论

---

**🎉 恭喜！您的RAG聊天应用现在正在运行！**

访问 http://localhost:5173 开始与AI聊天吧！🤖✨