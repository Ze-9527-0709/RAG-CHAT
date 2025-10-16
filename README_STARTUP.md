# 🚀 RAG Chat App 启动指南

## 🎯 快速启动

### 方式一：一键启动 (推荐)
```bash
# 赋予执行权限（只需首次运行）
chmod +x start.sh stop.sh install_local_model.sh

# 测试模式 (lightweight mock 后端)
./start.sh

# 完整 RAG 模式 (智能模型降级系统)
./start.sh full

# 停止所有服务
./stop.sh
```

### 方式二：手动启动
```bash
# 1. 启动后端 (选择一个)
cd backend
python mock_app.py      # 测试模式
python app.py           # 完整 RAG 模式

# 2. 启动前端 (新终端)  
cd frontend
npm run dev
```

## 🧠 智能模型系统

新增了**智能模型降级功能**，根据 OpenAI 余额自动选择最优模型：
- **GPT-4o-mini** (余额充足时)
- **GPT-3.5 Turbo** (余额较低时) 
- **本地 Llama2** (完全免费备用)

详细信息请查看 [MODEL_FALLBACK.md](./MODEL_FALLBACK.md)

## 📋 系统要求

- **Python 3.8+** 
- **Node.js 18+**
- **环境变量**: OpenAI API Key, Pinecone API Key (仅完整模式)
- **可选**: Ollama (本地模型支持)

## 🛠️ 首次设置

### 1. 安装依赖
```bash
# Python 依赖
cd backend
pip install -r requirements.txt

# Node.js 依赖  
cd ../frontend
npm install
```

### 2. 环境配置
```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件
nano .env
```

**基础配置 (测试模式):**
```bash
# 仅前端功能测试，无需 API Key
MODE=mock
```

**完整配置 (RAG 模式):**
```bash
# OpenAI 配置
OPENAI_API_KEY=your_openai_key
OPENAI_ESTIMATED_BALANCE=25.50

# 智能降级设置
MIN_OPENAI_BALANCE=5.00
MAX_MODEL_RETRIES=2

# Pinecone 配置 (RAG 功能)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment

# 本地模型 (可选)
LOCAL_MODEL_ENDPOINT=http://localhost:11434/api/generate
LOCAL_MODEL_NAME=llama2:7b-chat
```

### 3. 本地模型安装 (可选但推荐)
```bash
# 自动安装 Ollama 和模型
./install_local_model.sh

# 验证安装
curl http://localhost:11434/api/tags
```

## 🌐 访问应用

启动成功后，在浏览器访问：
- **前端 UI**: http://localhost:5173
- **后端 API**: http://localhost:8000

## 📊 查看日志

```bash
# 后端日志
tail -f backend.log

# 前端日志
tail -f frontend.log
```

## 🔧 当前配置

- **后端**: 模拟 API（mock_app.py）
  - 不需要 OpenAI API Key
  - 不需要 Pinecone 配置
  - 返回模拟的 Markdown 响应

- **前端**: React + Vite
  - 浅色主题 UI
  - 支持多会话管理
  - Markdown 渲染
  - 流式响应

## 🔄 切换到完整 RAG 后端

如果需要使用真实的 RAG 功能：

1. 安装完整依赖：
```bash
cd backend
venv/bin/pip install -r requirements.txt
```

2. 配置环境变量（创建 `.env` 文件）：
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

3. 修改 `start.sh` 中的启动命令：
```bash
# 将这行
./venv/bin/uvicorn mock_app:app --port 8000

# 改为
./venv/bin/uvicorn app:app --port 8000
```

## ⚠️ 常见问题

**端口被占用**
```bash
# 查看占用端口的进程
lsof -i :8000
lsof -i :5173

# 强制停止
./stop.sh
```

**虚拟环境问题**
```bash
# 重新创建虚拟环境
cd backend
rm -rf venv
python3.13 -m venv venv
venv/bin/pip install -q fastapi 'uvicorn[standard]' python-dotenv openai
```

## 📱 功能特性

✅ 浅色主题 UI（Copilot 风格）  
✅ 多会话管理  
✅ Markdown 消息渲染  
✅ 流式响应动画  
✅ 文档引用显示  
✅ 响应式设计  
