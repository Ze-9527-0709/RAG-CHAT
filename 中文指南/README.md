# RAG Chat App 🤖💬

**语言**: [English](../README.md) | [中文](README.md)

---

一个现代化的RAG（检索增强生成）聊天应用，支持多种AI模型和文件上传功能。采用React + FastAPI架构，具有优美的玻璃拟态UI设计。

> **🚀 新用户？** 查看 [5分钟快速开始指南](QUICK_START.md) | **🛠️ 开发者？** 继续阅读完整文档

---

## ✨ 功能特点

- 🎨 **现代化UI设计**：采用玻璃拟态效果的高端界面设计
- 🤖 **多模型支持**：支持GPT、Claude、Llava等多种AI模型
- 📁 **文件上传**：支持上传PDF、图片等文件进行智能问答
- 🔍 **RAG检索**：基于向量数据库的智能文档检索
- 💬 **实时聊天**：支持流式响应和打字效果
- 🌐 **跨平台部署**：支持Docker容器化部署

## 🚀 快速开始

### 系统要求

- **Node.js** 18.0+ 
- **Python** 3.8-3.11 ⚠️ **（重要：由于LangChain兼容性问题，不支持Python 3.12+）**
- **Git**
- **Docker** （可选，用于容器化部署）

> **📌 Python版本说明**：AI依赖项如LangChain对版本有严格要求。推荐使用Python 3.10或3.11以获得最佳兼容性。

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
   cd RAG-CHAT
   ```

2. **环境设置**
   ```bash
   # 后端设置
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

   # 前端设置
   cd ../frontend
   npm install
   ```

3. **配置API密钥**
   ```bash
   # 复制环境变量模板
   cp backend/.env.example backend/.env
   
   # 编辑.env文件，填入你的API密钥
   OPENAI_API_KEY=你的openai密钥
   PINECONE_API_KEY=你的pinecone密钥
   PINECONE_INDEX_NAME=你的索引名称
   ```

4. **启动服务**
   ```bash
   # 启动后端（终端1）
   cd backend && python app.py
   
   # 启动前端（终端2）
   cd frontend && npm run dev
   ```

5. **访问应用**
   - 前端：http://localhost:5173
   - 后端API：http://localhost:8000

## 📖 文档

- [快速开始指南](QUICK_START.md) - 5分钟设置指南
- [Python环境设置](PYTHON_SETUP.md) - 详细的Python配置
- [Python 3.14用户指南](PYTHON_3.14_USERS.md) - Python 3.14+特殊说明

## 🛠️ 架构

### 技术栈

**前端：**
- React 18 + TypeScript
- Vite快速开发
- Tailwind CSS样式
- 玻璃拟态UI效果

**后端：**
- FastAPI (Python)
- OpenAI GPT集成
- Pinecone向量数据库
- LangChain用于RAG
- 流式响应

### 项目结构

```
RAG-Chat-App/
├── frontend/          # React前端
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── backend/           # FastAPI后端
│   ├── app.py
│   ├── requirements.txt
│   └── .env.example
├── ingest/            # 文档摄取
│   └── ingest.py
└── docker-compose.yml # 容器部署
```

## 🐳 Docker部署

### 使用Docker Compose快速部署

1. **设置环境变量**
   ```bash
   cp backend/.env.example backend/.env
   # 编辑.env填入你的API密钥
   ```

2. **构建并启动**
   ```bash
   docker-compose up -d
   ```

3. **访问应用**
   - 应用：http://localhost:5173
   - API文档：http://localhost:8000/docs

### 手动Docker构建

```bash
# 构建后端
cd backend
docker build -t rag-chat-backend .

# 构建前端
cd ../frontend
docker build -t rag-chat-frontend .

# 运行容器
docker run -d -p 8000:8000 --env-file backend/.env rag-chat-backend
docker run -d -p 5173:80 rag-chat-frontend
```

## 📁 文件上传与RAG

### 支持的文件类型

- **文档**：PDF, TXT, MD
- **图片**：PNG, JPG, JPEG, GIF
- **压缩包**：ZIP（自动解压）

### 文档处理流水线

1. **文件上传** → 前端发送文件到后端
2. **文本提取** → 从文件中提取文本内容
3. **分块** → 将文本分割成可管理的块
4. **嵌入** → 生成向量嵌入
5. **存储** → 存储到Pinecone向量数据库
6. **检索** → 查询相似块作为上下文
7. **生成** → LLM生成上下文相关的响应

### RAG配置

```python
# 向量存储配置
INDEX_NAME = "your-pinecone-index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 检索参数
TOP_K = 4  # 检索相似块的数量
SIMILARITY_THRESHOLD = 0.7
```

## 🔧 配置

### 环境变量

创建 `backend/.env` 文件：

```env
# OpenAI配置
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Pinecone配置
PINECONE_API_KEY=你的pinecone密钥
PINECONE_INDEX_NAME=rag-chat-index

# 可选：自定义嵌入模型
EMBEDDING_MODEL=text-embedding-3-small
```

### 模型配置

应用支持多种AI模型：

```python
# 可用模型
SUPPORTED_MODELS = {
    "gpt-4o-mini": "OpenAI GPT-4o Mini",
    "gpt-4": "OpenAI GPT-4",
    "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
    "claude-3": "Anthropic Claude-3",
}
```

## 🧪 测试

### 后端测试

```bash
cd backend
python -m pytest tests/
```

### 前端测试

```bash
cd frontend
npm test
```

### API测试

测试后端API端点：

```bash
# 健康检查
curl http://localhost:8000/health

# 聊天端点
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "你好！"}'
```

## 🐛 故障排除

### 常见问题

**Python版本兼容性**
- **问题**：LangChain兼容性问题
- **解决方案**：使用Python 3.8-3.11，避免3.12+

**Pinecone连接错误**
- **问题**：无效的API密钥或索引名称
- **解决方案**：检查`.env`配置

**OpenAI API错误**
- **问题**：速率限制或无效密钥
- **解决方案**：验证API密钥和计费状态

**前端构建问题**
- **问题**：Node版本不兼容
- **解决方案**：使用Node.js 18.0+

### 性能优化

**向量数据库**
- 使用适当的索引维度
- 针对你的用例优化块大小
- 监控Pinecone使用配额

**响应速度**
- 启用流式响应
- 使用更快的嵌入模型
- 实现响应缓存

## 🤝 贡献

我们欢迎贡献！请按照以下步骤：

1. **Fork仓库**
2. **创建功能分支**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **提交更改**
   ```bash
   git commit -m "添加惊人功能"
   ```
4. **推送到分支**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **开启Pull Request**

### 开发指南

- 遵循Python PEP 8风格指南
- 前端开发使用TypeScript
- 为新功能添加测试
- 更新相关文档

### 代码风格

```bash
# Python格式化
black backend/
flake8 backend/

# TypeScript格式化
cd frontend && npm run format
```

## 📄 许可证

该项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **OpenAI** 提供GPT模型
- **Pinecone** 提供向量数据库
- **LangChain** 提供RAG框架
- **React** 和 **FastAPI** 社区

---

**⭐ 如果你觉得这个项目有帮助，请给它一个星标！**

更详细的设置说明，请查看我们的 [快速开始指南](QUICK_START.md)。