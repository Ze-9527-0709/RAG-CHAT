# RAG Chat App 🤖💬

一个现代化的RAG（检索增强生成）聊天应用，支持多种AI模型和文件上传功能。采用React + FastAPI架构，具有优美的玻璃拟态UI设计。

> **🚀 新用户？** 查看 [5分钟快速开始指南](QUICK_START.md) | **🛠️ 开发者？** 继续阅读完整文档

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
- **Python** 3.8+
- **Git**
- **Docker** (可选，用于容器化部署)

### 1. 克隆项目

```bash
git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
cd RAG-CHAT
```

### 2. 环境配置

复制环境变量模板并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置必要的API密钥：

```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone向量数据库配置
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# 其他配置
BACKEND_PORT=8000
FRONTEND_PORT=5173
```

### 3. 安装依赖

#### 后端依赖 (Python)

```bash
cd backend
pip install -r requirements.txt
```

#### 前端依赖 (Node.js)

```bash
cd frontend
npm install
```

### 4. 启动应用

#### 方法一：手动启动 (推荐用于开发)

**启动后端：**
```bash
cd backend
python app.py
```
后端将在 http://localhost:8000 启动

**启动前端：**
```bash
cd frontend
npm run dev
```
前端将在 http://localhost:5173 启动

#### 方法二：使用脚本启动

```bash
# 启动所有服务
chmod +x start.sh
./start.sh

# 停止所有服务
chmod +x stop.sh
./stop.sh
```

#### 方法三：Docker容器化部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 停止所有服务
docker-compose down
```

### 5. 访问应用

打开浏览器访问：http://localhost:5173

## 📋 使用说明

### 基础聊天
1. 在输入框中输入您的问题
2. 点击发送按钮或按回车键
3. AI将为您提供实时回答

### 文件上传
1. 点击📎按钮选择文件
2. 支持的格式：PDF、PNG、JPG、JPEG
3. 上传后可以基于文件内容进行问答

### 模型切换
1. 点击右上角的模型选择器
2. 选择不同的AI模型
3. 不同模型具有不同的能力特点

## 🛠️ 项目结构

```
RAG-Chat-App/
├── frontend/              # React前端应用
│   ├── src/
│   │   ├── App.tsx       # 主要组件
│   │   └── styles.css    # 样式文件
│   ├── package.json
│   └── vite.config.ts
├── backend/              # FastAPI后端服务
│   ├── app.py           # 主应用文件
│   ├── requirements.txt # Python依赖
│   └── Dockerfile
├── docs/                # 文档存储目录
├── uploads/             # 文件上传目录
├── docker-compose.yml   # Docker配置
├── .env.example        # 环境变量模板
└── README.md           # 项目说明
```

## 🔧 高级配置

### API密钥获取

1. **OpenAI API Key**: 访问 https://platform.openai.com/api-keys
2. **Pinecone API Key**: 访问 https://app.pinecone.io/

### 自定义配置

编辑 `.env` 文件可以自定义：
- API端点URL
- 模型参数
- 向量数据库配置
- 服务端口号

### Docker部署选项

```bash
# 仅构建不启动
docker-compose build

# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart
```

## 🐛 故障排除

### 常见问题

**1. 端口被占用**
```bash
# 查看端口占用
lsof -i :5173
lsof -i :8000

# 终止占用进程
kill -9 <PID>
```

**2. 依赖安装失败**
```bash
# 清理npm缓存
npm cache clean --force

# 清理Python缓存
pip cache purge
```

**3. API密钥错误**
- 检查 `.env` 文件配置
- 确认API密钥的有效性
- 查看后端日志获取详细错误信息

### 日志查看

```bash
# 查看后端日志
tail -f backend.log

# 查看前端日志
tail -f frontend.log

# 查看Docker日志
docker-compose logs backend
docker-compose logs frontend
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 支持与联系

- 🐛 **问题反馈**：[GitHub Issues](https://github.com/Ze-9527-0709/RAG-CHAT/issues)
- 💬 **讨论交流**：[GitHub Discussions](https://github.com/Ze-9527-0709/RAG-CHAT/discussions)

---

⭐ 如果这个项目对您有帮助，请给一个星星！
