# RAG Chat App 🤖💬

**Language / 语言**: [English](#english) | [中文](#中文)

---

## English

A modern RAG (Retrieval Augmented Generation) chat application supporting multiple AI models and file upload capabilities. Built with React + FastAPI architecture, featuring beautiful glassmorphism UI design.

> **🚀 New User?** Check out [5-Minute Quick Start Guide](QUICK_START.md) | **🛠️ Developer?** Continue reading full documentation

---

## 中文

一个现代化的RAG（检索增强生成）聊天应用，支持多种AI模型和文件上传功能。采用React + FastAPI架构，具有优美的玻璃拟态UI设计。

> **🚀 新用户？** 查看 [5分钟快速开始指南](QUICK_START.md) | **🛠️ 开发者？** 继续阅读完整文档

## ✨ Features | 功能特点

**English:**
- 🎨 **Modern UI Design**: Premium glassmorphism interface with advanced visual effects
- 🤖 **Multi-Model Support**: Compatible with GPT, Claude, Llava, and other AI models
- 📁 **File Upload**: Support for PDF, images, and document intelligent Q&A
- 🔍 **RAG Retrieval**: Smart document retrieval based on vector database
- 💬 **Real-time Chat**: Streaming responses with typing animations
- 🌐 **Cross-platform Deployment**: Docker containerization support

**中文:**
- 🎨 **现代化UI设计**：采用玻璃拟态效果的高端界面设计
- 🤖 **多模型支持**：支持GPT、Claude、Llava等多种AI模型
- 📁 **文件上传**：支持上传PDF、图片等文件进行智能问答
- 🔍 **RAG检索**：基于向量数据库的智能文档检索
- 💬 **实时聊天**：支持流式响应和打字效果
- 🌐 **跨平台部署**：支持Docker容器化部署

## 🚀 Quick Start | 快速开始

### System Requirements | 系统要求

**English:**
- **Node.js** 18.0+ 
- **Python** 3.8-3.11 ⚠️ **(Important: Python 3.12+ not supported due to LangChain compatibility issues)**
- **Git**
- **Docker** (Optional, for containerized deployment)

> **📌 Python Version Note**: AI dependencies like LangChain have strict version requirements. Python 3.10 or 3.11 recommended for best compatibility.

> **🚨 If your system has Python 3.12/3.13/3.14**: Run `./fix_python_env.sh` to auto-setup compatible environment, or check [Python Environment Guide](PYTHON_SETUP.md).

**中文:**
- **Node.js** 18.0+ 
- **Python** 3.8-3.11 ⚠️ **(重要：不支持Python 3.12+，LangChain兼容性问题)**
- **Git**
- **Docker** (可选，用于容器化部署)

> **📌 Python版本说明**: LangChain等AI依赖库对Python版本要求严格，推荐使用Python 3.10或3.11以获得最佳兼容性。

> **🚨 如果您的系统是Python 3.12/3.13/3.14**: 请运行 `./fix_python_env.sh` 自动设置兼容环境，或查看 [Python环境配置指南](PYTHON_SETUP.md)。

### 1. Clone Repository | 克隆项目

**English:**
```bash
git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
cd RAG-CHAT
```

**中文:**
```bash
git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
cd RAG-CHAT
```

### 2. Environment Setup | 环境配置

**English:**
Copy environment template and configure:

```bash
cp .env.example .env
```

Edit `.env` file with your API keys:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Vector Database Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# Other Settings
BACKEND_PORT=8000
FRONTEND_PORT=5173
```

**中文:**
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

### 3. Install Dependencies | 安装依赖

**English:**

#### Backend Dependencies (Python)

```bash
cd backend
pip install -r requirements.txt
```

#### Frontend Dependencies (Node.js)

```bash
cd frontend
npm install
```

**中文:**

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

### 4. Launch Application | 启动应用

**English:**

#### Method 1: Manual Launch (Recommended for Development)

**Start Backend:**
```bash
cd backend
python app.py
```
Backend will start at http://localhost:8000

**Start Frontend:**
```bash
cd frontend
npm run dev
```
Frontend will start at http://localhost:5173

#### Method 2: Script Launch

```bash
# Start all services
chmod +x start.sh
./start.sh

# Stop all services
chmod +x stop.sh
./stop.sh
```

#### Method 3: Docker Containerized Deployment

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# Stop all services
docker-compose down
```

**中文:**

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

### 5. Access Application | 访问应用

**English:**
Open browser and visit: http://localhost:5173

**中文:**
打开浏览器访问：http://localhost:5173

## 📋 Usage Guide | 使用说明

**English:**

### Basic Chat
1. Enter your question in the input box
2. Click send button or press Enter key
3. AI will provide real-time responses

### File Upload
1. Click 📎 button to select files
2. Supported formats: PDF, PNG, JPG, JPEG
3. Ask questions based on uploaded content

### Model Switching
1. Click model selector in top-right corner
2. Choose different AI models
3. Different models have different capabilities

**中文:**

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

## 🛠️ Project Structure | 项目结构

**English:**
```
RAG-Chat-App/
├── 📁 frontend/              # React frontend app
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── styles.css       # Glassmorphism styles
│   │   └── main.tsx         # Entry file
│   ├── package.json         # Frontend dependencies
│   └── vite.config.ts       # Vite build config
├── 📁 backend/               # FastAPI backend service
│   ├── app.py              # Main application server
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container config
├── 📁 ingest/               # Document ingestion service
│   └── ingest.py           # Document processing script
├── 📁 docs/                 # User documentation directory
├── 📁 uploads/              # File upload storage
├── 🐳 docker-compose.yml    # Container orchestration config
├── ⚙️  .env.example          # Environment variables template
├── 🚀 setup.sh              # Automated installation script
├── 🔧 fix_python_env.sh     # Python environment fixer
├── 📖 README.md             # Main project documentation
├── 🚀 QUICK_START.md        # 5-minute quick start
└── 🐍 PYTHON_SETUP.md       # Python environment guide
```

**中文:**
```
RAG-Chat-App/
├── 📁 frontend/              # React前端应用
│   ├── src/
│   │   ├── App.tsx          # 主React组件
│   │   ├── styles.css       # 玻璃拟态样式
│   │   └── main.tsx         # 入口文件
│   ├── package.json         # 前端依赖配置
│   └── vite.config.ts       # Vite构建配置
├── 📁 backend/               # FastAPI后端服务
│   ├── app.py              # 主应用服务器
│   ├── requirements.txt    # Python依赖列表
│   └── Dockerfile          # 后端容器配置
├── 📁 ingest/               # 文档摄取服务
│   └── ingest.py           # 文档处理脚本
├── 📁 docs/                 # 用户文档目录
├── 📁 uploads/              # 文件上传存储
├── 🐳 docker-compose.yml    # 容器编排配置
├── ⚙️  .env.example          # 环境变量模板
├── 🚀 setup.sh              # 自动安装脚本
├── 🔧 fix_python_env.sh     # Python环境修复
├── 📖 README.md             # 项目主文档
├── 🚀 QUICK_START.md        # 5分钟快速开始
└── 🐍 PYTHON_SETUP.md       # Python环境指南
```

## 🔧 Advanced Configuration | 高级配置

**English:**

### API Key Acquisition

1. **OpenAI API Key**: Visit https://platform.openai.com/api-keys
2. **Pinecone API Key**: Visit https://app.pinecone.io/

### Custom Configuration

Edit `.env` file to customize:
- API endpoint URLs
- Model parameters
- Vector database configuration
- Service port numbers

**中文:**

### API密钥获取

1. **OpenAI API Key**: 访问 https://platform.openai.com/api-keys
2. **Pinecone API Key**: 访问 https://app.pinecone.io/

### 自定义配置

编辑 `.env` 文件可以自定义：
- API端点URL
- 模型参数
- 向量数据库配置
- 服务端口号

### Docker Deployment Options | Docker部署选项

**English:**
```bash
# Build without starting
docker-compose build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

**中文:**
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

## 🐛 Troubleshooting | 故障排除

### Common Issues | 常见问题

**English:**

**1. Port Already in Use**
```bash
# Check port usage
lsof -i :5173
lsof -i :8000

# Kill occupying process
kill -9 <PID>
```

**2. Dependency Installation Failed**
```bash
# First check Python version (Must be 3.8-3.11)
python3 --version

# LangChain related errors are usually version issues
# Solution 1: Use pyenv to manage Python versions
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv local 3.11.0

# Solution 2: Use virtual environment
python3 -m venv venv --python=python3.11
source venv/bin/activate

# Clear cache
npm cache clean --force
pip cache purge
```

**3. API Key Errors**
- Check `.env` file configuration
- Verify API key validity
- Check backend logs for detailed error information

**中文:**

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
# 首先检查Python版本 (必须是3.8-3.11)
python3 --version

# LangChain相关错误通常是版本问题
# 解决方案1: 使用pyenv管理Python版本
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv local 3.11.0

# 解决方案2: 使用虚拟环境
python3 -m venv venv --python=python3.11
source venv/bin/activate

# 清理缓存
npm cache clean --force
pip cache purge
```

**3. API密钥错误**
- 检查 `.env` 文件配置
- 确认API密钥的有效性
- 查看后端日志获取详细错误信息

### Log Viewing | 日志查看

**English:**
```bash
# View backend logs
tail -f backend.log

# View frontend logs
tail -f frontend.log

# View Docker logs
docker-compose logs backend
docker-compose logs frontend
```

**中文:**
```bash
# 查看后端日志
tail -f backend.log

# 查看前端日志
tail -f frontend.log

# 查看Docker日志
docker-compose logs backend
docker-compose logs frontend
```

## 🤝 Contributing | 贡献指南

**English:**
Issues and Pull Requests are welcome!

1. Fork the project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Create Pull Request

**中文:**
欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 License | 许可证

**English:**
This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

**中文:**
本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 Support & Contact | 支持与联系

**English:**
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Ze-9527-0709/RAG-CHAT/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Ze-9527-0709/RAG-CHAT/discussions)

**中文:**
- 🐛 **问题反馈**：[GitHub Issues](https://github.com/Ze-9527-0709/RAG-CHAT/issues)
- 💬 **讨论交流**：[GitHub Discussions](https://github.com/Ze-9527-0709/RAG-CHAT/discussions)

---

**English:** ⭐ If this project helps you, please give it a star!

**中文:** ⭐ 如果这个项目对您有帮助，请给一个星星！