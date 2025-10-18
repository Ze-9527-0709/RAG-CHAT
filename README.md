# RAG Chat App 🤖💬

**Language**: [English](README.md) | [中文](中文指南/README.md)

---

A modern RAG (Retrieval Augmented Generation) chat application supporting multiple AI models and file upload capabilities. Built with React + FastAPI architecture, featuring beautiful glassmorphism UI design.

> **🚀 New User?** Check out [5-Minute Quick Start Guide](QUICK_START.md) | **🛠️ Developer?** Continue reading full documentation

---

## ✨ Features

- 🎨 **Modern UI Design**: Premium glassmorphism interface with advanced visual effects
- 🤖 **Multi-Model Support**: Compatible with GPT, Claude, Llava, and other AI models
- 📁 **File Upload**: Support for PDF, images, and document intelligent Q&A
- 🔍 **RAG Retrieval**: Smart document retrieval based on vector database
- 💬 **Real-time Chat**: Streaming responses with typing animations
- 🌐 **Cross-platform Deployment**: Docker containerization support

## 🚀 Quick Start

### System Requirements

- **Node.js** 18.0+ 
- **Python** 3.8-3.11 ⚠️ **(Important: Python 3.12+ not supported due to LangChain compatibility issues)**
- **Git**
- **Docker** (Optional, for containerized deployment)

> **📌 Python Version Note**: AI dependencies like LangChain have strict version requirements. Python 3.10 or 3.11 recommended for best compatibility.

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
   cd RAG-CHAT
   ```

2. **Environment Setup**
   ```bash
   # Backend setup
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

   # Frontend setup
   cd ../frontend
   npm install
   ```

3. **Configure API Keys**
   ```bash
   # Copy environment template
   cp backend/.env.example backend/.env
   
   # Edit .env file with your API keys
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index_name
   ```

4. **Start Services**
   ```bash
   # Start backend (Terminal 1)
   cd backend && python app.py
   
   # Start frontend (Terminal 2)
   cd frontend && npm run dev
   ```

5. **Access Application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

## 📖 Documentation

- [Quick Start Guide](QUICK_START.md) - 5-minute setup guide
- [Python Environment Setup](PYTHON_SETUP.md) - Detailed Python configuration
- [Python 3.14 Users Guide](PYTHON_3.14_USERS.md) - Special instructions for Python 3.14+
- [中文指南](中文指南/) - Complete Chinese documentation

## 🛠️ Architecture

### Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite for fast development
- Tailwind CSS for styling
- Glassmorphism UI effects

**Backend:**
- FastAPI (Python)
- OpenAI GPT integration
- Pinecone vector database
- LangChain for RAG
- Streaming responses

### Project Structure

```
RAG-Chat-App/
├── frontend/          # React frontend
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── backend/           # FastAPI backend
│   ├── app.py
│   ├── requirements.txt
│   └── .env.example
├── ingest/            # Document ingestion
│   └── ingest.py
└── docker-compose.yml # Container deployment
```

## 🐳 Docker Deployment

### Quick Deploy with Docker Compose

1. **Set Environment Variables**
   ```bash
   cp backend/.env.example backend/.env
   # Edit .env with your API keys
   ```

2. **Build and Start**
   ```bash
   docker-compose up -d
   ```

3. **Access Application**
   - Application: http://localhost:5173
   - API Documentation: http://localhost:8000/docs

### Manual Docker Build

```bash
# Build backend
cd backend
docker build -t rag-chat-backend .

# Build frontend
cd ../frontend
docker build -t rag-chat-frontend .

# Run containers
docker run -d -p 8000:8000 --env-file backend/.env rag-chat-backend
docker run -d -p 5173:80 rag-chat-frontend
```

## 📁 File Upload & RAG

### Supported File Types

- **Documents**: PDF, TXT, MD
- **Images**: PNG, JPG, JPEG, GIF
- **Archives**: ZIP (auto-extract)

### Document Processing Pipeline

1. **File Upload** → Frontend sends file to backend
2. **Text Extraction** → Extract text content from files
3. **Chunking** → Split text into manageable chunks
4. **Embedding** → Generate vector embeddings
5. **Storage** → Store in Pinecone vector database
6. **Retrieval** → Query similar chunks for context
7. **Generation** → LLM generates contextual responses

### RAG Configuration

```python
# Vector store configuration
INDEX_NAME = "your-pinecone-index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval parameters
TOP_K = 4  # Number of similar chunks to retrieve
SIMILARITY_THRESHOLD = 0.7
```

## 🔧 Configuration

### Environment Variables

Create `backend/.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=rag-chat-index

# Optional: Custom embedding model
EMBEDDING_MODEL=text-embedding-3-small
```

### Model Configuration

The app supports multiple AI models:

```python
# Available models
SUPPORTED_MODELS = {
    "gpt-4o-mini": "OpenAI GPT-4o Mini",
    "gpt-4": "OpenAI GPT-4",
    "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
    "claude-3": "Anthropic Claude-3",
}
```

## 🧪 Testing

### Backend Testing

```bash
cd backend
python -m pytest tests/
```

### Frontend Testing

```bash
cd frontend
npm test
```

### API Testing

Test the backend API endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "Hello!"}'
```

## 🐛 Troubleshooting

### Common Issues

**Python Version Compatibility**
- **Problem**: LangChain compatibility issues
- **Solution**: Use Python 3.8-3.11, avoid 3.12+

**Pinecone Connection Errors**
- **Problem**: Invalid API key or index name
- **Solution**: Check `.env` configuration

**OpenAI API Errors**
- **Problem**: Rate limits or invalid key
- **Solution**: Verify API key and billing status

**Frontend Build Issues**
- **Problem**: Node version incompatibility
- **Solution**: Use Node.js 18.0+

### Performance Optimization

**Vector Database**
- Use appropriate index dimensions
- Optimize chunk size for your use case
- Monitor Pinecone usage quotas

**Response Speed**
- Enable streaming responses
- Use faster embedding models
- Implement response caching

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit Changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to Branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open Pull Request**

### Development Guidelines

- Follow Python PEP 8 style guide
- Use TypeScript for frontend development
- Add tests for new features
- Update documentation for changes

### Code Style

```bash
# Python formatting
black backend/
flake8 backend/

# TypeScript formatting
cd frontend && npm run format
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **Pinecone** for vector database
- **LangChain** for RAG framework
- **React** and **FastAPI** communities

---

**⭐ If you find this project helpful, please give it a star!**

For more detailed setup instructions, see our [Quick Start Guide](QUICK_START.md).