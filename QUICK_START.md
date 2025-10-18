# 🚀 RAG Chat App - 5-Minute Quick Start Guide

**Language**: [English](QUICK_START.md) | [中文](中文指南/QUICK_START.md)

---

> **Beginner Friendly** | **No Complex Setup** | **One-Click Launch**

## 🎯 Preparation (5 minutes)

### 1️⃣ Download Code

```bash
# Method 1: Using Git (Recommended)
git clone https://github.com/Ze-9527-0709/RAG-CHAT.git
cd RAG-CHAT

# Method 2: Direct Download
# Visit https://github.com/Ze-9527-0709/RAG-CHAT
# Click "Code" -> "Download ZIP" -> Extract locally
```

### 2️⃣ Install Required Software

**📦 Node.js** (JavaScript Runtime)
- Download: https://nodejs.org/
- Choose LTS version (Recommended 18.x+)
- Verify after installation: `node -v`

**🐍 Python** (AI Backend Language)
- Download: https://python.org/downloads/
- **Important**: Choose version 3.8-3.11 (LangChain compatibility)
- ⚠️ **Avoid Python 3.12+** (will cause dependency conflicts)
- Verify: `python --version` or `python3 --version`

> **🚨 Python Version Alert**: If you already have Python 3.12/3.13/3.14, run `./fix_python_env.sh` to auto-setup compatible environment.

## 🚀 One-Click Installation Scripts

### Option 1: Automated Script (Recommended)

```bash
# Grant execution permission
chmod +x setup.sh

# Run automated setup
./setup.sh

# Start application
./start.sh
```

**What the script does:**
1. ✅ Checks system dependencies
2. ✅ Creates Python virtual environment
3. ✅ Installs all required packages
4. ✅ Sets up environment configuration
5. ✅ Launches backend and frontend services

### Option 2: Manual Installation

If automated script fails, follow manual steps:

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend

# Install Node.js dependencies
npm install
```

## ⚙️ API Configuration (2 minutes)

### Get API Keys

1. **OpenAI API Key** (Required for AI chat)
   - Visit: https://platform.openai.com/api-keys
   - Create new API key
   - Copy the key (starts with `sk-`)

2. **Pinecone API Key** (Optional, for document upload)
   - Visit: https://app.pinecone.io/
   - Create free account
   - Get API key from dashboard

### Configure Environment

```bash
# Copy configuration template
cp backend/.env.example backend/.env

# Edit configuration file
nano backend/.env  # or use your preferred editor
```

**Minimal Configuration (Chat only):**
```env
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
```

**Full Configuration (Chat + Document upload):**
```env
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=rag-chat-index
```

## 🎮 Launch Application (1 minute)

### Method 1: Script Launch (Easiest)
```bash
# Start all services
./start.sh

# Application will be available at:
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
```

### Method 2: Manual Launch
```bash
# Terminal 1: Start Backend
cd backend
source venv/bin/activate  # Activate Python environment
python app.py

# Terminal 2: Start Frontend
cd frontend
npm run dev
```

### Method 3: Docker Launch
```bash
# Make sure Docker is installed and running
docker-compose up -d

# Check status
docker-compose ps
```

## 💬 Start Chatting!

1. **Open Browser**: Visit http://localhost:5173
2. **Basic Chat**: Type your question and press Enter
3. **Upload Files**: Click 📎 button to upload PDF/images
4. **Model Selection**: Click model dropdown to switch AI models

## 🔧 Quick Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Kill processes using ports 5173 or 8000
lsof -ti:5173 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

**Python Version Issues**
```bash
# Check Python version
python --version

# If Python 3.12+, use this fix
./fix_python_env.sh
```

**API Key Errors**
- Double-check your `.env` file configuration
- Ensure no extra spaces around API keys
- Verify API key validity on provider websites

**Frontend Won't Load**
```bash
# Clear Node.js cache
cd frontend
npm cache clean --force
npm install
```

**Backend Crashes**
```bash
# Check backend logs
cd backend
python app.py

# Look for specific error messages
```

### Performance Tips

- **Use gpt-4o-mini model** for faster responses
- **Upload smaller files** (< 10MB) for better performance
- **Restart services** if responses become slow

## 🎯 Next Steps

**Explore Features:**
- 📁 Upload different file types (PDF, images)
- 🤖 Try different AI models
- 💬 Test streaming chat responses
- 🔍 Ask questions about uploaded documents

**Advanced Configuration:**
- 📖 Read [Full Documentation](README.md)
- 🐍 Check [Python Environment Guide](PYTHON_SETUP.md)
- 🛠️ Explore customization options

**Need Help?**
- 📋 Check [Troubleshooting Guide](README.md#troubleshooting)
- 💡 Browse common issues and solutions
- 🚀 Join our community discussions

---

**🎉 Congratulations! Your RAG Chat App is now running!**

Visit http://localhost:5173 and start chatting with AI! 🤖✨