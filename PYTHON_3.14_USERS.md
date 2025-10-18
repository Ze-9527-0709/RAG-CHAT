# 🐍 Python 3.14+ Users Guide

**Language**: [English](PYTHON_3.14_USERS.md) | [中文](中文指南/PYTHON_3.14_USERS.md)

---

> **Special Setup Guide for Python 3.14+ Users**

If your system has Python 3.14 or newer installed, this guide will help you set up a compatible environment for the RAG Chat application.

## 🚨 Why Python 3.14+ Won't Work

### Compatibility Issues

The RAG Chat application relies on several AI/ML libraries that haven't been updated for Python 3.14+:

- **LangChain**: Core RAG functionality - requires Python ≤ 3.11
- **Sentence Transformers**: Text embeddings - compatibility issues with 3.12+
- **Various ML Dependencies**: NumPy, SciPy compiled binaries may not be available

### Error Symptoms

You might see errors like:
```bash
ERROR: Could not find a version that satisfies the requirement langchain
ERROR: No matching distribution found for sentence-transformers
ImportError: No module named '_ctypes'
```

## 🛠️ Solution Options

### Option 1: Use pyenv (Recommended)

**Install pyenv:**

**macOS (Homebrew):**
```bash
# Install pyenv
brew install pyenv

# Add to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

**Linux (curl):**
```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

**Install compatible Python:**
```bash
# List available Python versions
pyenv install --list | grep "3.11"

# Install Python 3.11 (latest stable)
pyenv install 3.11.9

# Set as project-specific version
cd RAG-Chat-App
pyenv local 3.11.9

# Verify
python --version  # Should show Python 3.11.9
```

### Option 2: Use Conda/Miniconda

**Install Miniconda:**
```bash
# Download Miniconda installer
# Visit: https://docs.conda.io/en/latest/miniconda.html

# For Linux/macOS:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Follow installation prompts
```

**Create compatible environment:**
```bash
# Create environment with Python 3.11
conda create -n rag-chat python=3.11

# Activate environment
conda activate rag-chat

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Option 3: Use Docker (Isolation)

**Create Dockerfile with compatible Python:**

Create `Dockerfile.python311`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

**Build and run:**
```bash
# Build Docker image
docker build -f Dockerfile.python311 -t rag-chat-backend .

# Run container
docker run -p 8000:8000 rag-chat-backend
```

### Option 4: System-Wide Python Installation

**⚠️ Warning**: This may affect other applications.

**Ubuntu/Debian:**
```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-pip

# Use specific version
python3.11 -m venv venv
source venv/bin/activate
```

**CentOS/RHEL/Fedora:**
```bash
# Install from EPEL or build from source
sudo dnf install python3.11 python3.11-pip python3.11-venv
```

## 🚀 Automated Setup Script

### The fix_python_env.sh Script

The project includes an automated fix script:

```bash
# Make executable
chmod +x fix_python_env.sh

# Run the fixer
./fix_python_env.sh
```

**What the script does:**

1. **Detects your current Python version**
2. **Checks for pyenv availability**
3. **Installs pyenv if needed**
4. **Downloads and installs Python 3.11**
5. **Creates virtual environment with compatible Python**
6. **Installs all required dependencies**

### Manual Script Content

If you want to understand what the script does:

```bash
#!/bin/bash
set -e

echo "🔧 Python 3.14+ Environment Fixer for RAG Chat App"

# Check current Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | grep -o "3\.[0-9][0-9]*" || echo "not found")
echo "Current Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" > "3.11" ]] || [[ "$PYTHON_VERSION" == "not found" ]]; then
    echo "⚠️  Python 3.12+ detected or Python not found. Setting up compatible environment..."
    
    # Check if pyenv is installed
    if ! command -v pyenv &> /dev/null; then
        echo "📦 Installing pyenv..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install pyenv
            else
                echo "Please install Homebrew first: https://brew.sh"
                exit 1
            fi
        else
            # Linux
            curl https://pyenv.run | bash
        fi
        
        # Add to PATH
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi
    
    # Install Python 3.11
    echo "🐍 Installing Python 3.11..."
    pyenv install 3.11.9 || echo "Python 3.11.9 already installed"
    
    # Set local version
    pyenv local 3.11.9
    
    echo "✅ Python 3.11.9 set for this project"
else
    echo "✅ Compatible Python version detected"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate and install dependencies
echo "🔌 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt

echo "🎉 Setup complete! Your environment is ready."
echo "To activate: source venv/bin/activate"
```

## ✅ Verification Steps

### Check Environment Setup

After setup, verify everything works:

```bash
# 1. Check Python version
python --version
# Should show: Python 3.11.x

# 2. Activate environment
source venv/bin/activate  # or conda activate rag-chat

# 3. Test key imports
python -c "import langchain; print('✅ LangChain:', langchain.__version__)"
python -c "import openai; print('✅ OpenAI:', openai.__version__)"
python -c "import sentence_transformers; print('✅ Sentence Transformers: OK')"

# 4. Test backend startup
cd backend
python app.py
# Should start without errors
```

### Run Full Application

```bash
# Start backend
cd backend
source venv/bin/activate  # if using venv
# or: conda activate rag-chat  # if using conda
python app.py

# In another terminal, start frontend
cd frontend
npm install
npm run dev
```

## 🐛 Troubleshooting

### Common Issues After Setup

**Issue: pyenv command not found after installation**

```bash
# Add to your shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

**Issue: Python 3.11 installation fails**

```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
liblzma-dev python3-openssl git

# Then retry
pyenv install 3.11.9
```

**Issue: Virtual environment activation fails**

```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**Issue: Still getting import errors**

```bash
# Clear all caches
pip cache purge
rm -rf ~/.cache/pip

# Reinstall from scratch
pip uninstall -y -r backend/requirements.txt
pip install -r backend/requirements.txt
```

## 📋 Alternative Approaches

### Using Virtual Environments Only

If you can't install alternative Python versions:

```bash
# Try installing with specific constraints
pip install --constraint https://raw.githubusercontent.com/langchain-ai/langchain/master/constraints.txt langchain

# Or use older package versions
pip install langchain==0.0.350  # Example: older compatible version
```

### Using Development/Preview Versions

**⚠️ Experimental - may be unstable**

```bash
# Try installing preview versions that might support Python 3.14
pip install --pre langchain
pip install --pre sentence-transformers
```

### Container-Only Development

If all else fails, develop entirely in Docker:

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.python311
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    command: python app.py
    
  frontend:
    image: node:18
    volumes:
      - ./frontend:/app
    working_dir: /app
    ports:
      - "5173:5173"
    command: bash -c "npm install && npm run dev"
```

Run with:
```bash
docker-compose -f docker-compose.dev.yml up
```

## 🎯 Summary

For Python 3.14+ users, the recommended approach is:

1. **Use pyenv** to install Python 3.11 alongside your system Python
2. **Create project-specific virtual environment** with compatible Python
3. **Use the automated `fix_python_env.sh` script** for convenience
4. **Verify setup** with the provided test commands

This approach maintains system Python while providing compatibility for the RAG Chat application.

---

**🚀 Ready to continue?** Go back to [Quick Start Guide](QUICK_START.md) to launch your application!