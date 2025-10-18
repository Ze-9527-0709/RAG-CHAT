# 🐍 Python Environment Setup Guide

**Language**: [English](PYTHON_SETUP.md) | [中文](中文指南/PYTHON_SETUP.md)

---

> **Complete Python Environment Configuration for RAG Chat App**

This guide provides detailed instructions for setting up Python environments compatible with the RAG Chat application, with special attention to LangChain dependency requirements.

## 🚨 Critical Version Requirements

### Supported Python Versions
- ✅ **Python 3.8** - Fully supported
- ✅ **Python 3.9** - Fully supported  
- ✅ **Python 3.10** - Recommended
- ✅ **Python 3.11** - Recommended
- ❌ **Python 3.12** - Not supported (LangChain compatibility issues)
- ❌ **Python 3.13** - Not supported
- ❌ **Python 3.14** - Not supported

> **⚠️ Important**: LangChain and related AI libraries have strict version requirements. Using unsupported versions will cause installation failures.

## 🔍 Check Current Python Version

```bash
# Check Python version
python --version
python3 --version

# Check if pip is available
pip --version
pip3 --version
```

## 🛠️ Installation Methods

### Method 1: Official Python Installer (Recommended for Beginners)

1. **Visit Official Website**
   - Go to: https://www.python.org/downloads/
   - Download Python 3.10 or 3.11 (latest stable)

2. **Installation Steps**
   ```bash
   # macOS: Download .pkg installer
   # Windows: Download .exe installer  
   # Linux: Use package manager or compile from source
   ```

3. **Verify Installation**
   ```bash
   python3 --version
   # Should show: Python 3.10.x or Python 3.11.x
   ```

### Method 2: Package Managers

#### macOS (Homebrew)
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Set as default (optional)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-pip

# Install additional tools
sudo apt install python3.11-dev python3.11-distutils
```

#### CentOS/RHEL/Fedora
```bash
# Install Python 3.11 (Fedora)
sudo dnf install python3.11 python3.11-pip python3.11-venv

# For CentOS/RHEL, enable EPEL repository first
sudo yum install epel-release
sudo yum install python311 python311-pip
```

### Method 3: Python Version Manager (pyenv) - Advanced Users

pyenv allows you to install and manage multiple Python versions.

#### Install pyenv

**macOS:**
```bash
# Install using Homebrew
brew install pyenv

# Add to shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

**Linux:**
```bash
# Install using curl
curl https://pyenv.run | bash

# Add to shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

#### Use pyenv to Install Python

```bash
# List available Python versions
pyenv install --list | grep 3.11

# Install Python 3.11.x (latest)
pyenv install 3.11.5

# Set as global default
pyenv global 3.11.5

# Verify
python --version
```

## 🏠 Virtual Environment Setup

Virtual environments isolate project dependencies and prevent conflicts.

### Method 1: venv (Built-in, Recommended)

```bash
# Navigate to project directory
cd RAG-Chat-App

# Create virtual environment
python3 -m venv venv

# Activate environment
# macOS/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

# Verify activation (should show venv path)
which python
python --version

# Install project dependencies
pip install -r backend/requirements.txt
```

### Method 2: conda (If using Anaconda/Miniconda)

```bash
# Create conda environment with Python 3.11
conda create -n rag-chat python=3.11

# Activate environment
conda activate rag-chat

# Install pip packages
pip install -r backend/requirements.txt

# Or install via conda when available
conda install numpy pandas
pip install -r backend/requirements.txt
```

### Method 3: virtualenv (Third-party)

```bash
# Install virtualenv
pip install virtualenv

# Create environment
virtualenv -p python3.11 venv

# Activate
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r backend/requirements.txt
```

## 🐍 Conda Environment Setup (Advanced)

### Install Miniconda

1. **Download Miniconda**
   - Visit: https://docs.conda.io/en/latest/miniconda.html
   - Choose installer for your OS

2. **Install and Setup**
   ```bash
   # macOS/Linux
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # Follow prompts, restart terminal
   conda --version
   ```

### Create Project Environment

```bash
# Create environment with specific Python version
conda create -n rag-chat python=3.11 pip

# Activate environment
conda activate rag-chat

# Install core packages via conda (faster)
conda install numpy pandas scipy

# Install AI packages via pip
pip install -r backend/requirements.txt

# List installed packages
conda list
pip list
```

### Environment Management

```bash
# List all environments
conda env list

# Remove environment
conda env remove -n rag-chat

# Export environment
conda env export > environment.yml

# Create from exported file
conda env create -f environment.yml
```

## 🔧 Dependency Installation & Troubleshooting

### Core Dependencies

The RAG Chat app requires these key packages:

```bash
# AI and ML packages
pip install openai langchain langchain-huggingface
pip install pinecone-client sentence-transformers

# Web framework
pip install fastapi uvicorn

# Data processing
pip install pandas numpy

# File handling
pip install python-multipart
```

### Common Installation Issues

#### Issue 1: LangChain Compatibility

**Problem**: `ERROR: Cannot install langchain with Python 3.12+`

**Solution**:
```bash
# Check Python version
python --version

# If 3.12+, install compatible Python version
pyenv install 3.11.5
pyenv local 3.11.5

# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

#### Issue 2: Compilation Errors

**Problem**: `error: Microsoft Visual C++ 14.0 is required` (Windows)

**Solutions**:
```bash
# Option 1: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Option 2: Use pre-compiled wheels
pip install --only-binary=all -r backend/requirements.txt

# Option 3: Use conda for problematic packages
conda install numpy scipy pandas
pip install -r backend/requirements.txt
```

#### Issue 3: Permission Errors

**Problem**: `Permission denied` during installation

**Solutions**:
```bash
# Use user installation (not recommended in venv)
pip install --user package_name

# Fix permissions (macOS/Linux)
sudo chown -R $(whoami) ~/.local

# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

#### Issue 4: Network/SSL Errors

**Problem**: SSL certificate verification errors

**Solutions**:
```bash
# Upgrade pip and certificates
pip install --upgrade pip

# Temporary workaround (not recommended for production)
pip install --trusted-host pypi.org --trusted-host pypi.python.org package_name

# Use company proxy settings
pip install --proxy http://user:pass@proxy.company.com:port package_name
```

### Verification Commands

```bash
# Test Python installation
python -c "import sys; print(sys.version)"

# Test key dependencies
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import langchain; print('LangChain:', langchain.__version__)"
python -c "import pinecone; print('Pinecone: OK')"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"

# Test backend startup
cd backend
python app.py
# Should start without errors
```

## 🔄 Environment Switching & Management

### Activate/Deactivate Environments

```bash
# Activate venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Deactivate any environment
deactivate

# Activate conda environment
conda activate rag-chat

# Deactivate conda environment
conda deactivate
```

### Multiple Project Management

```bash
# Project 1: RAG Chat
cd ~/projects/rag-chat
source venv/bin/activate

# Project 2: Other AI project
cd ~/projects/other-ai
source other-venv/bin/activate

# Using conda for multiple projects
conda create -n project1 python=3.11
conda create -n project2 python=3.10
```

## 🚀 Quick Setup Scripts

### Automated Environment Setup

Create `setup_python.sh`:

```bash
#!/bin/bash
set -e

echo "🐍 Setting up Python environment for RAG Chat App..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -o "3\.[0-9][0-9]*" | head -1)
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ] && [ "$MINOR" -le 11 ]; then
    echo "✅ Python $PYTHON_VERSION is compatible"
else
    echo "❌ Python $PYTHON_VERSION is not compatible"
    echo "Please install Python 3.8-3.11"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate environment
echo "🔌 Activating environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r backend/requirements.txt

echo "🎉 Setup complete! Activate with: source venv/bin/activate"
```

Make executable and run:
```bash
chmod +x setup_python.sh
./setup_python.sh
```

### Environment Health Check

Create `check_env.py`:

```python
#!/usr/bin/env python3
import sys
import importlib

def check_python_version():
    version = sys.version_info
    if 3.8 <= version.major == 3 <= version.minor <= 3.11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Not compatible")
        return False

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

if __name__ == "__main__":
    print("🔍 RAG Chat App Environment Check\n")
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Check required packages
    packages = [
        ('OpenAI', 'openai'),
        ('LangChain', 'langchain'),
        ('FastAPI', 'fastapi'),
        ('Uvicorn', 'uvicorn'),
        ('Pinecone', 'pinecone'),
        ('Sentence Transformers', 'sentence_transformers'),
        ('HuggingFace Transformers', 'transformers'),
    ]
    
    all_packages_ok = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_packages_ok = False
    
    print()
    if python_ok and all_packages_ok:
        print("🎉 Environment is ready for RAG Chat App!")
    else:
        print("⚠️  Environment needs attention. Please install missing components.")
        if not python_ok:
            print("   - Install compatible Python version (3.8-3.11)")
        if not all_packages_ok:
            print("   - Install missing packages: pip install -r backend/requirements.txt")
```

Run the check:
```bash
python check_env.py
```

## 🆘 Emergency Fixes

### Complete Environment Reset

If everything is broken:

```bash
# Remove existing environment
rm -rf venv

# Clear pip cache
pip cache purge

# Reinstall from scratch
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Alternative Python Installation (Linux)

If system Python is problematic:

```bash
# Install from deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip

# Use specific Python version
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

## 📋 Best Practices

### Development Workflow

1. **Always use virtual environments**
2. **Pin dependency versions** in requirements.txt
3. **Document environment setup** for team members
4. **Regular dependency updates** with testing
5. **Environment consistency** across development/production

### Requirements Management

```bash
# Generate current requirements
pip freeze > requirements.txt

# Install exact versions
pip install -r requirements.txt

# Update specific package
pip install --upgrade package_name
pip freeze > requirements.txt

# Check for outdated packages
pip list --outdated
```

### Security Considerations

```bash
# Check for security vulnerabilities
pip install safety
safety check

# Update packages with security fixes
pip install --upgrade pip setuptools wheel
pip install --upgrade -r requirements.txt
```

---

**🎯 Ready to start?** Return to [Quick Start Guide](QUICK_START.md) to launch your RAG Chat App!