# 🚀 RAG Chat App - 5分钟快速开始指南

> **新手友好** | 无需复杂配置 | 一键启动

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
- 下载：https://nodejs.org/
- 选择LTS版本 (推荐 18.x+)
- 安装后验证：`node -v`

**🐍 Python** (后端语言)
- 下载：https://python.org/
- 选择 3.8+ 版本
- 安装后验证：`python3 -v`

### 3️⃣ 获取API密钥 (免费)

**OpenAI API Key** (必需)
1. 访问：https://platform.openai.com/
2. 注册/登录账号
3. 点击 "API Keys" -> "Create new secret key"
4. 复制并保存密钥

**Pinecone API Key** (可选，用于文档搜索)
1. 访问：https://app.pinecone.io/
2. 注册/登录账号
3. 创建免费项目
4. 复制API密钥

## ⚡ 一键安装 (2分钟)

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

## 🔧 配置API密钥 (1分钟)

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

## 🚀 启动应用 (30秒)

```bash
# 一键启动所有服务
./start.sh
```

等待启动完成，然后访问：**http://localhost:5173**

## 🎉 开始使用

1. **💬 基础聊天**：直接输入问题，按回车
2. **📁 上传文件**：点击📎按钮，上传PDF/图片
3. **🤖 切换模型**：右上角选择不同AI模型

## 🛑 停止应用

```bash
./stop.sh
```

## ❓ 遇到问题？

### 常见问题快速修复

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
# 清理缓存重新安装
npm cache clean --force
pip3 cache purge
./setup.sh
```

### 🆘 获得帮助

- 📖 **详细文档**：查看 [README.md](README.md)
- 🐛 **问题反馈**：https://github.com/Ze-9527-0709/RAG-CHAT/issues
- 💬 **社区讨论**：https://github.com/Ze-9527-0709/RAG-CHAT/discussions

---

**🌟 小贴士**: 第一次启动可能需要几分钟下载依赖，请耐心等待！