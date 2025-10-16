# Backend Files Comparison

## 📊 app.py vs mock_app.py 功能对比

| 功能特性 | app.py (完整RAG) | mock_app.py (轻量测试) |
|---------|-----------------|----------------------|
| **用途** | 生产环境完整RAG系统 | 开发测试UI功能 |
| **依赖** | 需要OpenAI + Pinecone | 只需FastAPI |
| **API端点** | `/chat`, `/chat_stream` | `/api/chat`, `/api/chat_stream`, `/api/upload` |
| **文件上传** | ❌ 缺少 | ✅ 支持 |
| **RAG功能** | ✅ 完整RAG检索 | ❌ 模拟响应 |
| **向量数据库** | ✅ Pinecone集成 | ❌ 无 |
| **环境要求** | 需要API密钥配置 | 无需配置 |
| **启动速度** | 慢（需要下载模型） | 快（无依赖） |
| **响应内容** | 真实AI回答 | 固定模拟内容 |

## 🎯 建议方案

### 方案A：两个都保留（推荐） ⭐

**保留理由：**
- `mock_app.py` - 用于快速开发和UI测试
- `app.py` - 用于完整RAG功能

**操作：**
1. 给 `app.py` 添加文件上传功能
2. 更新启动脚本选择模式

### 方案B：只保留一个

**如果选择只保留 `mock_app.py`：**
- ✅ 快速启动，适合演示
- ❌ 没有真实RAG功能

**如果选择只保留 `app.py`：**
- ✅ 完整功能
- ❌ 需要配置API密钥，启动慢
- ❌ 缺少文件上传功能

## 🛠️ 推荐行动方案

### 1. 完善 app.py（添加文件上传）
```python
# 在 app.py 中添加：
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    # 1. 保存文件
    # 2. 自动分块处理
    # 3. 生成向量嵌入
    # 4. 存入Pinecone
    return {"status": "success", "processed": True}
```

### 2. 改进启动脚本
```bash
# 在 start.sh 中添加模式选择：
if [ "$1" = "full" ]; then
    # 启动完整RAG版本
    ./venv/bin/uvicorn app:app --port 8000
else
    # 默认启动轻量版本
    ./venv/bin/uvicorn mock_app:app --port 8000
fi
```

### 3. 使用场景
- **开发/演示**: `./start.sh` (默认mock模式)
- **生产/完整功能**: `./start.sh full`

## 💡 最终建议

**保留两个文件**，它们各有用途：

1. **日常开发**: 使用 `mock_app.py` 快速测试UI
2. **完整功能**: 使用 `app.py` 体验真实RAG
3. **文件管理**: 两个文件都很小，不占空间
4. **灵活性**: 根据需要选择不同模式

这样既保证了开发效率，又不失去完整功能！