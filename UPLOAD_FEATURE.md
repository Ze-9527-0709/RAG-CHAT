# 📤 文件上传功能说明

## 🎯 功能概述

现在 RAG Chat App 支持上传文档，让 AI 助手学习您的文件内容！

## 📍 位置

文件上传按钮位于**聊天界面右上角**，显示为 `📤 Upload Files`

## 📝 使用方法

1. **点击上传按钮**：位于聊天页面右上角
2. **选择文件**：可以同时选择多个文件
3. **等待上传**：按钮会显示 "Uploading..." 状态
4. **查看确认**：上传成功后会显示绿色勾号和文件列表

## 📄 支持的文件格式

- `.txt` - 纯文本文件
- `.pdf` - PDF 文档
- `.doc` / `.docx` - Word 文档
- `.md` - Markdown 文件

## 💾 文件存储

上传的文件会保存在：
```
backend/uploads/
```

## ⚙️ 技术细节

### 前端功能
- **位置**：`frontend/src/App.tsx` 中的 `handleFileUpload` 函数
- **UI 组件**：
  - 上传按钮（绿色渐变）
  - 隐藏的文件输入
  - 上传状态提示
- **体验优化**：
  - 上传进度提示
  - 成功/失败消息
  - 自动消息确认

### 后端 API
- **端点**：`POST /api/upload`
- **位置**：`backend/mock_app.py` 中的 `upload_files` 函数
- **功能**：
  - 接收多文件上传
  - 保存到 uploads 目录
  - 返回上传结果

## 🔄 与 RAG 集成

### 当前模式（Mock）
在模拟模式下，文件会被保存但不会立即被 RAG 系统处理。

### 完整 RAG 模式
切换到完整的 RAG 后端（`app.py`）后，上传的文件会：

1. **自动分块**：将文档切分成合适的片段
2. **生成嵌入**：使用 HuggingFace embeddings
3. **存入向量库**：保存到 Pinecone
4. **即时可用**：下次对话立即可以引用

## 🎨 UI 样式

上传按钮使用绿色渐变设计，与整体浅色主题协调：

```css
.btn-upload {
  background: linear-gradient(135deg, #1a7f37 0%, #2ea043 100%);
  color: white;
  /* 悬停时有阴影和位移动画 */
}
```

## 📊 上传后的效果

上传成功后，会在当前会话中看到系统消息：

```
📄 Files uploaded successfully!

- document1.pdf
- guide.txt
- notes.md

These documents are now available for the RAG assistant to reference.
```

## 🔧 配置选项

### 文件大小限制
默认无限制，可以在后端添加：

```python
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    for file in files:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(400, "File too large")
```

### 文件类型验证
已在前端通过 `accept` 属性限制：

```tsx
accept=".txt,.pdf,.doc,.docx,.md"
```

## 🚀 未来增强

计划添加的功能：
- [ ] 上传进度条
- [ ] 文件预览
- [ ] 已上传文件列表
- [ ] 删除已上传文件
- [ ] 文件内容搜索
- [ ] 批量上传管理

## 🐛 故障排除

**问题：上传按钮点击无反应**
- 检查后端是否运行在端口 8000
- 查看浏览器控制台是否有错误

**问题：上传失败**
- 检查 `backend/uploads/` 目录是否有写入权限
- 查看 `backend.log` 日志文件

**问题：文件未被 RAG 使用**
- 确认是否使用完整 RAG 后端（非 mock 模式）
- 检查 Pinecone 配置是否正确
