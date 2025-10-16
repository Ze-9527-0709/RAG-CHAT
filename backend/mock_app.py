"""
Minimal backend for UI testing - without RAG dependencies
Run with: uvicorn mock_app:app --reload --port 8000
"""
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    max_history: int = 8
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str

# Mock data
MOCK_RESPONSE = """# Welcome to RAG Chat! 🚀

This is a **mock response** to demonstrate the beautiful new UI while dependencies are installing.

## Features Working:

1. **Markdown Rendering** with syntax highlighting
2. **Streaming support** (real-time token display)
3. **Session management** (create, rename, delete)
4. **Clean interface** without reference clutter

### Code Example:

```python
def hello_world():
    print("The UI looks amazing!")
    return "Copilot-style interface ✨"
```

### Key Points:

- Deep dark theme (#0d1117)
- Gradient buttons and accents
- Smooth animations
- Responsive layout

> Once backend dependencies finish installing, you'll get **real RAG responses** with Pinecone + OpenAI!

**Status**: Mock mode - full RAG coming soon! 🎯
"""

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Non-streaming mock endpoint"""
    await asyncio.sleep(0.5)  # Simulate processing
    return ChatResponse(
        answer=MOCK_RESPONSE
    )

@app.post("/api/chat_stream")
async def chat_stream(req: ChatRequest):
    """Streaming mock endpoint"""
    from fastapi.responses import StreamingResponse
    
    async def generate():
        # Stream response word by word
        words = MOCK_RESPONSE.split()
        for word in words:
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield f"data: {word} \n\n"
        
        yield "event: done\ndata: \n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "mock"}

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handle file uploads for RAG learning
    Accepts: .txt, .pdf, .doc, .docx, .md files
    """
    uploaded_filenames = []
    
    for file in files:
        # Save file to upload directory
        file_path = UPLOAD_DIR / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        uploaded_filenames.append(file.filename)
        print(f"📄 Uploaded: {file.filename} ({len(content)} bytes)")
    
    return {
        "status": "success",
        "files_count": len(uploaded_filenames),
        "filenames": uploaded_filenames,
        "message": f"Successfully uploaded {len(uploaded_filenames)} file(s). Files are stored in {UPLOAD_DIR.absolute()}"
    }

@app.get("/api/learning_stats/{session_id}")
async def get_learning_stats(session_id: str):
    """Mock learning statistics for UI testing"""
    return {
        "total_conversations": 15,
        "learned_concepts": 8,
        "total_feedback": 3,
        "top_concepts": [
            {
                "concept": "UI Design and User Experience",
                "usage": 5,
                "confidence": 0.9
            },
            {
                "concept": "React Components and State Management", 
                "usage": 3,
                "confidence": 0.8
            }
        ],
        "user_style_analysis": {
            "formality": 0.6,
            "detail_level": 0.7,
            "friendliness": 0.8,
            "user_conversations_count": 12,
            "analysis": {
                "communication_style": "Professional",
                "detail_preference": "Detailed",
                "friendliness_level": "Friendly"
            }
        }
    }

@app.get("/api/uploaded_files")
async def list_uploaded_files():
    """Mock uploaded files list for UI testing"""
    files = []
    if UPLOAD_DIR.exists():
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "upload_time": "2025-10-15T22:00:00.000000",
                    "file_type": file_path.suffix
                })
    
    return {
        "files": files,
        "count": len(files),
        "message": f"Found {len(files)} uploaded files"
    }

@app.post("/api/chat_with_image")
async def chat_with_image(
    session_id: str = Form(...),
    message: str = Form(""),
    max_history: int = Form(8),
    image: UploadFile = File(...)
):
    """Mock chat endpoint with image support"""
    
    # Save the uploaded image
    image_path = UPLOAD_DIR / f"chat_{image.filename}"
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    # Mock response that acknowledges the image
    mock_response = f"""# Image Analysis Result 🖼️

I can see you've uploaded an image: **{image.filename}**

## What I observe:
- Image size: {len(content)} bytes
- File type: {image.content_type}
- This is a **mock response** for testing the image upload functionality

## Your message:
{message if message else "*No text message provided*"}

### Mock Analysis:
```
📷 Image successfully received and processed
🤖 AI vision capabilities working in mock mode
✨ Ready for real image analysis implementation
```

*Note: This is a demonstration response. In production, this would contain actual image analysis results.*"""
    
    return {"answer": mock_response}

@app.post("/api/chat_stream_with_image")
async def chat_stream_with_image(
    session_id: str = Form(...),
    message: str = Form(""),
    max_history: int = Form(8),
    stream: str = Form("true"),
    image: UploadFile = File(...)
):
    """Mock streaming chat endpoint with image support"""
    
    # Save the uploaded image
    image_path = UPLOAD_DIR / f"chat_stream_{image.filename}"
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    # Mock streaming response
    def generate_stream():
        yield f"event: model_info\ndata: {{\"model\": \"gpt-4-vision\", \"tier\": \"vision\", \"reason\": \"Image analysis mode\"}}\n\n"
        
        response_parts = [
            "# Image Analysis with Streaming 🖼️\n\n",
            f"I can see your image: **{image.filename}**\n\n",
            "## Processing image...\n",
            "- ✅ Image received successfully\n",
            "- 📊 Analyzing visual content...\n",
            f"- 📏 Image size: {len(content)} bytes\n",
            f"- 🏷️ Content type: {image.content_type}\n\n",
            "## Your message:\n",
            f"{message if message else '*No text message provided*'}\n\n",
            "### Mock Vision Analysis:\n",
            "```\n",
            "🎯 Object detection: Active\n",
            "🎨 Color analysis: Complete\n", 
            "📝 Text recognition: Ready\n",
            "🧠 Scene understanding: Enabled\n",
            "```\n\n",
            "*This is a mock streaming response for testing image upload functionality.*"
        ]
        
        for part in response_parts:
            yield f"data: {part}"
            # Small delay to simulate streaming
            import time
            time.sleep(0.1)
        
        yield "event: done\ndata: \n\n"
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
