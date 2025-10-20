"""
Minimal backend for UI testing - without RAG dependencies
Run with: uvicorn mock_app:app --reload --port 8000
"""
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# Import shared models
from shared.models import (
    ChatRequest, ChatResponse, StreamChatRequest,
    UploadResponse, UploadedFilesResponse, LearningStats,
    GrowthInsights
)

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

# Models moved to shared/models.py

# Session storage for mock backend
MOCK_SESSIONS = {}

# Mock responses for different sessions
def get_mock_response(session_id: str, message: str) -> str:
    # Initialize session if it doesn't exist
    if session_id not in MOCK_SESSIONS:
        MOCK_SESSIONS[session_id] = []
    
    # Add user message to session
    MOCK_SESSIONS[session_id].append({"role": "user", "content": message})
    
    # Generate session-specific response
    session_number = len([k for k in MOCK_SESSIONS.keys() if k <= session_id]) 
    message_count = len(MOCK_SESSIONS[session_id])
    
    response = f"""# Welcome to RAG Chat Session {session_number}! 🚀

This is message #{message_count} in this session. **Session ID**: `{session_id[:8]}...`

## Session Features Working:

1. **Isolated Sessions** - Each session maintains separate conversation history
2. **Session Switching** - Messages stay in their respective sessions  
3. **Message Counting** - This session has {message_count} messages
4. **Session Persistence** - Your conversation history is maintained

### Your Message:
> "{message}"

### Session Status:
- **Current Session**: {session_id[:8]}...
- **Messages in Session**: {message_count}
- **Total Sessions**: {len(MOCK_SESSIONS)}

```python
def session_demo():
    session_id = "{session_id[:8]}..."
    message_count = {message_count}
    return f"Session {{session_id}} - Message {{message_count}}"
```

### Test Instructions:

1. Create a new session
2. Send messages in different sessions
3. Switch between sessions to verify isolation

> **Session Management**: Working perfectly! Each session maintains its own conversation. ✨

**Status**: Mock mode with session isolation! 🎯
"""
    
    # Add assistant response to session
    MOCK_SESSIONS[session_id].append({"role": "assistant", "content": response})
    
    return response

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Non-streaming mock endpoint"""
    await asyncio.sleep(0.5)  # Simulate processing
    mock_response = get_mock_response(req.session_id, req.message)
    return ChatResponse(
        answer=mock_response
    )

@app.post("/api/chat_stream")
async def chat_stream(req: ChatRequest):
    """Streaming mock endpoint"""
    from fastapi.responses import StreamingResponse
    
    async def generate():
        # Get session-specific response
        mock_response = get_mock_response(req.session_id, req.message)
        
        # Stream response word by word
        words = mock_response.split()
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

# Model selection API
@app.get("/api/models")
async def get_models():
    """Get available models for selection"""
    return {
        "models": [
            {
                "id": "gpt-4o",
                "name": "gpt-4o",
                "display_name": "GPT-4o (Best Quality)",
                "description": "Highest quality responses, best for complex tasks",
                "available": True,
                "auto_available": False,
                "is_local": False,
                "max_tokens": 128000,
                "cost_per_1k": 0.0025,
                "failure_count": 0,
                "status": "ready"
            },
            {
                "id": "gpt-4o-mini",
                "name": "gpt-4o-mini",
                "display_name": "GPT-4o Mini (Balanced)",
                "description": "Great balance of quality and speed, cost-effective",
                "available": True,
                "auto_available": True,
                "is_local": False,
                "max_tokens": 128000,
                "cost_per_1k": 0.00015,
                "failure_count": 0,
                "status": "ready"
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "gpt-3.5-turbo",
                "display_name": "GPT-3.5 Turbo (Fast)",
                "description": "Fast responses, good for simple questions",
                "available": True,
                "auto_available": True,
                "is_local": False,
                "max_tokens": 16385,
                "cost_per_1k": 0.0005,
                "failure_count": 0,
                "status": "ready"
            },
            {
                "id": "local-llama",
                "name": "llama3.1:8b",
                "display_name": "Llama 3.1 8B (Local)",
                "description": "Private local model, no API costs, works offline",
                "available": True,
                "auto_available": True,
                "is_local": True,
                "max_tokens": 8192,
                "cost_per_1k": 0.0,
                "failure_count": 0,
                "status": "ready"
            }
        ]
    }

@app.get("/api/model_status")
async def get_model_status():
    """Get current model status"""
    return {
        "current_model": {
            "model_name": "gpt-4o-mini",
            "tier": "gpt-4o-mini",
            "is_user_selected": False,
            "selection_reason": "Auto-selected based on availability and cost"
        }
    }

@app.post("/api/models/auto")
async def enable_auto_model():
    """Enable auto model selection"""
    return {
        "success": True,
        "message": "Auto model selection enabled",
        "current_model": "gpt-4o-mini"
    }

@app.post("/api/models/{model_id}")
async def select_model(model_id: str):
    """Select a specific model"""
    return {
        "success": True,
        "message": f"Model {model_id} selected",
        "current_model": model_id
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Mock backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
