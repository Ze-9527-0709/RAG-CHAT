import os
import asyncio
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---- 智能模型降级系统 ----
from model_fallback import ModelFallbackManager, ModelTier, MODEL_CONFIGS, ModelConfig, estimate_tokens

# ---- 环境变量 ----
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Initialize OpenAI client with graceful handling of missing API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    print("✅ OpenAI client initialized successfully")
else:
    client = None
    print("⚠️ No OpenAI API key found - OpenAI models will be unavailable")

# Initialize model fallback manager
model_manager = ModelFallbackManager(client)

# ---- FastAPI 应用程序初始化 ----
app = FastAPI(title="RAG Chat API", version="1.0.0")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 数据模型定义 ----
class ChatMessage(BaseModel):
    session_id: str
    message: str
    model: str = "auto"
    use_rag: bool = True

class ModelInfo(BaseModel):
    id: str
    name: str
    display_name: str
    description: str
    available: bool
    auto_available: bool = False
    is_local: bool = False
    max_tokens: int = 4000
    cost_per_1k: float = 0.0
    failure_count: int = 0
    status: str = "ready"

# ---- 全局变量 ----
sessions: Dict[str, List[Dict]] = {}
rag_enabled = False

# ---- 辅助函数 ----
def initialize_session(session_id: str):
    """Initialize new chat session"""
    if session_id not in sessions:
        sessions[session_id] = []
        print(f"🆕 Created new session: {session_id}")

def add_to_session(session_id: str, role: str, content: str):
    """Add message to session"""
    sessions[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

def get_session_messages(session_id: str):
    """Get session history messages"""
    return sessions.get(session_id, [])[-limit:]

# ---- API 端点 ----
@app.get("/")
async def root():
    return {"message": "RAG Chat API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/models")
async def get_available_models():
    """Get available model list"""
    try:
        models = []
        
        # 添加 GPT 模型
        for tier in [ModelTier.GPT4, ModelTier.GPT4_MINI, ModelTier.GPT35]:
            config = MODEL_CONFIGS[tier]
            is_available = await model_manager._check_openai_availability(tier, 100)
            failure_count = model_manager.failure_counts.get(tier, 0)
            
            # 根据失败次数确定状态
            if failure_count >= model_manager.max_retries:
                status = "quota_exceeded"
            elif is_available:
                status = "ready"
            else:
                status = "unavailable"
            
            models.append(ModelInfo(
                id=tier.value,
                name=config.name,
                display_name=f"{config.name.upper().replace('-', ' ')} ({'Best Quality' if tier == ModelTier.GPT4 else 'Balanced' if tier == ModelTier.GPT4_MINI else 'Fast'})",
                description="Highest quality responses, best for complex tasks" if tier == ModelTier.GPT4 else 
                           "Great balance of quality and speed, cost-effective" if tier == ModelTier.GPT4_MINI else 
                           "Fast responses, good for simple questions",
                available=is_available,
                auto_available=tier in [ModelTier.GPT35, ModelTier.LOCAL],
                is_local=False,
                max_tokens=config.max_tokens,
                cost_per_1k=config.cost_per_1k_tokens,
                failure_count=failure_count,
                status=status
            ))
        
        # 添加本地模型
        local_available = await model_manager._check_local_model()
        local_config = MODEL_CONFIGS[ModelTier.LOCAL]
        
        models.append(ModelInfo(
            id=ModelTier.LOCAL.value,
            name=local_config.name if hasattr(local_config, 'name') else "llama3.1:8b",
            display_name="Llama 3.1 8B (Local)",
            description="Private local model, no API costs, works offline",
            available=local_available,
            auto_available=True,
            is_local=True,
            max_tokens=8192,
            cost_per_1k=0.0,
            failure_count=0,
            status="ready" if local_available else "unavailable"
        ))
        
        return {"models": models}
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return {"models": [], "error": str(e)}

@app.get("/api/model_status")
async def get_model_status():
    """Get current selected model status"""
    try:
        current_model = model_manager.get_current_model()
        if current_model:
            return {
                "current_model": current_model.value,
                "tier": current_model.value,
                "model_name": MODEL_CONFIGS[current_model].name,
                "is_user_selected": hasattr(model_manager, 'user_preferred_model') and model_manager.user_preferred_model == current_model,
                "failure_count": model_manager.failure_counts.get(current_model, 0)
            }
        else:
            return {"current_model": None, "message": "No model available"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/models/select/{model_id}")
async def select_model(request: ModelSelection):
    """Manual model selection"""
    try:
        # 将model_id转换为ModelTier
        tier_map = {
            "gpt-4o": ModelTier.GPT4,
            "gpt-4o-mini": ModelTier.GPT4_MINI,
            "gpt-3.5-turbo": ModelTier.GPT35,
            "local-llama": ModelTier.LOCAL
        }
        
        if model_id not in tier_map:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
        
        selected_tier = tier_map[model_id]
        model_manager.user_preferred_model = selected_tier
        
        return {
            "success": True,
            "selected_model": model_id,
            "tier": selected_tier.value,
            "message": f"Selected model: {model_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/auto")
async def enable_auto_selection():
    """Enable automatic model selection"""
    try:
        model_manager.user_preferred_model = None
        return {
            "success": True,
            "mode": "auto",
            "message": "Auto model selection enabled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """Handle chat request"""
    try:
        session_id = message.session_id
        user_message = message.message
        
        # 初始化会话
        initialize_session(session_id)
        
        # 添加用户消息到会话
        add_to_session(session_id, "user", user_message)
        
        # 获取会话历史
        history = get_session_messages(session_id)
        
        # 使用模型管理器获取响应
        response = await model_manager.get_chat_response(
            user_message,
            history=history,
            estimated_tokens=estimate_tokens(user_message)
        )
        
        # 添加助手响应到会话
        add_to_session(session_id, "assistant", response["content"])
        
        return {
            "response": response["content"],
            "model_used": response.get("model_used", "unknown"),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get session history"""
    return {
        "session_id": session_id,
        "messages": sessions.get(session_id, [])
    }

@app.delete("/api/sessions/{session_id}")
async def clear_session_history(session_id: str):
    """Clear session history"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": f"Session {session_id} cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)