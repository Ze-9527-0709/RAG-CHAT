"""
Shared data models for RAG Chat Application
Eliminates duplication between app.py and mock_app.py
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ===== Chat Models =====

class ChatRequest(BaseModel):
    """Base chat request model"""
    session_id: str
    message: str
    max_history: int = 8

class StreamChatRequest(ChatRequest):
    """Streaming chat request model"""
    stream: bool = True

class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str

# ===== Model Management Models =====

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    name: str
    display_name: str
    description: str
    available: bool
    is_local: bool
    max_tokens: int
    cost_per_1k: float
    failure_count: int = 0
    status: Optional[str] = None

class ModelStatusResponse(BaseModel):
    """Current model status response"""
    current_model: Dict[str, Any]
    available_models: List[ModelInfo]

# ===== File Upload Models =====

class UploadResponse(BaseModel):
    """File upload response"""
    message: str
    files_processed: int
    total_size: int
    upload_status: str

class FileInfo(BaseModel):
    """File information"""
    filename: str
    size: int
    upload_time: str
    processed: bool = False

class UploadedFilesResponse(BaseModel):
    """Response for uploaded files list"""
    files: List[FileInfo]
    total_count: int

# ===== Learning System Models =====

class FeedbackRequest(BaseModel):
    """User feedback request"""
    conversation_id: str
    feedback_type: str  # 'positive' or 'negative'
    content: str = ""

class LearningStats(BaseModel):
    """Learning statistics"""
    total_conversations: int
    learned_concepts: int
    total_feedback: int
    top_concepts: List[Dict[str, Any]]
    user_style_analysis: Dict[str, Any]

class GrowthInsights(BaseModel):
    """AI growth insights"""
    recommendations: List[str]
    progress_score: float
    areas_for_improvement: List[str]

# ===== Agent Identity Models =====

class AgentIdentityUpdate(BaseModel):
    """Agent identity update request"""
    name: str
    personality: str

class CoreMemoryAdd(BaseModel):
    """Core memory addition request"""
    memory_type: str
    content: str
    importance: int = 5

# ===== Performance Models =====

class PerformanceStats(BaseModel):
    """Performance statistics"""
    cache_hits: int
    cache_misses: int
    avg_response_time: float
    total_requests: int
    active_sessions: int

# ===== Health Check Model =====

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: Optional[str] = None
    mode: Optional[str] = None
    version: Optional[str] = "1.0.0"