"""
Base API classes for RAG Chat Application
Eliminates duplication between main and mock APIs
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
from datetime import datetime
from .models import (
    ChatRequest, ChatResponse, StreamChatRequest,
    HealthResponse, UploadResponse, UploadedFilesResponse,
    ModelStatusResponse, LearningStats, GrowthInsights,
    FeedbackRequest, PerformanceStats
)

class BaseAPI:
    """Base API class with common endpoints"""
    
    def __init__(self, app: FastAPI, mode: str = "production"):
        self.app = app
        self.mode = mode
        self.setup_common_routes()
    
    def setup_common_routes(self):
        """Setup routes that are common to both real and mock APIs"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status="ok",
                timestamp=datetime.now().isoformat(),
                mode=self.mode
            )
    
    # Abstract methods to be implemented by subclasses
    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle regular chat request"""
        raise NotImplementedError("Subclasses must implement handle_chat")
    
    async def handle_chat_stream(self, request: StreamChatRequest) -> StreamingResponse:
        """Handle streaming chat request"""
        raise NotImplementedError("Subclasses must implement handle_chat_stream")
    
    async def handle_upload(self, files: List[UploadFile]) -> UploadResponse:
        """Handle file upload"""
        raise NotImplementedError("Subclasses must implement handle_upload")
    
    async def get_uploaded_files(self) -> UploadedFilesResponse:
        """Get list of uploaded files"""
        raise NotImplementedError("Subclasses must implement get_uploaded_files")


class ModelManagementMixin:
    """Mixin for model management functionality"""
    
    def setup_model_routes(self):
        """Setup model management routes"""
        
        @self.app.get("/api/models")
        async def get_available_models():
            return await self.get_models()
        
        @self.app.get("/api/model_status")
        async def get_model_status():
            return await self.get_current_model_status()
        
        @self.app.post("/api/models/{model_id}")
        async def select_model(model_id: str):
            return await self.select_model_by_id(model_id)
        
        @self.app.post("/api/models/auto")
        async def enable_auto_model():
            return await self.enable_auto_selection()
    
    # Abstract methods for model management
    async def get_models(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement get_models")
    
    async def get_current_model_status(self) -> ModelStatusResponse:
        raise NotImplementedError("Subclasses must implement get_current_model_status")
    
    async def select_model_by_id(self, model_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement select_model_by_id")
    
    async def enable_auto_selection(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement enable_auto_selection")


class LearningSystemMixin:
    """Mixin for AI learning system functionality"""
    
    def setup_learning_routes(self):
        """Setup learning system routes"""
        
        @self.app.post("/api/feedback")
        async def add_feedback(feedback: FeedbackRequest):
            return await self.process_feedback(feedback)
        
        @self.app.get("/api/learning_stats/{session_id}")
        async def get_learning_stats(session_id: str):
            return await self.get_session_learning_stats(session_id)
        
        @self.app.get("/api/growth_insights/{session_id}")
        async def get_growth_insights(session_id: str):
            return await self.get_session_growth_insights(session_id)
    
    # Abstract methods for learning system
    async def process_feedback(self, feedback: FeedbackRequest) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process_feedback")
    
    async def get_session_learning_stats(self, session_id: str) -> LearningStats:
        raise NotImplementedError("Subclasses must implement get_session_learning_stats")
    
    async def get_session_growth_insights(self, session_id: str) -> GrowthInsights:
        raise NotImplementedError("Subclasses must implement get_session_growth_insights")


class ChatAPIMixin:
    """Mixin for chat functionality"""
    
    def setup_chat_routes(self):
        """Setup chat-related routes"""
        
        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat(req: ChatRequest):
            return await self.handle_chat(req)
        
        @self.app.post("/api/chat_stream")
        async def chat_stream(req: StreamChatRequest):
            return await self.handle_chat_stream(req)
        
        @self.app.post("/api/chat_with_image")
        async def chat_with_image(
            session_id: str,
            message: str = "",
            max_history: int = 8,
            image: UploadFile = File(...)
        ):
            return await self.handle_chat_with_image(session_id, message, max_history, image)
        
        @self.app.post("/api/chat_stream_with_image")
        async def chat_stream_with_image(
            session_id: str,
            message: str = "",
            max_history: int = 8,
            stream: str = "true",
            image: UploadFile = File(...)
        ):
            return await self.handle_chat_stream_with_image(session_id, message, max_history, stream, image)


class FileManagementMixin:
    """Mixin for file management functionality"""
    
    def setup_file_routes(self):
        """Setup file management routes"""
        
        @self.app.post("/api/upload", response_model=UploadResponse)
        async def upload_files(files: List[UploadFile] = File(...)):
            return await self.handle_upload(files)
        
        @self.app.get("/api/uploaded_files", response_model=UploadedFilesResponse)
        async def list_uploaded_files():
            return await self.get_uploaded_files()
        
        @self.app.post("/api/process_existing_files")
        async def process_existing_files():
            return await self.process_all_files()
    
    # Abstract method for file processing
    async def process_all_files(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process_all_files")


class FullAPI(BaseAPI, ModelManagementMixin, LearningSystemMixin, ChatAPIMixin, FileManagementMixin):
    """Complete API class combining all mixins"""
    
    def __init__(self, app: FastAPI, mode: str = "production"):
        super().__init__(app, mode)
        # Setup all route mixins
        self.setup_model_routes()
        self.setup_learning_routes()
        self.setup_chat_routes()
        self.setup_file_routes()