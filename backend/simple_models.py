"""
Simple Model Manager - Direct user model selection without complex fallback logic
"""
import os
import json
import requests
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from openai import OpenAI

class ModelType(Enum):
    """Available model types"""
    GPT4 = "gpt-4o"
    GPT4_MINI = "gpt-4o-mini"
    GPT35 = "gpt-3.5-turbo"
    LOCAL_LLAMA = "llama3.1:8b"

@dataclass
class SimpleModelConfig:
    """Simple model configuration"""
    name: str
    display_name: str
    description: str
    is_local: bool
    requires_api: bool
    max_tokens: int
    cost_per_1k: float = 0.0
    endpoint: Optional[str] = None

# Simple model configurations
MODEL_CONFIGS = {
    ModelType.GPT4: SimpleModelConfig(
        name="gpt-4o",
        display_name="GPT-4o",
        description="Most capable OpenAI model - best for complex tasks",
        is_local=False,
        requires_api=True,
        max_tokens=128000,
        cost_per_1k=0.0025
    ),
    ModelType.GPT4_MINI: SimpleModelConfig(
        name="gpt-4o-mini",
        display_name="GPT-4o Mini",
        description="Fast and efficient OpenAI model - good balance of speed and quality",
        is_local=False,
        requires_api=True,
        max_tokens=128000,
        cost_per_1k=0.00015
    ),
    ModelType.GPT35: SimpleModelConfig(
        name="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        description="Fast OpenAI model - good for simple tasks",
        is_local=False,
        requires_api=True,
        max_tokens=16385,
        cost_per_1k=0.0005
    ),
    ModelType.LOCAL_LLAMA: SimpleModelConfig(
        name="llama3.1:8b",
        display_name="Local Llama 3.1 8B",
        description="Local model via Ollama - free and private",
        is_local=True,
        requires_api=False,
        max_tokens=8192,
        cost_per_1k=0.0,
        endpoint="http://localhost:11434/api/generate"
    )
}

class SimpleModelManager:
    """Simple model manager with direct user control"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.client = openai_client
        self.selected_model = ModelType.LOCAL_LLAMA  # Default to local model
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models for user selection"""
        models = []
        
        for model_type, config in MODEL_CONFIGS.items():
            # Check if model is actually available
            is_available = True
            status = "ready"
            
            if config.requires_api:
                if self.client is None:
                    is_available = False
                    status = "no_api_key"
            else:
                # Check if local model (Ollama) is running
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code != 200:
                        is_available = False
                        status = "ollama_not_running"
                except:
                    is_available = False
                    status = "ollama_not_running"
            
            models.append({
                "id": model_type.value,
                "name": config.name,
                "display_name": config.display_name,
                "description": config.description,
                "available": is_available,
                "is_local": config.is_local,
                "max_tokens": config.max_tokens,
                "cost_per_1k": config.cost_per_1k,
                "status": status
            })
        
        return models
    
    def select_model(self, model_id: str) -> bool:
        """Select a model by ID"""
        for model_type in ModelType:
            if model_type.value == model_id:
                self.selected_model = model_type
                print(f"✅ Model switched to: {MODEL_CONFIGS[model_type].display_name}")
                return True
        return False
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        config = MODEL_CONFIGS[self.selected_model]
        return {
            "id": self.selected_model.value,
            "name": config.name,
            "display_name": config.display_name,
            "description": config.description,
            "is_local": config.is_local,
            "max_tokens": config.max_tokens,
            "cost_per_1k": config.cost_per_1k
        }
    
    async def execute_chat(self, messages: List[Dict], stream: bool = False):
        """Execute chat with the selected model"""
        config = MODEL_CONFIGS[self.selected_model]
        
        if config.requires_api:
            return await self._execute_openai_chat(messages, config, stream)
        else:
            # Local model execution is synchronous
            return self._execute_local_chat(messages, config, stream)
    
    async def _execute_openai_chat(self, messages: List[Dict], config: SimpleModelConfig, stream: bool):
        """Execute OpenAI chat"""
        if self.client is None:
            raise Exception("OpenAI client not available - no API key configured")
        
        print(f"🤖 Using OpenAI model: {config.display_name}")
        response = self.client.chat.completions.create(
            model=config.name,
            messages=messages,
            max_tokens=4000,
            temperature=0.7,
            stream=stream
        )
        return response
    
    def _execute_local_chat(self, messages: List[Dict], config: SimpleModelConfig, stream: bool):
        """Execute local model chat (synchronous)"""
        # Format messages for Ollama
        prompt = self._format_messages_for_local(messages)
        print(f"🦙 Using local model: {config.display_name}")
        
        payload = {
            "model": config.name,
            "prompt": prompt,
            "stream": stream
        }
        
        response = requests.post(
            config.endpoint,
            json=payload,
            timeout=60,
            stream=stream
        )
        
        if response.status_code != 200:
            raise Exception(f"Local model error: HTTP {response.status_code}")
        
        return response
    
    def _format_messages_for_local(self, messages: List[Dict]) -> str:
        """Format OpenAI-style messages for local model"""
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"Human: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)