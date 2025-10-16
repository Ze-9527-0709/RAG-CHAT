"""
Intelligent Model Fallback System for RAG Chat Backend
Automatically switches between GPT-4, GPT-3.5, and local models based on:
1. OpenAI API quota/token availability
2. API response errors
3. User preferences
"""
import os
import logging
import json
import time
import time
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass
from openai import OpenAI
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTier(Enum):
    """Model tiers from highest to lowest capability"""
    GPT4 = "gpt-4o"
    GPT4_MINI = "gpt-4o-mini"
    GPT35 = "gpt-3.5-turbo"
    LOCAL = "local-llama"

@dataclass
class ModelConfig:
    """Configuration for each model"""
    name: str
    max_tokens: int
    cost_per_1k_tokens: float  # USD
    requires_api: bool
    endpoint: Optional[str] = None

# Model configurations
MODEL_CONFIGS = {
    ModelTier.GPT4: ModelConfig(
        name="gpt-4o",
        max_tokens=128000,
        cost_per_1k_tokens=0.0025,
        requires_api=True
    ),
    ModelTier.GPT4_MINI: ModelConfig(
        name="gpt-4o-mini", 
        max_tokens=128000,
        cost_per_1k_tokens=0.00015,
        requires_api=True
    ),
    ModelTier.GPT35: ModelConfig(
        name="gpt-3.5-turbo",
        max_tokens=16385,
        cost_per_1k_tokens=0.0005,
        requires_api=True
    ),
    ModelTier.LOCAL: ModelConfig(
        name="llama2-7b-chat",
        max_tokens=4096,
        cost_per_1k_tokens=0.0,
        requires_api=False,
        endpoint="http://localhost:11434/api/generate"  # Ollama default
    )
}

class ModelFallbackManager:
    """Manages intelligent model selection and fallback"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.current_tier = ModelTier.GPT4
        self.last_quota_check = 0
        self.quota_cache = {}
        self.failure_counts = {tier: 0 for tier in ModelTier}
        self.last_failure_time = {tier: 0 for tier in ModelTier}
        
        # Load user preferences
        self.min_balance_threshold = float(os.getenv("MIN_OPENAI_BALANCE", "5.0"))  # $5 minimum
        self.max_retries = int(os.getenv("MAX_MODEL_RETRIES", "2"))
        self.quota_check_interval = int(os.getenv("QUOTA_CHECK_INTERVAL", "300"))  # 5 minutes
        
    async def get_optimal_model(self, estimated_tokens: int = 1000) -> Tuple[ModelTier, str]:
        """
        Get the best available model based on current conditions
        Returns: (ModelTier, reason)
        """
        logger.info(f"Selecting optimal model for ~{estimated_tokens} tokens")
        
        # Check OpenAI quota if needed
        if self._should_check_quota():
            await self._check_openai_quota()
        
        # Try models in order of preference
        for tier in ModelTier:
            if await self._can_use_model(tier, estimated_tokens):
                if tier != self.current_tier:
                    logger.info(f"Switching from {self.current_tier.value} to {tier.value}")
                    self.current_tier = tier
                return tier, f"Selected {tier.value}"
        
        # Fallback to local if everything fails
        logger.warning("All models unavailable, falling back to local")
        self.current_tier = ModelTier.LOCAL
        return ModelTier.LOCAL, "All cloud models unavailable"
    
    async def _can_use_model(self, tier: ModelTier, estimated_tokens: int) -> bool:
        """Check if a specific model can be used"""
        config = MODEL_CONFIGS[tier]
        
        # Check if failures have expired (5 minutes for rate limits)
        current_time = time.time()
        if self.failure_counts[tier] >= self.max_retries:
            if current_time - self.last_failure_time[tier] > 300:  # 5 minutes
                logger.info(f"Resetting failure count for {tier.value} after cooldown")
                self.failure_counts[tier] = 0
            else:
                logger.debug(f"Skipping {tier.value} due to recent failures")
                return False
        
        # Check token limits
        if estimated_tokens > config.max_tokens:
            logger.debug(f"Request too large for {tier.value}")
            return False
        
        # Check OpenAI models
        if config.requires_api:
            return await self._check_openai_availability(tier, estimated_tokens)
        
        # Check local model
        if tier == ModelTier.LOCAL:
            return await self._check_local_model()
        
        return False
    
    async def _check_openai_availability(self, tier: ModelTier, estimated_tokens: int) -> bool:
        """Check if OpenAI model is available and affordable"""
        try:
            config = MODEL_CONFIGS[tier]
            estimated_cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens
            
            # Check cached quota info
            if self.quota_cache.get('remaining_balance', 0) < self.min_balance_threshold:
                logger.debug(f"Insufficient balance for {tier.value}")
                return False
            
            if estimated_cost > self.quota_cache.get('remaining_balance', 0):
                logger.debug(f"Estimated cost ${estimated_cost:.4f} exceeds balance")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking OpenAI availability: {e}")
            return False
    
    async def _check_local_model(self) -> bool:
        """Check if local model (Ollama) is available"""
        try:
            local_config = MODEL_CONFIGS[ModelTier.LOCAL]
            if not local_config.endpoint:
                return False
                
            # Quick health check
            response = requests.get(
                local_config.endpoint.replace('/api/generate', '/api/tags'),
                timeout=5
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.debug(f"Local model not available: {e}")
            return False
    
    def _should_check_quota(self) -> bool:
        """Determine if we should refresh quota information"""
        return (time.time() - self.last_quota_check) > self.quota_check_interval
    
    async def _check_openai_quota(self):
        """Check OpenAI account quota and usage"""
        try:
            # Note: OpenAI doesn't provide direct billing API access
            # This is a placeholder for when such API becomes available
            # For now, we'll estimate based on recent usage patterns
            
            # Simulate quota check (replace with actual API when available)
            self.quota_cache = {
                'remaining_balance': float(os.getenv("OPENAI_ESTIMATED_BALANCE", "10.0")),
                'daily_limit_remaining': 1000000,  # tokens
                'last_updated': time.time()
            }
            
            self.last_quota_check = time.time()
            logger.info(f"Quota check: ${self.quota_cache['remaining_balance']:.2f} remaining")
            
        except Exception as e:
            logger.error(f"Failed to check OpenAI quota: {e}")
            # Assume limited availability on error
            self.quota_cache = {
                'remaining_balance': self.min_balance_threshold / 2,
                'daily_limit_remaining': 10000,
                'last_updated': time.time()
            }
    
    async def execute_chat(self, messages: List[Dict], tier: ModelTier, stream: bool = False):
        """Execute chat with specified model tier"""
        config = MODEL_CONFIGS[tier]
        
        try:
            if config.requires_api:
                return await self._execute_openai_chat(messages, config, stream)
            else:
                return await self._execute_local_chat(messages, config, stream)
                
        except Exception as e:
            # Record failure and try next tier
            self.failure_counts[tier] += 1
            self.last_failure_time[tier] = time.time()
            logger.error(f"Model {tier.value} failed: {e}")
            raise
    
    async def _execute_openai_chat(self, messages: List[Dict], config: ModelConfig, stream: bool):
        """Execute OpenAI chat completion"""
        try:
            response = self.client.chat.completions.create(
                model=config.name,
                messages=messages,
                max_tokens=4000,
                temperature=0.7,
                stream=stream
            )
            
            # Reset failure count on success
            tier = next(t for t, c in MODEL_CONFIGS.items() if c.name == config.name)
            self.failure_counts[tier] = 0
            
            return response
            
        except Exception as e:
            # Handle rate limiting specially
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                # Increase failure count more aggressively for rate limits
                tier = next(t for t, c in MODEL_CONFIGS.items() if c.name == config.name)
                self.failure_counts[tier] = self.max_retries  # Temporarily disable this model
                self.last_failure_time[tier] = time.time()
                logger.warning(f"Rate limit hit for {config.name}, temporarily disabling for 5 minutes")
            
            logger.error(f"OpenAI API error with {config.name}: {e}")
            raise
    
    async def _execute_local_chat(self, messages: List[Dict], config: ModelConfig, stream: bool):
        """Execute local model chat completion"""
        try:
            # Format messages for Ollama
            prompt = self._format_messages_for_local(messages)
            
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
            
            if response.status_code == 200:
                self.failure_counts[ModelTier.LOCAL] = 0
                return response
            else:
                raise Exception(f"Local model returned {response.status_code}")
                
        except Exception as e:
            logger.error(f"Local model error: {e}")
            # Return a helpful fallback response instead of crashing
            if "Connection refused" in str(e):
                logger.info("Local model unavailable, returning fallback response")
                return self._create_fallback_response()
            raise
    
    def _format_messages_for_local(self, messages: List[Dict]) -> str:
        """Format OpenAI-style messages for local model"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        formatted += "Assistant: "
        return formatted
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about currently selected model"""
        config = MODEL_CONFIGS[self.current_tier]
        return {
            "tier": self.current_tier.value,
            "model_name": config.name,
            "max_tokens": config.max_tokens,
            "cost_per_1k": config.cost_per_1k_tokens,
            "is_local": not config.requires_api,
            "failure_count": self.failure_counts[self.current_tier],
            "estimated_balance": self.quota_cache.get('remaining_balance', 'unknown')
        }
    
    def _create_fallback_response(self):
        """Create a helpful fallback response when all models are unavailable"""
        import json
        from typing import Any
        
        fallback_message = """🤖 抱歉，所有AI模型暂时不可用。

📋 **当前状态:**
- OpenAI模型: 配额不足或速率限制
- 本地模型: 未安装或未运行

💡 **解决方案:**
1. **OpenAI配额**: 访问 platform.openai.com 检查账单
2. **等待重置**: 速率限制通常会在一小时内重置
3. **本地模型**: 可以安装Ollama作为备用

📊 **AI Growth Stats功能**: 正常可用
🔍 **文件处理功能**: 正常可用
💬 **聊天历史**: 已保存

请稍后再试或联系管理员协助解决。"""

        # Create a mock response object that mimics requests.Response
        class MockResponse:
            def __init__(self, content: str):
                self.status_code = 200
                self._content = content
                
            def json(self) -> Dict[str, Any]:
                return {"response": self._content}
                
            def iter_lines(self):
                yield self._content.encode()
                
        return MockResponse(fallback_message)

# Utility function to estimate token count (rough approximation)
def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token for English"""
    return len(text) // 4

# Environment variables for configuration
"""
Add these to your .env file:

# Model fallback settings
OPENAI_ESTIMATED_BALANCE=25.50    # Your current OpenAI balance
MIN_OPENAI_BALANCE=5.00           # Minimum balance before switching to cheaper models
MAX_MODEL_RETRIES=2               # Max retries per model before fallback
QUOTA_CHECK_INTERVAL=300          # Seconds between quota checks

# Local model settings (optional)
LOCAL_MODEL_ENDPOINT=http://localhost:11434/api/generate
LOCAL_MODEL_NAME=llama2-7b-chat
"""