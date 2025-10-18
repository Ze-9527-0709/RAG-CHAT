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
        name="llama3.1:8b",
        max_tokens=8192,
        cost_per_1k_tokens=0.0,
        requires_api=False,
        endpoint="http://localhost:11434/api/generate"  # Ollama default
    )
}

class ModelFallbackManager:
    """Manages intelligent model selection and fallback"""
    
    def __init__(self, openai_client: OpenAI | None):
        self.client = openai_client
        self.current_tier = ModelTier.LOCAL if openai_client is None else ModelTier.GPT4
        self.user_preferred_model = None  # User's manually selected model
        self.last_quota_check = 0
        self.quota_cache = {}
        self.failure_counts = {tier: 0 for tier in ModelTier}
        self.last_failure_time = {tier: 0 for tier in ModelTier}
        
        # If no OpenAI client, mark all OpenAI models as unavailable
        if openai_client is None:
            self.failure_counts[ModelTier.GPT4] = 999
            self.failure_counts[ModelTier.GPT4_MINI] = 999
            self.failure_counts[ModelTier.GPT35] = 999
            print("⚠️ OpenAI client unavailable - will use local model only")
        
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
        
        # If user has manually selected a model, try to use it first
        if self.user_preferred_model is not None:
            logger.info(f"User has selected preferred model: {self.user_preferred_model.value}")
            if await self._can_use_model(self.user_preferred_model, estimated_tokens):
                if self.user_preferred_model != self.current_tier:
                    logger.info(f"Using user-selected model: {self.user_preferred_model.value}")
                    self.current_tier = self.user_preferred_model
                return self.user_preferred_model, f"User selected {self.user_preferred_model.value}"
            else:
                logger.warning(f"User-selected model {self.user_preferred_model.value} is not available, falling back to auto selection")
        
        # Check OpenAI quota if needed
        if self._should_check_quota():
            await self._check_openai_quota()
        
        # If OpenAI client is not available, skip to local model
        if self.client is None:
            logger.info("No OpenAI client available, using local model")
            if await self._check_local_model():
                self.current_tier = ModelTier.LOCAL
                return ModelTier.LOCAL, "No API key - using local model"
        
        # If too many OpenAI models have failed recently, prefer local model
        openai_failures = sum(1 for tier in [ModelTier.GPT4, ModelTier.GPT4_MINI, ModelTier.GPT35] 
                             if self.failure_counts[tier] >= self.max_retries)
        if openai_failures >= 2:  # If 2 or more OpenAI models are failing
            logger.info(f"Multiple OpenAI models failing ({openai_failures}/3), preferring local model")
            if await self._check_local_model():
                self.current_tier = ModelTier.LOCAL
                return ModelTier.LOCAL, "OpenAI models unreliable - using local model"
        
        # Try models in order of preference
        for tier in ModelTier:
            if await self._can_use_model(tier, estimated_tokens):
                if tier != self.current_tier:
                    logger.info(f"Switching from {self.current_tier.value} to {tier.value}")
                    self.current_tier = tier
                return tier, f"Auto-selected {tier.value}"
        
        # If all OpenAI models are down and local is available, use it
        if await self._check_local_model():
            logger.warning("All OpenAI models unavailable, switching to local model")
            self.current_tier = ModelTier.LOCAL
            return ModelTier.LOCAL, "All cloud models unavailable, using local"
        
        # Fallback to local even if check fails (will handle gracefully)
        logger.warning("All models unavailable, attempting local fallback")
        self.current_tier = ModelTier.LOCAL
        return ModelTier.LOCAL, "All models unavailable"
    
    async def _can_use_model(self, tier: ModelTier, estimated_tokens: int) -> bool:
        """Check if a specific model can be used"""
        config = MODEL_CONFIGS[tier]
        
        # Check if failures have expired (longer cooldown for quota issues)
        current_time = time.time()
        if self.failure_counts[tier] >= self.max_retries:
            # Different cooldown periods based on failure type
            # For quota issues: 24 hours (86400 seconds)
            # For other issues: 5 minutes (300 seconds)
            cooldown_period = 86400 if hasattr(self, f'quota_failure_{tier.value}') else 300
            
            if current_time - self.last_failure_time[tier] > cooldown_period:
                logger.info(f"Resetting failure count for {tier.value} after {cooldown_period/60:.0f} minute cooldown")
                self.failure_counts[tier] = 0
                # Clear quota failure flag if it exists
                if hasattr(self, f'quota_failure_{tier.value}'):
                    delattr(self, f'quota_failure_{tier.value}')
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
        # If no OpenAI client, all OpenAI models are unavailable
        if self.client is None:
            logger.debug(f"OpenAI client unavailable - {tier.value} not accessible")
            return False
        
        # If this model has failed recently due to quota/rate limits, don't try it
        if self.failure_counts[tier] >= self.max_retries:
            current_time = time.time()
            # Check if enough time has passed since last failure (5 minutes for quota issues)
            if current_time - self.last_failure_time.get(tier, 0) < 300:
                logger.debug(f"Model {tier.value} temporarily disabled due to recent quota failures")
                return False
            else:
                # Reset failure count after cooldown
                self.failure_counts[tier] = 0
                logger.info(f"Re-enabling {tier.value} after quota cooldown period")
            
        try:
            config = MODEL_CONFIGS[tier]
            estimated_cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens
            
            # Check cached quota info only if we have valid quota data
            remaining_balance = self.quota_cache.get('remaining_balance')
            if remaining_balance is not None and remaining_balance > 0:
                # Only enforce quota limits if we have valid quota information
                if remaining_balance < self.min_balance_threshold:
                    logger.debug(f"Insufficient balance ${remaining_balance:.2f} for {tier.value}")
                    return False
                
                if estimated_cost > remaining_balance:
                    logger.debug(f"Estimated cost ${estimated_cost:.4f} exceeds balance ${remaining_balance:.2f}")
                    return False
            else:
                # If no quota information available, allow model usage but log warning
                logger.warning(f"No quota information available for {tier.value}, allowing usage")
            
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
        # If no OpenAI client, skip quota check
        if self.client is None:
            self.quota_cache = {
                'remaining_balance': 0.0,
                'daily_limit_remaining': 0,
                'status': 'no_api_key'
            }
            return
            
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
        if self.client is None:
            raise Exception("OpenAI client not available - no API key configured")
            
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
            # Handle rate limiting and connection errors specially
            error_str = str(e)
            tier = next(t for t, c in MODEL_CONFIGS.items() if c.name == config.name)
            
            if ("rate_limit_exceeded" in error_str or "429" in error_str or 
                "insufficient_quota" in error_str):
                # Quota errors - disable for 24 hours
                self.failure_counts[tier] = self.max_retries
                self.last_failure_time[tier] = time.time()
                setattr(self, f'quota_failure_{tier.value}', True)  # Mark as quota failure
                logger.warning(f"Quota issue for {config.name}, disabling for 24 hours")
            elif "Connection error" in error_str:
                # Connection errors - disable for 5 minutes
                self.failure_counts[tier] = self.max_retries
                self.last_failure_time[tier] = time.time()
                logger.warning(f"Connection issue for {config.name}, disabling for 5 minutes")
            
            logger.error(f"OpenAI API error with {config.name}: {e}")
            raise
    
    async def _execute_local_chat(self, messages: List[Dict], config: ModelConfig, stream: bool):
        """Execute local model chat completion"""
        try:
            # Format messages for Ollama
            prompt = self._format_messages_for_local(messages)
            print(f"🦙 Local model prompt: {prompt[:100]}...")
            
            payload = {
                "model": config.name,
                "prompt": prompt,
                "stream": stream
            }
            
            print(f"🦙 Making request to {config.endpoint}")
            response = requests.post(
                config.endpoint,
                json=payload,
                timeout=60,
                stream=stream
            )
            
            print(f"🦙 Local model response status: {response.status_code}")
            if response.status_code == 200:
                self.failure_counts[ModelTier.LOCAL] = 0
                print(f"🦙 Local model response content: {response.text[:200]}...")
                return response
            else:
                raise Exception(f"Local model returned {response.status_code}")
                
        except Exception as e:
            logger.error(f"Local model error: {e}")
            print(f"🦙 Local model error: {e}")
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
            "estimated_balance": self.quota_cache.get('remaining_balance', 'unknown'),
            "is_user_selected": self.user_preferred_model == self.current_tier,
            "user_preferred_model": self.user_preferred_model.value if self.user_preferred_model else None
        }
    
    def _create_fallback_response(self):
        """Create a helpful fallback response when all models are unavailable"""
        import json
        from typing import Any
        
        fallback_message = """🤖 Sorry, all AI models are temporarily unavailable.

📋 **Current Status:**
- OpenAI Models: Quota insufficient or rate limited
- Local Models: Not installed or not running

💡 **Solutions:**
1. **OpenAI Quota**: Visit platform.openai.com to check billing
2. **Wait for Reset**: Rate limits usually reset within an hour
3. **Local Models**: You can install Ollama as backup

📊 **AI Growth Stats**: Available
🔍 **File Processing**: Available  
💬 **Chat History**: Saved

Please try again later or contact administrator for assistance."""

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