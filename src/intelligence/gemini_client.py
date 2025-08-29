"""
Advanced Google Gemini 2.0 Client

Sophisticated Gemini AI integration with support for:
- Gemini 2.0 Flash and Pro models
- Function calling capabilities
- Context caching for cost optimization
- Streaming responses
- Safety settings configuration
- Structured output with Pydantic models
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Type, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import SafetySetting, HarmCategory, HarmBlockThreshold
from google.generativeai.caching import CachedContent
from pydantic import BaseModel, Field, ValidationError
import structlog
import vertexai
from vertexai.generative_models import GenerativeModel
from google.auth import default
from google.auth.transport.requests import Request

# Type variable for generic Pydantic models
T = TypeVar('T', bound=BaseModel)

logger = structlog.get_logger(__name__)


class GeminiModel(Enum):
    """Available Gemini 2.0 models"""
    FLASH = "gemini-2.0-flash-001"
    PRO = "gemini-2.0-pro-001"  # When available
    FLASH_EXP = "gemini-2.0-flash-exp"  # Experimental


class SafetyLevel(Enum):
    """Safety configuration levels"""
    STRICT = "strict"
    DEFAULT = "default"
    PERMISSIVE = "permissive"


@dataclass
class ModelConfig:
    """Configuration for Gemini model parameters"""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    response_mime_type: str = "text/plain"
    
    def to_generation_config(self) -> Dict[str, Any]:
        """Convert to Gemini generation config format"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": self.response_mime_type
        }


@dataclass
class FunctionSpec:
    """Function specification for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_function_declaration(self) -> Dict[str, Any]:
        """Convert to Gemini function declaration format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class StructuredOutput(BaseModel):
    """Base class for structured outputs"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    
    class Config:
        extra = "allow"


class GeminiResponse(BaseModel):
    """Wrapper for Gemini API responses"""
    content: str
    model_used: str
    tokens_used: Optional[int] = None
    cached: bool = False
    function_calls: List[Dict[str, Any]] = Field(default_factory=list)
    safety_ratings: List[Dict[str, Any]] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    response_time: float = 0.0


class GeminiClient:
    """
    Advanced Gemini 2.0 client with enterprise features
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,  # Optional - use ADC if not provided
        project_id: Optional[str] = None,
        location: str = "us-central1",
        default_model: GeminiModel = GeminiModel.FLASH,
        safety_level: SafetyLevel = SafetyLevel.DEFAULT,
        enable_caching: bool = True,
        cache_ttl_hours: int = 24,
        request_timeout: int = 60
    ):
        """
        Initialize the Gemini client
        
        Args:
            api_key: Optional Google AI API key (if None, uses Application Default Credentials)
            project_id: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud location for Vertex AI
            default_model: Default model to use
            safety_level: Safety configuration level
            enable_caching: Enable context caching
            cache_ttl_hours: Cache time-to-live in hours
            request_timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.default_model = default_model
        self.safety_level = safety_level
        self.enable_caching = enable_caching
        self.cache_ttl_hours = cache_ttl_hours
        self.request_timeout = request_timeout
        
        # Initialize client based on authentication method
        if api_key:
            # Use API key authentication
            genai.configure(api_key=api_key)
            self.use_vertex = False
        else:
            # Use Application Default Credentials with Vertex AI
            if not project_id:
                # Try to get project ID from environment or ADC
                import os
                project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
                if not project_id:
                    credentials, project = default()
                    project_id = project
            
            if not project_id:
                raise ValueError("project_id is required when using Application Default Credentials")
            
            vertexai.init(project=project_id, location=location)
            self.use_vertex = True
        
        # Cache management
        self._cache_registry: Dict[str, CachedContent] = {}
        self._function_registry: Dict[str, FunctionSpec] = {}
        
        # Safety settings
        self._safety_settings = self._get_safety_settings(safety_level)
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests
        
        logger.info(
            "Gemini client initialized",
            model=default_model.value,
            safety_level=safety_level.value,
            caching_enabled=enable_caching
        )
    
    def _get_safety_settings(self, level: SafetyLevel) -> List[SafetySetting]:
        """Get safety settings based on level"""
        if level == SafetyLevel.STRICT:
            threshold = HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        elif level == SafetyLevel.PERMISSIVE:
            threshold = HarmBlockThreshold.BLOCK_NONE
        else:  # DEFAULT
            threshold = HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=threshold
            )
        ]
    
    def register_function(self, func_spec: FunctionSpec) -> None:
        """Register a function for function calling"""
        self._function_registry[func_spec.name] = func_spec
        logger.info("Function registered", function_name=func_spec.name)
    
    def register_multiple_functions(self, func_specs: List[FunctionSpec]) -> None:
        """Register multiple functions"""
        for spec in func_specs:
            self.register_function(spec)
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    async def create_cache(
        self,
        cache_key: str,
        content: List[str],
        ttl_hours: Optional[int] = None
    ) -> Optional[str]:
        """
        Create a cached content for cost optimization
        
        Args:
            cache_key: Unique key for the cache
            content: List of content strings to cache
            ttl_hours: Time-to-live in hours (defaults to instance setting)
            
        Returns:
            Cache name if successful, None if failed
        """
        if not self.enable_caching:
            return None
        
        try:
            ttl = ttl_hours or self.cache_ttl_hours
            ttl_delta = timedelta(hours=ttl)
            
            # Create cached content
            cache_content = await genai.caching.CachedContent.create_async(
                model=self.default_model.value,
                system_instruction="You are an expert DeFi and blockchain analyst.",
                contents=content,
                ttl=ttl_delta,
                display_name=f"aerodrome_cache_{cache_key}"
            )
            
            self._cache_registry[cache_key] = cache_content
            
            logger.info(
                "Cache created successfully",
                cache_key=cache_key,
                cache_name=cache_content.name,
                ttl_hours=ttl
            )
            
            return cache_content.name
            
        except Exception as e:
            logger.error("Failed to create cache", error=str(e), cache_key=cache_key)
            return None
    
    async def get_cached_model(self, cache_key: str) -> Optional[genai.GenerativeModel]:
        """Get a model with cached content"""
        if cache_key not in self._cache_registry:
            return None
        
        try:
            cached_content = self._cache_registry[cache_key]
            model = genai.GenerativeModel.from_cached_content(cached_content)
            return model
        except Exception as e:
            logger.error("Failed to get cached model", error=str(e), cache_key=cache_key)
            return None
    
    async def generate_content(
        self,
        prompt: str,
        model: Optional[GeminiModel] = None,
        config: Optional[ModelConfig] = None,
        cache_key: Optional[str] = None,
        functions: Optional[List[str]] = None,
        system_instruction: Optional[str] = None
    ) -> GeminiResponse:
        """
        Generate content with advanced options
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to instance default)
            config: Generation configuration
            cache_key: Use cached content
            functions: List of function names to enable
            system_instruction: System instruction for the model
            
        Returns:
            GeminiResponse with detailed information
        """
        await self._rate_limit()
        
        start_time = time.time()
        model_name = (model or self.default_model).value
        config = config or ModelConfig()
        
        try:
            # Get model (cached or fresh)
            if cache_key and cache_key in self._cache_registry:
                gemini_model = await self.get_cached_model(cache_key)
                cached = True
            else:
                # Prepare function declarations
                function_declarations = []
                if functions:
                    for func_name in functions:
                        if func_name in self._function_registry:
                            spec = self._function_registry[func_name]
                            function_declarations.append(spec.to_function_declaration())
                
                tools = [{"function_declarations": function_declarations}] if function_declarations else None
                
                gemini_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=config.to_generation_config(),
                    safety_settings=self._safety_settings,
                    tools=tools,
                    system_instruction=system_instruction
                )
                cached = False
            
            # Generate content
            response = await gemini_model.generate_content_async(prompt)
            
            # Process response
            content = response.text if response.text else ""
            
            # Extract function calls
            function_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append({
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args)
                        })
            
            # Extract safety ratings
            safety_ratings = []
            if response.candidates and response.candidates[0].safety_ratings:
                for rating in response.candidates[0].safety_ratings:
                    safety_ratings.append({
                        "category": rating.category.name,
                        "probability": rating.probability.name
                    })
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract usage metadata
            tokens_used = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = response.usage_metadata.total_token_count
            
            finish_reason = None
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = response.candidates[0].finish_reason.name
            
            gemini_response = GeminiResponse(
                content=content,
                model_used=model_name,
                tokens_used=tokens_used,
                cached=cached,
                function_calls=function_calls,
                safety_ratings=safety_ratings,
                finish_reason=finish_reason,
                response_time=response_time
            )
            
            logger.info(
                "Content generated successfully",
                model=model_name,
                tokens=tokens_used,
                response_time=response_time,
                cached=cached,
                function_calls_count=len(function_calls)
            )
            
            return gemini_response
            
        except Exception as e:
            logger.error(
                "Content generation failed",
                error=str(e),
                model=model_name,
                prompt_length=len(prompt)
            )
            raise
    
    async def generate_structured_content(
        self,
        prompt: str,
        output_schema: Type[T],
        model: Optional[GeminiModel] = None,
        config: Optional[ModelConfig] = None,
        cache_key: Optional[str] = None,
        max_retries: int = 3
    ) -> T:
        """
        Generate structured content with Pydantic validation
        
        Args:
            prompt: Input prompt
            output_schema: Pydantic model class for output
            model: Model to use
            config: Generation configuration  
            cache_key: Use cached content
            max_retries: Maximum retry attempts for parsing
            
        Returns:
            Validated Pydantic model instance
        """
        # Configure for JSON output
        if not config:
            config = ModelConfig()
        config.response_mime_type = "application/json"
        
        # Add schema to prompt
        schema_prompt = f"""
        {prompt}
        
        Please respond with valid JSON that matches this schema:
        {output_schema.schema_json(indent=2)}
        
        Ensure the response is properly formatted JSON.
        """
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.generate_content(
                    prompt=schema_prompt,
                    model=model,
                    config=config,
                    cache_key=cache_key
                )
                
                # Parse JSON response
                json_data = json.loads(response.content)
                
                # Validate with Pydantic
                structured_output = output_schema(**json_data)
                
                logger.info(
                    "Structured content generated successfully",
                    schema=output_schema.__name__,
                    attempt=attempt + 1
                )
                
                return structured_output
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    "Failed to parse structured content",
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=max_retries
                )
                
                if attempt == max_retries:
                    logger.error(
                        "Max retries exceeded for structured content",
                        schema=output_schema.__name__
                    )
                    raise
                
                # Wait before retry
                await asyncio.sleep(1.0 * (attempt + 1))
    
    async def stream_content(
        self,
        prompt: str,
        model: Optional[GeminiModel] = None,
        config: Optional[ModelConfig] = None,
        cache_key: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream content generation
        
        Args:
            prompt: Input prompt
            model: Model to use
            config: Generation configuration
            cache_key: Use cached content
            
        Yields:
            Content chunks as they are generated
        """
        await self._rate_limit()
        
        model_name = (model or self.default_model).value
        config = config or ModelConfig()
        
        try:
            # Get model (cached or fresh)
            if cache_key and cache_key in self._cache_registry:
                gemini_model = await self.get_cached_model(cache_key)
            else:
                gemini_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=config.to_generation_config(),
                    safety_settings=self._safety_settings
                )
            
            # Stream content
            response = await gemini_model.generate_content_async(
                prompt,
                stream=True
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
            
            logger.info("Content streaming completed", model=model_name)
            
        except Exception as e:
            logger.error("Content streaming failed", error=str(e), model=model_name)
            raise
    
    async def analyze_with_search_grounding(
        self,
        query: str,
        search_context: Optional[str] = None,
        model: Optional[GeminiModel] = None,
        config: Optional[ModelConfig] = None
    ) -> GeminiResponse:
        """
        Analyze with search grounding (when available)
        
        Args:
            query: Analysis query
            search_context: Additional search context
            model: Model to use
            config: Generation configuration
            
        Returns:
            GeminiResponse with grounded analysis
        """
        grounded_prompt = f"""
        Analyze the following query with real-time information and context:
        
        Query: {query}
        
        {f"Additional Context: {search_context}" if search_context else ""}
        
        Provide a comprehensive analysis based on current market conditions,
        recent developments, and relevant data. Include specific examples and
        quantitative insights where possible.
        """
        
        return await self.generate_content(
            prompt=grounded_prompt,
            model=model,
            config=config,
            system_instruction="""You are an expert DeFi analyst with access to 
            real-time market data. Provide accurate, current, and actionable insights."""
        )
    
    async def cleanup_expired_caches(self) -> None:
        """Clean up expired cached content"""
        if not self.enable_caching:
            return
        
        expired_keys = []
        
        for cache_key, cached_content in self._cache_registry.items():
            try:
                # Check if cache is still valid
                if hasattr(cached_content, 'expire_time'):
                    if datetime.utcnow() > cached_content.expire_time:
                        expired_keys.append(cache_key)
                        await cached_content.delete_async()
            except Exception as e:
                logger.warning(
                    "Error checking cache expiry",
                    cache_key=cache_key,
                    error=str(e)
                )
                expired_keys.append(cache_key)
        
        # Remove expired caches from registry
        for key in expired_keys:
            del self._cache_registry[key]
        
        if expired_keys:
            logger.info("Expired caches cleaned up", count=len(expired_keys))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "default_model": self.default_model.value,
            "safety_level": self.safety_level.value,
            "caching_enabled": self.enable_caching,
            "cached_content_count": len(self._cache_registry),
            "registered_functions_count": len(self._function_registry),
            "registered_functions": list(self._function_registry.keys())
        }