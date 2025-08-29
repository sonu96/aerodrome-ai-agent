"""
Memory Configuration

Configuration settings for the Mem0-powered memory system including
storage tiers, pruning parameters, and pattern extraction settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass
class MemoryConfig:
    """Configuration for memory system operations"""
    
    # Mem0 Configuration
    user_id: str = "aerodrome_agent"
    vector_store: str = "qdrant"  # or "chroma", "postgres"
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4"
    
    # Storage Configuration  
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    collection_name: str = "aerodrome_memories"
    
    # Memory Limits
    max_memories: int = 10000
    max_memory_age_days: int = 90
    max_query_memories: int = 10
    
    # Pattern Extraction
    pattern_threshold: int = 5  # Min occurrences to create pattern
    pattern_confidence_threshold: float = 0.8
    pattern_similarity_threshold: float = 0.85
    
    # Pruning Settings
    pruning_enabled: bool = True
    pruning_interval_hours: int = 24
    hot_tier_days: int = 7
    warm_tier_days: int = 30
    cold_tier_days: int = 60
    
    # Performance Settings
    batch_size: int = 100
    search_limit: int = 50
    embedding_cache_size: int = 1000
    
    # Quality Control
    min_memory_score: float = 0.5
    duplicate_threshold: float = 0.95
    relevance_threshold: float = 0.7
    
    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create configuration from environment variables"""
        return cls(
            user_id=os.getenv("MEMORY_USER_ID", "aerodrome_agent"),
            vector_store=os.getenv("MEMORY_VECTOR_STORE", "qdrant"),
            embedding_model=os.getenv("MEMORY_EMBEDDING_MODEL", "text-embedding-ada-002"),
            llm_model=os.getenv("MEMORY_LLM_MODEL", "gpt-4"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("MEMORY_COLLECTION", "aerodrome_memories"),
            max_memories=int(os.getenv("MEMORY_MAX_MEMORIES", "10000")),
            max_memory_age_days=int(os.getenv("MEMORY_MAX_AGE_DAYS", "90")),
            max_query_memories=int(os.getenv("MEMORY_MAX_QUERY", "10")),
            pattern_threshold=int(os.getenv("MEMORY_PATTERN_THRESHOLD", "5")),
            pattern_confidence_threshold=float(os.getenv("MEMORY_PATTERN_CONFIDENCE", "0.8")),
            pattern_similarity_threshold=float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.85")),
            pruning_enabled=os.getenv("MEMORY_PRUNING_ENABLED", "true").lower() == "true",
            pruning_interval_hours=int(os.getenv("MEMORY_PRUNING_INTERVAL", "24")),
            hot_tier_days=int(os.getenv("MEMORY_HOT_TIER_DAYS", "7")),
            warm_tier_days=int(os.getenv("MEMORY_WARM_TIER_DAYS", "30")),
            cold_tier_days=int(os.getenv("MEMORY_COLD_TIER_DAYS", "60")),
            batch_size=int(os.getenv("MEMORY_BATCH_SIZE", "100")),
            search_limit=int(os.getenv("MEMORY_SEARCH_LIMIT", "50")),
            embedding_cache_size=int(os.getenv("MEMORY_CACHE_SIZE", "1000")),
            min_memory_score=float(os.getenv("MEMORY_MIN_SCORE", "0.5")),
            duplicate_threshold=float(os.getenv("MEMORY_DUPLICATE_THRESHOLD", "0.95")),
            relevance_threshold=float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.7"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_mem0_config(self) -> Dict[str, Any]:
        """Convert to Mem0 configuration format"""
        config = {
            "vector_store": {
                "provider": self.vector_store,
                "config": {
                    "host": self.qdrant_host,
                    "port": self.qdrant_port,
                    "collection_name": self.collection_name
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model
                }
            }
        }
        
        if self.qdrant_api_key:
            config["vector_store"]["config"]["api_key"] = self.qdrant_api_key
        
        return config
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if not self.user_id:
            raise ValueError("user_id is required")
        
        if self.max_memories <= 0:
            raise ValueError("max_memories must be positive")
        
        if self.max_memory_age_days <= 0:
            raise ValueError("max_memory_age_days must be positive")
        
        if self.pattern_threshold <= 0:
            raise ValueError("pattern_threshold must be positive")
        
        if self.pattern_confidence_threshold < 0 or self.pattern_confidence_threshold > 1:
            raise ValueError("pattern_confidence_threshold must be between 0 and 1")
        
        if self.hot_tier_days >= self.warm_tier_days:
            raise ValueError("hot_tier_days must be less than warm_tier_days")
        
        if self.warm_tier_days >= self.cold_tier_days:
            raise ValueError("warm_tier_days must be less than cold_tier_days")
        
        if self.min_memory_score < 0 or self.min_memory_score > 1:
            raise ValueError("min_memory_score must be between 0 and 1")
        
        if self.duplicate_threshold < 0 or self.duplicate_threshold > 1:
            raise ValueError("duplicate_threshold must be between 0 and 1")