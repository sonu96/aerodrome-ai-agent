"""
Enhanced Mem0 client with advanced features for the Aerodrome AI Agent.

This module provides an enhanced Mem0 client with support for graph memory,
batch operations, advanced filtering, memory export/import, and async operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Tuple
from contextlib import asynccontextmanager
import aiohttp
from dataclasses import asdict

try:
    from mem0 import Memory
    from mem0.client.main import MemoryClient
except ImportError:
    logging.warning("mem0ai not installed. Some features will be unavailable.")
    Memory = None
    MemoryClient = None

from .memory_categories import MemoryCategory, MemoryCategoryConfig, MemoryMetadata

logger = logging.getLogger(__name__)


class EnhancedMem0Client:
    """Enhanced Mem0 client with advanced features."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: str = "https://api.mem0.ai",
        version: str = "v2",
        config: Optional[Dict[str, Any]] = None,
        enable_graph: bool = True,
        neo4j_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced Mem0 client.
        
        Args:
            api_key: Mem0 API key
            host: Mem0 API host
            version: API version
            config: Additional configuration
            enable_graph: Enable graph memory features
            neo4j_config: Neo4j configuration for graph memory
        """
        self.api_key = api_key
        self.host = host
        self.version = version
        self.enable_graph = enable_graph
        self.neo4j_config = neo4j_config or {}
        
        # Initialize base client
        if Memory and MemoryClient:
            try:
                self.memory = Memory()
                self.client = MemoryClient(api_key=api_key, host=host, version=version)
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client: {e}")
                self.memory = None
                self.client = None
        else:
            self.memory = None
            self.client = None
            
        # Configuration
        self.config = config or {}
        self.batch_size = self.config.get("batch_size", 100)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        # Graph memory configuration
        if enable_graph and neo4j_config:
            self._init_graph_config()
            
        # Session management
        self._session = None
        
    def _init_graph_config(self):
        """Initialize graph memory configuration."""
        self.graph_config = {
            "provider": "neo4j",
            "config": {
                "url": self.neo4j_config.get("url", "bolt://localhost:7687"),
                "username": self.neo4j_config.get("username", "neo4j"),
                "password": self.neo4j_config.get("password", "password"),
                "database": self.neo4j_config.get("database", "neo4j")
            }
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        url = f"{self.host}/{self.version}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            async with self._session.request(
                method, url, json=data, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429 and retry_count < self.max_retries:
                    # Rate limiting - retry with exponential backoff
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))
                    return await self._make_request(method, endpoint, data, params, retry_count + 1)
                else:
                    response.raise_for_status()
                    
        except Exception as e:
            if retry_count < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** retry_count))
                return await self._make_request(method, endpoint, data, params, retry_count + 1)
            raise e
            
    async def add_memory(
        self,
        content: str,
        category: MemoryCategory,
        confidence: float,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        enable_graph: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Add memory with enhanced metadata and optional graph storage.
        
        Args:
            content: Memory content
            category: Memory category
            confidence: Confidence score (0.0 to 1.0)
            user_id: User ID for scoped memories
            metadata: Additional metadata
            enable_graph: Override graph storage for this memory
            
        Returns:
            Memory creation response
        """
        # Create standardized metadata
        memory_metadata = MemoryMetadata.create_metadata(
            category=category,
            confidence=confidence,
            source="aerodrome_agent",
            created_at=datetime.utcnow().isoformat(),
            **(metadata or {})
        )
        
        # Prepare memory data
        memory_data = {
            "text": content,
            "metadata": memory_metadata
        }
        
        if user_id:
            memory_data["user_id"] = user_id
            
        # Add graph configuration if enabled
        use_graph = enable_graph if enable_graph is not None else (
            self.enable_graph and MemoryCategoryConfig.requires_graph_pruning(category)
        )
        
        if use_graph and self.graph_config:
            memory_data["graph"] = self.graph_config
            
        try:
            if self.client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.create(**memory_data)
                )
                logger.info(f"Added memory for category {category.value}")
                return response
            else:
                # Direct API call
                return await self._make_request("POST", "memories", data=memory_data)
                
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
            
    async def batch_add_memories(
        self,
        memories: List[Dict[str, Any]],
        category: MemoryCategory,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Add multiple memories in batch.
        
        Args:
            memories: List of memory dictionaries with 'content', 'confidence', and optional 'metadata'
            category: Memory category for all memories
            user_id: User ID for scoped memories
            
        Returns:
            List of creation responses
        """
        batch_size = MemoryCategoryConfig.get_batch_size(category)
        results = []
        
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            batch_tasks = []
            
            for memory_data in batch:
                task = self.add_memory(
                    content=memory_data["content"],
                    category=category,
                    confidence=memory_data["confidence"],
                    user_id=user_id,
                    metadata=memory_data.get("metadata")
                )
                batch_tasks.append(task)
                
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch memory creation failed: {result}")
                else:
                    results.append(result)
                    
            # Rate limiting between batches
            await asyncio.sleep(0.1)
            
        logger.info(f"Batch added {len(results)} memories for category {category.value}")
        return results
        
    async def search_memories(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        user_id: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_graph: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search memories with advanced filtering using v2 API.
        
        Args:
            query: Search query
            category: Filter by memory category
            user_id: User ID for scoped search
            confidence_threshold: Minimum confidence score
            limit: Maximum number of results
            filters: Additional metadata filters
            use_graph: Use graph-based search
            
        Returns:
            List of matching memories
        """
        search_params = {
            "text": query,
            "limit": limit
        }
        
        if user_id:
            search_params["user_id"] = user_id
            
        # Build metadata filters
        metadata_filters = {}
        
        if category:
            metadata_filters["category"] = category.value
            # Add category-specific filters
            custom_filters = MemoryCategoryConfig.get_custom_filters(category)
            metadata_filters.update(custom_filters)
            
        if confidence_threshold:
            metadata_filters["confidence"] = {"$gte": confidence_threshold}
            
        if filters:
            metadata_filters.update(filters)
            
        if metadata_filters:
            search_params["filters"] = metadata_filters
            
        # Graph search configuration
        if use_graph and self.graph_config:
            search_params["graph"] = True
            
        try:
            if self.client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.search(**search_params)
                )
                return response.get("results", [])
            else:
                # Direct API call
                response = await self._make_request("POST", "memories/search", data=search_params)
                return response.get("results", [])
                
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
            
    async def get_memories_by_category(
        self,
        category: MemoryCategory,
        user_id: Optional[str] = None,
        limit: int = 1000,
        include_low_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for a specific category.
        
        Args:
            category: Memory category
            user_id: User ID for scoped retrieval
            limit: Maximum number of memories
            include_low_confidence: Include memories below category threshold
            
        Returns:
            List of memories
        """
        filters = {"category": category.value}
        
        if not include_low_confidence:
            policy = MemoryCategoryConfig.get_policy(category)
            filters["confidence"] = {"$gte": policy.min_confidence_threshold}
            
        return await self.search_memories(
            query="*",  # Match all
            user_id=user_id,
            limit=limit,
            filters=filters
        )
        
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update existing memory.
        
        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            metadata: Updated metadata (optional)
            user_id: User ID for scoped update
            
        Returns:
            Update response
        """
        update_data = {}
        
        if content:
            update_data["text"] = content
            
        if metadata:
            # Update access metadata
            updated_metadata = MemoryMetadata.update_access_metadata(metadata)
            update_data["metadata"] = updated_metadata
            
        if user_id:
            update_data["user_id"] = user_id
            
        try:
            if self.client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.update(memory_id, **update_data)
                )
                return response
            else:
                # Direct API call
                return await self._make_request("PUT", f"memories/{memory_id}", data=update_data)
                
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            raise
            
    async def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete memory by ID.
        
        Args:
            memory_id: Memory ID to delete
            user_id: User ID for scoped deletion
            
        Returns:
            Success status
        """
        try:
            delete_params = {}
            if user_id:
                delete_params["user_id"] = user_id
                
            if self.client:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.delete(memory_id, **delete_params)
                )
            else:
                # Direct API call
                await self._make_request("DELETE", f"memories/{memory_id}", params=delete_params)
                
            logger.info(f"Deleted memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
            
    async def batch_delete_memories(
        self,
        memory_ids: List[str],
        user_id: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Delete multiple memories in batch.
        
        Args:
            memory_ids: List of memory IDs to delete
            user_id: User ID for scoped deletion
            batch_size: Batch size (uses default if not specified)
            
        Returns:
            Dictionary mapping memory IDs to success status
        """
        if not batch_size:
            batch_size = self.batch_size
            
        results = {}
        
        for i in range(0, len(memory_ids), batch_size):
            batch = memory_ids[i:i + batch_size]
            batch_tasks = []
            
            for memory_id in batch:
                task = self.delete_memory(memory_id, user_id)
                batch_tasks.append((memory_id, task))
                
            batch_results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)
            
            for (memory_id, _), result in zip(batch_tasks, batch_results):
                results[memory_id] = not isinstance(result, Exception) and result
                
            # Rate limiting between batches
            await asyncio.sleep(0.1)
            
        logger.info(f"Batch deleted {sum(results.values())} out of {len(memory_ids)} memories")
        return results
        
    async def export_memories(
        self,
        category: Optional[MemoryCategory] = None,
        user_id: Optional[str] = None,
        format_type: str = "json",
        include_metadata: bool = True
    ) -> Union[Dict[str, Any], str]:
        """
        Export memories to various formats.
        
        Args:
            category: Filter by category (optional)
            user_id: User ID for scoped export
            format_type: Export format ('json', 'csv', 'jsonl')
            include_metadata: Include memory metadata
            
        Returns:
            Exported data in specified format
        """
        # Get memories
        if category:
            memories = await self.get_memories_by_category(category, user_id, limit=10000)
        else:
            memories = await self.search_memories("*", user_id=user_id, limit=10000)
            
        if format_type == "json":
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "memory_count": len(memories),
                "category": category.value if category else "all",
                "memories": []
            }
            
            for memory in memories:
                memory_data = {
                    "id": memory.get("id"),
                    "content": memory.get("text", memory.get("content")),
                }
                
                if include_metadata:
                    memory_data["metadata"] = memory.get("metadata", {})
                    
                export_data["memories"].append(memory_data)
                
            return export_data
            
        elif format_type == "jsonl":
            lines = []
            for memory in memories:
                memory_data = {
                    "id": memory.get("id"),
                    "content": memory.get("text", memory.get("content")),
                }
                
                if include_metadata:
                    memory_data["metadata"] = memory.get("metadata", {})
                    
                lines.append(json.dumps(memory_data))
                
            return "\n".join(lines)
            
        elif format_type == "csv":
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = ["id", "content", "category", "confidence", "created_at"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for memory in memories:
                metadata = memory.get("metadata", {})
                writer.writerow({
                    "id": memory.get("id"),
                    "content": memory.get("text", memory.get("content")),
                    "category": metadata.get("category"),
                    "confidence": metadata.get("confidence"),
                    "created_at": metadata.get("created_at")
                })
                
            return output.getvalue()
            
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
    async def import_memories(
        self,
        data: Union[Dict[str, Any], str, List[Dict[str, Any]]],
        format_type: str = "json",
        default_category: Optional[MemoryCategory] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import memories from various formats.
        
        Args:
            data: Import data in specified format
            format_type: Import format ('json', 'csv', 'jsonl')
            default_category: Default category for memories without category
            user_id: User ID for scoped import
            
        Returns:
            Import results summary
        """
        memories_to_import = []
        
        if format_type == "json":
            if isinstance(data, dict) and "memories" in data:
                memories_data = data["memories"]
            elif isinstance(data, list):
                memories_data = data
            else:
                raise ValueError("Invalid JSON format for import")
                
            for memory_data in memories_data:
                memory_info = {
                    "content": memory_data.get("content", ""),
                    "metadata": memory_data.get("metadata", {}),
                    "confidence": memory_data.get("metadata", {}).get("confidence", 0.5)
                }
                memories_to_import.append(memory_info)
                
        elif format_type == "jsonl":
            lines = data.strip().split("\n")
            for line in lines:
                if line.strip():
                    memory_data = json.loads(line)
                    memory_info = {
                        "content": memory_data.get("content", ""),
                        "metadata": memory_data.get("metadata", {}),
                        "confidence": memory_data.get("metadata", {}).get("confidence", 0.5)
                    }
                    memories_to_import.append(memory_info)
                    
        else:
            raise ValueError(f"Unsupported import format: {format_type}")
            
        # Determine categories and import
        results = {"total": len(memories_to_import), "success": 0, "failed": 0, "errors": []}
        
        for memory_info in memories_to_import:
            try:
                metadata = memory_info["metadata"]
                category_str = metadata.get("category")
                
                if category_str:
                    category = MemoryCategory(category_str)
                elif default_category:
                    category = default_category
                else:
                    category = MemoryCategory.SPECULATIVE_INSIGHTS  # Default fallback
                    
                await self.add_memory(
                    content=memory_info["content"],
                    category=category,
                    confidence=memory_info["confidence"],
                    user_id=user_id,
                    metadata=metadata
                )
                
                results["success"] += 1
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(str(e))
                logger.error(f"Failed to import memory: {e}")
                
        logger.info(f"Import completed: {results['success']} success, {results['failed']} failed")
        return results
        
    async def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Args:
            user_id: User ID for scoped stats
            
        Returns:
            Memory statistics
        """
        stats = {
            "total_memories": 0,
            "by_category": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "age_distribution": {"recent": 0, "old": 0, "ancient": 0},
            "graph_memories": 0
        }
        
        # Get all memories
        all_memories = await self.search_memories("*", user_id=user_id, limit=10000)
        stats["total_memories"] = len(all_memories)
        
        now = datetime.utcnow()
        
        for memory in all_memories:
            metadata = memory.get("metadata", {})
            
            # Category distribution
            category = metadata.get("category", "unknown")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # Confidence distribution
            confidence = metadata.get("confidence", 0.5)
            if confidence >= 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif confidence >= 0.6:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
                
            # Age distribution
            created_at_str = metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    age = now - created_at.replace(tzinfo=None)
                    
                    if age.days <= 7:
                        stats["age_distribution"]["recent"] += 1
                    elif age.days <= 90:
                        stats["age_distribution"]["old"] += 1
                    else:
                        stats["age_distribution"]["ancient"] += 1
                except:
                    pass
                    
            # Graph memories
            if metadata.get("relationships") or "graph" in memory:
                stats["graph_memories"] += 1
                
        return stats