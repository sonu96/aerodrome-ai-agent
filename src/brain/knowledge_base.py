"""
Aerodrome Protocol Knowledge Base

Comprehensive knowledge management system that maintains complete protocol state,
integrates real-time data from the Aerodrome client, stores insights in Mem0 with
confidence scores, and provides intelligent query interfaces.

This system serves as the central knowledge repository for all protocol-related
information, combining real-time data with historical insights and AI-powered analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib

import structlog
from pydantic import BaseModel, Field

# Internal imports
from .confidence_scorer import (
    ConfidenceScorer,
    MemoryItem,
    MemoryCategory,
    DataSourceType,
    ConfidenceFactors
)
from ..memory import EnhancedMem0Client, MemoryMetadata
from ..protocol import AerodromeClient, PoolInfo, TokenInfo, VoterInfo, GaugeInfo
from ..intelligence import GeminiClient

logger = structlog.get_logger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge stored in the knowledge base"""
    POOL_DATA = "pool_data"
    TOKEN_DATA = "token_data"
    VOTING_DATA = "voting_data"
    MARKET_DATA = "market_data"
    PROTOCOL_METRICS = "protocol_metrics"
    HISTORICAL_ANALYSIS = "historical_analysis"
    PATTERN_INSIGHT = "pattern_insight"
    PREDICTION = "prediction"
    USER_INTERACTION = "user_interaction"


class DataFreshness(Enum):
    """Data freshness levels"""
    REAL_TIME = "real_time"      # < 1 minute
    FRESH = "fresh"              # < 5 minutes
    RECENT = "recent"            # < 1 hour
    STALE = "stale"              # < 24 hours
    EXPIRED = "expired"          # > 24 hours


@dataclass
class KnowledgeQuery:
    """Query structure for knowledge retrieval"""
    query_text: str
    knowledge_types: List[KnowledgeType] = field(default_factory=list)
    time_range: Optional[Tuple[datetime, datetime]] = None
    confidence_threshold: float = 0.3
    max_results: int = 10
    include_related: bool = True
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeItem:
    """Individual knowledge item with metadata"""
    id: str
    knowledge_type: KnowledgeType
    title: str
    content: Dict[str, Any]
    confidence: float
    freshness: DataFreshness
    created_at: datetime
    updated_at: datetime
    source: str
    tags: Set[str] = field(default_factory=set)
    related_items: Set[str] = field(default_factory=set)
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class KnowledgeResponse:
    """Response from knowledge query"""
    items: List[KnowledgeItem]
    total_found: int
    query_confidence: float
    search_time: float
    related_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProtocolStateManager:
    """Manages real-time protocol state and historical data"""
    
    def __init__(self, aerodrome_client: AerodromeClient):
        self.aerodrome_client = aerodrome_client
        self.logger = structlog.get_logger(__name__)
        
        # State caches
        self.pool_cache: Dict[str, PoolInfo] = {}
        self.token_cache: Dict[str, TokenInfo] = {}
        self.voting_cache: Dict[str, Any] = {}
        
        # Cache timestamps
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Update locks
        self._update_locks: Dict[str, asyncio.Lock] = {}
    
    async def get_pool_state(self, pool_address: str) -> Optional[PoolInfo]:
        """Get current pool state with caching"""
        try:
            cache_key = f"pool_{pool_address}"
            
            # Check cache validity
            if self._is_cache_valid(cache_key) and cache_key in self.pool_cache:
                return self.pool_cache[pool_address]
            
            # Get update lock
            if cache_key not in self._update_locks:
                self._update_locks[cache_key] = asyncio.Lock()
            
            async with self._update_locks[cache_key]:
                # Double-check cache after acquiring lock
                if self._is_cache_valid(cache_key) and pool_address in self.pool_cache:
                    return self.pool_cache[pool_address]
                
                # Fetch fresh data
                pool_data = await self.aerodrome_client.get_pool_details(pool_address)
                if pool_data:
                    self.pool_cache[pool_address] = pool_data
                    self.cache_timestamps[cache_key] = datetime.now()
                    
                return pool_data
                
        except Exception as e:
            await self.logger.aerror(
                "Error getting pool state", 
                pool_address=pool_address,
                error=str(e)
            )
            return None
    
    async def get_token_state(self, token_address: str) -> Optional[TokenInfo]:
        """Get current token state with caching"""
        try:
            cache_key = f"token_{token_address}"
            
            if self._is_cache_valid(cache_key) and token_address in self.token_cache:
                return self.token_cache[token_address]
            
            if cache_key not in self._update_locks:
                self._update_locks[cache_key] = asyncio.Lock()
            
            async with self._update_locks[cache_key]:
                if self._is_cache_valid(cache_key) and token_address in self.token_cache:
                    return self.token_cache[token_address]
                
                token_data = await self.aerodrome_client.get_token_info(token_address)
                if token_data:
                    self.token_cache[token_address] = token_data
                    self.cache_timestamps[cache_key] = datetime.now()
                    
                return token_data
                
        except Exception as e:
            await self.logger.aerror(
                "Error getting token state",
                token_address=token_address,
                error=str(e)
            )
            return None
    
    async def get_protocol_metrics(self) -> Dict[str, Any]:
        """Get current protocol-wide metrics"""
        try:
            cache_key = "protocol_metrics"
            
            if self._is_cache_valid(cache_key) and cache_key in self.voting_cache:
                return self.voting_cache[cache_key]
            
            if cache_key not in self._update_locks:
                self._update_locks[cache_key] = asyncio.Lock()
            
            async with self._update_locks[cache_key]:
                if self._is_cache_valid(cache_key) and cache_key in self.voting_cache:
                    return self.voting_cache[cache_key]
                
                # Aggregate protocol metrics
                metrics = await self._collect_protocol_metrics()
                self.voting_cache[cache_key] = metrics
                self.cache_timestamps[cache_key] = datetime.now()
                
                return metrics
                
        except Exception as e:
            await self.logger.aerror("Error getting protocol metrics", error=str(e))
            return {}
    
    async def _collect_protocol_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive protocol metrics"""
        try:
            # Get top pools for TVL calculation
            top_pools = await self.aerodrome_client.search_pools(
                limit=100,
                sort_by="tvl_desc"
            )
            
            total_tvl = sum(pool.tvl_usd for pool in top_pools)
            total_volume_24h = sum(pool.volume_24h for pool in top_pools)
            total_fees_24h = sum(pool.fees_24h for pool in top_pools)
            
            # Get voting metrics if available
            voting_metrics = {}
            try:
                current_epoch = await self.aerodrome_client.get_current_epoch()
                if current_epoch:
                    voting_analytics = await self.aerodrome_client.get_voting_analytics(current_epoch)
                    if voting_analytics:
                        voting_metrics = {
                            "current_epoch": current_epoch,
                            "total_voting_power": voting_analytics.total_voting_power,
                            "active_voters": len(voting_analytics.voters),
                            "total_bribes": voting_analytics.total_bribes_usd
                        }
            except:
                pass  # Voting data might not be available
            
            return {
                "total_tvl": total_tvl,
                "total_volume_24h": total_volume_24h,
                "total_fees_24h": total_fees_24h,
                "active_pools": len(top_pools),
                "timestamp": datetime.now().isoformat(),
                **voting_metrics
            }
            
        except Exception as e:
            await self.logger.aerror("Error collecting protocol metrics", error=str(e))
            return {}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = datetime.now() - self.cache_timestamps[cache_key]
        return age < self.cache_ttl
    
    def _determine_freshness(self, timestamp: datetime) -> DataFreshness:
        """Determine data freshness based on timestamp"""
        age = datetime.now() - timestamp
        
        if age < timedelta(minutes=1):
            return DataFreshness.REAL_TIME
        elif age < timedelta(minutes=5):
            return DataFreshness.FRESH
        elif age < timedelta(hours=1):
            return DataFreshness.RECENT
        elif age < timedelta(hours=24):
            return DataFreshness.STALE
        else:
            return DataFreshness.EXPIRED


class KnowledgeIndexer:
    """Manages knowledge indexing and search capabilities"""
    
    def __init__(self, memory_client: EnhancedMem0Client):
        self.memory_client = memory_client
        self.logger = structlog.get_logger(__name__)
        
        # Search indexes
        self.content_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[KnowledgeType, Set[str]] = {}
        
        # Relationship graph
        self.relationship_graph: Dict[str, Set[str]] = {}
    
    async def index_knowledge_item(self, item: KnowledgeItem) -> None:
        """Index a knowledge item for search and retrieval"""
        try:
            # Content indexing (simple keyword-based for now)
            keywords = self._extract_keywords(item.title + " " + str(item.content))
            for keyword in keywords:
                if keyword not in self.content_index:
                    self.content_index[keyword] = set()
                self.content_index[keyword].add(item.id)
            
            # Tag indexing
            for tag in item.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(item.id)
            
            # Type indexing
            if item.knowledge_type not in self.type_index:
                self.type_index[item.knowledge_type] = set()
            self.type_index[item.knowledge_type].add(item.id)
            
            # Relationship indexing
            for related_id in item.related_items:
                if item.id not in self.relationship_graph:
                    self.relationship_graph[item.id] = set()
                self.relationship_graph[item.id].add(related_id)
                
                if related_id not in self.relationship_graph:
                    self.relationship_graph[related_id] = set()
                self.relationship_graph[related_id].add(item.id)
            
            await self.logger.adebug(
                "Indexed knowledge item",
                item_id=item.id,
                keywords_count=len(keywords),
                tags_count=len(item.tags),
                relations_count=len(item.related_items)
            )
            
        except Exception as e:
            await self.logger.aerror(
                "Error indexing knowledge item",
                item_id=item.id,
                error=str(e)
            )
    
    async def search_knowledge(self, query: KnowledgeQuery) -> List[str]:
        """Search for knowledge items matching the query"""
        try:
            result_sets = []
            
            # Keyword search
            query_keywords = self._extract_keywords(query.query_text)
            for keyword in query_keywords:
                if keyword in self.content_index:
                    result_sets.append(self.content_index[keyword])
            
            # Type filtering
            if query.knowledge_types:
                type_results = set()
                for knowledge_type in query.knowledge_types:
                    if knowledge_type in self.type_index:
                        type_results.update(self.type_index[knowledge_type])
                if type_results:
                    result_sets.append(type_results)
            
            # Combine results (intersection for precision)
            if result_sets:
                matching_ids = result_sets[0]
                for result_set in result_sets[1:]:
                    matching_ids = matching_ids.intersection(result_set)
            else:
                matching_ids = set()
            
            # Add related items if requested
            if query.include_related and matching_ids:
                related_ids = set()
                for item_id in matching_ids:
                    if item_id in self.relationship_graph:
                        related_ids.update(self.relationship_graph[item_id])
                matching_ids.update(related_ids)
            
            return list(matching_ids)[:query.max_results]
            
        except Exception as e:
            await self.logger.aerror("Error searching knowledge", error=str(e))
            return []
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text for indexing"""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        keywords = set()
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.add(word)
        
        return keywords


class ProtocolKnowledgeBase:
    """
    Main protocol knowledge base that coordinates all knowledge management
    """
    
    def __init__(
        self,
        memory_client: EnhancedMem0Client,
        aerodrome_client: AerodromeClient,
        confidence_scorer: ConfidenceScorer,
        gemini_client: Optional[GeminiClient] = None
    ):
        """
        Initialize the protocol knowledge base.
        
        Args:
            memory_client: Mem0 client for persistent storage
            aerodrome_client: Aerodrome protocol client
            confidence_scorer: Confidence scoring system
            gemini_client: Optional AI client for insights
        """
        self.memory_client = memory_client
        self.aerodrome_client = aerodrome_client
        self.confidence_scorer = confidence_scorer
        self.gemini_client = gemini_client
        self.logger = structlog.get_logger(__name__)
        
        # Core components
        self.state_manager = ProtocolStateManager(aerodrome_client)
        self.indexer = KnowledgeIndexer(memory_client)
        
        # Knowledge storage
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        
        # Sync configuration
        self.last_sync = datetime.min
        self.sync_interval = timedelta(minutes=5)
        
        # Performance tracking
        self.query_count = 0
        self.avg_query_time = 0.0
    
    async def initialize(self) -> None:
        """Initialize the knowledge base"""
        try:
            await self.logger.ainfo("Initializing protocol knowledge base")
            
            # Load existing knowledge from memory
            await self._load_existing_knowledge()
            
            # Perform initial sync
            await self.sync_protocol_data()
            
            await self.logger.ainfo(
                "Protocol knowledge base initialized",
                items_loaded=len(self.knowledge_items)
            )
            
        except Exception as e:
            await self.logger.aerror("Failed to initialize knowledge base", error=str(e))
            raise
    
    async def _load_existing_knowledge(self) -> None:
        """Load existing knowledge items from memory storage"""
        try:
            if not self.memory_client:
                return
            
            # Search for protocol-related memories
            memories = await self.memory_client.search(
                query="aerodrome protocol",
                limit=1000
            )
            
            for memory in memories:
                try:
                    # Convert memory to knowledge item
                    knowledge_item = await self._memory_to_knowledge_item(memory)
                    if knowledge_item:
                        self.knowledge_items[knowledge_item.id] = knowledge_item
                        await self.indexer.index_knowledge_item(knowledge_item)
                        
                except Exception as e:
                    await self.logger.awarning(
                        "Failed to load knowledge item from memory",
                        memory_id=getattr(memory, 'id', 'unknown'),
                        error=str(e)
                    )
            
            await self.logger.ainfo(f"Loaded {len(self.knowledge_items)} knowledge items")
            
        except Exception as e:
            await self.logger.aerror("Error loading existing knowledge", error=str(e))
    
    async def _memory_to_knowledge_item(self, memory: Any) -> Optional[KnowledgeItem]:
        """Convert a memory object to a knowledge item"""
        try:
            # This would need to be adapted based on the actual memory structure
            memory_data = getattr(memory, 'data', {})
            if not memory_data:
                return None
            
            knowledge_type_str = memory_data.get('knowledge_type', 'protocol_metrics')
            knowledge_type = KnowledgeType(knowledge_type_str)
            
            return KnowledgeItem(
                id=getattr(memory, 'id', f"mem_{hash(str(memory_data))}"),
                knowledge_type=knowledge_type,
                title=memory_data.get('title', 'Protocol Data'),
                content=memory_data.get('content', {}),
                confidence=memory_data.get('confidence', 0.5),
                freshness=DataFreshness.STALE,  # Assume stale until refreshed
                created_at=datetime.fromisoformat(
                    memory_data.get('created_at', datetime.now().isoformat())
                ),
                updated_at=datetime.fromisoformat(
                    memory_data.get('updated_at', datetime.now().isoformat())
                ),
                source=memory_data.get('source', 'memory'),
                tags=set(memory_data.get('tags', [])),
                related_items=set(memory_data.get('related_items', []))
            )
            
        except Exception as e:
            await self.logger.aerror("Error converting memory to knowledge item", error=str(e))
            return None
    
    async def sync_protocol_data(self) -> None:
        """Synchronize with real-time protocol data"""
        try:
            # Check if sync is needed
            if datetime.now() - self.last_sync < self.sync_interval:
                return
            
            await self.logger.ainfo("Starting protocol data sync")
            sync_start = datetime.now()
            
            # Sync protocol metrics
            await self._sync_protocol_metrics()
            
            # Sync top pools data
            await self._sync_pools_data()
            
            # Generate AI insights if available
            if self.gemini_client:
                await self._generate_ai_insights()
            
            self.last_sync = datetime.now()
            sync_time = (self.last_sync - sync_start).total_seconds()
            
            await self.logger.ainfo(
                "Protocol data sync completed",
                sync_time=sync_time,
                total_items=len(self.knowledge_items)
            )
            
        except Exception as e:
            await self.logger.aerror("Error syncing protocol data", error=str(e))
    
    async def _sync_protocol_metrics(self) -> None:
        """Sync protocol-wide metrics"""
        try:
            metrics = await self.state_manager.get_protocol_metrics()
            if not metrics:
                return
            
            # Create knowledge item
            item_id = f"protocol_metrics_{int(time.time())}"
            knowledge_item = KnowledgeItem(
                id=item_id,
                knowledge_type=KnowledgeType.PROTOCOL_METRICS,
                title="Current Protocol Metrics",
                content=metrics,
                confidence=0.9,  # High confidence for direct protocol data
                freshness=DataFreshness.REAL_TIME,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="aerodrome_api",
                tags={"protocol", "metrics", "tvl", "volume"}
            )
            
            # Calculate confidence score
            memory_item = MemoryItem(
                id=item_id,
                category=MemoryCategory.PROTOCOL_CONSTANTS,
                data=metrics,
                confidence=0.0,  # Will be calculated
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source_type=DataSourceType.API_ENDPOINT
            )
            
            confidence, factors = await self.confidence_scorer.calculate_confidence(memory_item)
            knowledge_item.confidence = confidence
            
            # Store and index
            self.knowledge_items[item_id] = knowledge_item
            await self.indexer.index_knowledge_item(knowledge_item)
            
            # Store in persistent memory
            await self._store_knowledge_item(knowledge_item)
            
        except Exception as e:
            await self.logger.aerror("Error syncing protocol metrics", error=str(e))
    
    async def _sync_pools_data(self) -> None:
        """Sync top pools data"""
        try:
            # Get top pools by TVL
            top_pools = await self.aerodrome_client.search_pools(
                limit=20,
                sort_by="tvl_desc"
            )
            
            for pool in top_pools:
                try:
                    # Create knowledge item for each pool
                    item_id = f"pool_{pool.address}_{int(time.time())}"
                    knowledge_item = KnowledgeItem(
                        id=item_id,
                        knowledge_type=KnowledgeType.POOL_DATA,
                        title=f"Pool: {pool.token0.symbol}/{pool.token1.symbol}",
                        content=asdict(pool),
                        confidence=0.8,
                        freshness=DataFreshness.REAL_TIME,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        source="aerodrome_api",
                        tags={
                            "pool", pool.token0.symbol.lower(), pool.token1.symbol.lower(),
                            pool.pool_type.value, "tvl", "volume"
                        }
                    )
                    
                    # Calculate confidence
                    memory_item = MemoryItem(
                        id=item_id,
                        category=MemoryCategory.POOL_PERFORMANCE,
                        data=asdict(pool),
                        confidence=0.0,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        source_type=DataSourceType.API_ENDPOINT,
                        corroborating_sources={"quicknode", "aerodrome"}
                    )
                    
                    confidence, factors = await self.confidence_scorer.calculate_confidence(
                        memory_item,
                        metadata={
                            "provider": "quicknode",
                            "sample_size": 1,
                            "uptime": 0.99
                        }
                    )
                    knowledge_item.confidence = confidence
                    
                    # Store and index
                    self.knowledge_items[item_id] = knowledge_item
                    await self.indexer.index_knowledge_item(knowledge_item)
                    await self._store_knowledge_item(knowledge_item)
                    
                except Exception as e:
                    await self.logger.awarning(
                        "Error syncing pool data",
                        pool_address=pool.address,
                        error=str(e)
                    )
                    
        except Exception as e:
            await self.logger.aerror("Error syncing pools data", error=str(e))
    
    async def _generate_ai_insights(self) -> None:
        """Generate AI-powered insights from current data"""
        try:
            if not self.gemini_client:
                return
            
            # Get recent protocol data for analysis
            recent_items = [
                item for item in self.knowledge_items.values()
                if item.knowledge_type in [KnowledgeType.PROTOCOL_METRICS, KnowledgeType.POOL_DATA]
                and datetime.now() - item.created_at < timedelta(hours=1)
            ]
            
            if len(recent_items) < 5:  # Need sufficient data for insights
                return
            
            # Prepare data for AI analysis
            data_summary = {
                "protocol_metrics": [],
                "top_pools": [],
                "timestamp": datetime.now().isoformat()
            }
            
            for item in recent_items:
                if item.knowledge_type == KnowledgeType.PROTOCOL_METRICS:
                    data_summary["protocol_metrics"].append(item.content)
                elif item.knowledge_type == KnowledgeType.POOL_DATA:
                    data_summary["top_pools"].append({
                        "symbol": item.title,
                        "tvl": item.content.get("tvl_usd", 0),
                        "volume_24h": item.content.get("volume_24h", 0),
                        "apr": item.content.get("apr", 0)
                    })
            
            # Generate insights using AI
            insight_prompt = f"""
            Analyze the following Aerodrome protocol data and provide insights:
            
            {json.dumps(data_summary, indent=2)}
            
            Please provide:
            1. Key trends in TVL and volume
            2. Pool performance analysis
            3. Market opportunities
            4. Risk factors to consider
            5. Actionable recommendations
            
            Format your response as structured JSON with confidence scores.
            """
            
            try:
                ai_response = await self.gemini_client.generate_content(insight_prompt)
                
                # Store AI insights as knowledge
                item_id = f"ai_insight_{int(time.time())}"
                knowledge_item = KnowledgeItem(
                    id=item_id,
                    knowledge_type=KnowledgeType.PATTERN_INSIGHT,
                    title="AI Protocol Analysis",
                    content={
                        "analysis": ai_response.content,
                        "data_sources": len(recent_items),
                        "analysis_timestamp": datetime.now().isoformat()
                    },
                    confidence=0.7,  # AI insights get medium confidence
                    freshness=DataFreshness.REAL_TIME,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    source="gemini_ai",
                    tags={"ai", "analysis", "insights", "trends"}
                )
                
                self.knowledge_items[item_id] = knowledge_item
                await self.indexer.index_knowledge_item(knowledge_item)
                await self._store_knowledge_item(knowledge_item)
                
                await self.logger.ainfo("Generated AI insights", insight_id=item_id)
                
            except Exception as e:
                await self.logger.awarning("Failed to generate AI insights", error=str(e))
                
        except Exception as e:
            await self.logger.aerror("Error in AI insight generation", error=str(e))
    
    async def _store_knowledge_item(self, item: KnowledgeItem) -> None:
        """Store knowledge item in persistent memory"""
        try:
            if not self.memory_client:
                return
            
            # Prepare memory data
            memory_data = {
                "knowledge_type": item.knowledge_type.value,
                "title": item.title,
                "content": item.content,
                "confidence": item.confidence,
                "freshness": item.freshness.value,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "source": item.source,
                "tags": list(item.tags),
                "related_items": list(item.related_items)
            }
            
            # Create memory with metadata
            metadata = MemoryMetadata(
                category=self._knowledge_type_to_memory_category(item.knowledge_type),
                confidence=item.confidence,
                tags=list(item.tags),
                source=item.source
            )
            
            await self.memory_client.add(
                messages=[{
                    "role": "system",
                    "content": f"Knowledge: {item.title} - {json.dumps(memory_data)}"
                }],
                user_id=f"knowledge_base_{item.id}",
                metadata=asdict(metadata)
            )
            
        except Exception as e:
            await self.logger.awarning(
                "Failed to store knowledge item",
                item_id=item.id,
                error=str(e)
            )
    
    def _knowledge_type_to_memory_category(self, knowledge_type: KnowledgeType) -> MemoryCategory:
        """Map knowledge type to memory category"""
        mapping = {
            KnowledgeType.POOL_DATA: MemoryCategory.POOL_PERFORMANCE,
            KnowledgeType.TOKEN_DATA: MemoryCategory.PROTOCOL_CONSTANTS,
            KnowledgeType.VOTING_DATA: MemoryCategory.VOTING_PATTERNS,
            KnowledgeType.MARKET_DATA: MemoryCategory.MARKET_CORRELATIONS,
            KnowledgeType.PROTOCOL_METRICS: MemoryCategory.PROTOCOL_CONSTANTS,
            KnowledgeType.HISTORICAL_ANALYSIS: MemoryCategory.MARKET_CORRELATIONS,
            KnowledgeType.PATTERN_INSIGHT: MemoryCategory.SPECULATIVE_INSIGHTS,
            KnowledgeType.PREDICTION: MemoryCategory.SPECULATIVE_INSIGHTS,
            KnowledgeType.USER_INTERACTION: MemoryCategory.SPECULATIVE_INSIGHTS
        }
        return mapping.get(knowledge_type, MemoryCategory.SPECULATIVE_INSIGHTS)
    
    async def query_knowledge(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """Query the knowledge base"""
        import time
        start_time = time.time()
        
        try:
            self.query_count += 1
            
            # Search for matching items
            matching_ids = await self.indexer.search_knowledge(query)
            
            # Filter and score results
            results = []
            for item_id in matching_ids:
                if item_id in self.knowledge_items:
                    item = self.knowledge_items[item_id]
                    
                    # Apply confidence threshold
                    if item.confidence < query.confidence_threshold:
                        continue
                    
                    # Apply time range filter
                    if query.time_range:
                        start_time_filter, end_time = query.time_range
                        if not (start_time_filter <= item.created_at <= end_time):
                            continue
                    
                    # Update access tracking
                    item.access_count += 1
                    item.last_accessed = datetime.now()
                    
                    results.append(item)
            
            # Sort by confidence and freshness
            results.sort(
                key=lambda x: (x.confidence, x.freshness.value, x.updated_at),
                reverse=True
            )
            
            # Limit results
            results = results[:query.max_results]
            
            # Calculate query confidence
            if results:
                query_confidence = sum(item.confidence for item in results) / len(results)
            else:
                query_confidence = 0.0
            
            # Generate related suggestions
            suggestions = await self._generate_related_suggestions(query, results)
            
            query_time = time.time() - start_time
            
            # Update performance metrics
            if self.query_count == 1:
                self.avg_query_time = query_time
            else:
                alpha = 0.1
                self.avg_query_time = alpha * query_time + (1 - alpha) * self.avg_query_time
            
            response = KnowledgeResponse(
                items=results,
                total_found=len(matching_ids),
                query_confidence=query_confidence,
                search_time=query_time,
                related_suggestions=suggestions,
                metadata={
                    "query_count": self.query_count,
                    "avg_query_time": self.avg_query_time,
                    "cache_hit_rate": self._calculate_cache_hit_rate()
                }
            )
            
            await self.logger.ainfo(
                "Knowledge query completed",
                query_text=query.query_text[:100],
                results_count=len(results),
                query_time=query_time,
                confidence=query_confidence
            )
            
            return response
            
        except Exception as e:
            await self.logger.aerror(
                "Error querying knowledge base",
                query=query.query_text[:100],
                error=str(e)
            )
            
            return KnowledgeResponse(
                items=[],
                total_found=0,
                query_confidence=0.0,
                search_time=time.time() - start_time,
                related_suggestions=[],
                metadata={"error": str(e)}
            )
    
    async def _generate_related_suggestions(
        self, 
        query: KnowledgeQuery, 
        results: List[KnowledgeItem]
    ) -> List[str]:
        """Generate related query suggestions"""
        try:
            suggestions = []
            
            # Extract common tags from results
            all_tags = set()
            for item in results:
                all_tags.update(item.tags)
            
            # Generate tag-based suggestions
            for tag in list(all_tags)[:5]:
                suggestions.append(f"Show me more about {tag}")
            
            # Generate type-based suggestions
            knowledge_types = set(item.knowledge_type for item in results)
            for ktype in knowledge_types:
                if ktype != KnowledgeType.PROTOCOL_METRICS:
                    suggestions.append(f"What else do you know about {ktype.value.replace('_', ' ')}?")
            
            return suggestions[:3]  # Limit to 3 suggestions
            
        except Exception as e:
            await self.logger.awarning("Error generating suggestions", error=str(e))
            return []
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for performance monitoring"""
        # This would track actual cache performance
        # For now, return a placeholder
        return 0.85
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            type_counts = {}
            freshness_counts = {}
            confidence_sum = 0.0
            
            for item in self.knowledge_items.values():
                # Count by type
                type_name = item.knowledge_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                
                # Count by freshness
                freshness_name = item.freshness.value
                freshness_counts[freshness_name] = freshness_counts.get(freshness_name, 0) + 1
                
                # Sum confidence
                confidence_sum += item.confidence
            
            total_items = len(self.knowledge_items)
            avg_confidence = confidence_sum / max(total_items, 1)
            
            return {
                "total_items": total_items,
                "avg_confidence": avg_confidence,
                "total_queries": self.query_count,
                "avg_query_time": self.avg_query_time,
                "type_distribution": type_counts,
                "freshness_distribution": freshness_counts,
                "last_sync": self.last_sync.isoformat() if self.last_sync != datetime.min else None,
                "cache_hit_rate": self._calculate_cache_hit_rate()
            }
            
        except Exception as e:
            await self.logger.aerror("Error calculating knowledge stats", error=str(e))
            return {"error": str(e)}


# Example usage
async def main():
    """Example usage of the Protocol Knowledge Base"""
    from ..memory import EnhancedMem0Client
    from ..protocol import AerodromeClient
    from ..intelligence import GeminiClient
    from .confidence_scorer import ConfidenceScorer
    
    # Initialize components (with mock configurations)
    memory_client = EnhancedMem0Client(api_key="test_key")
    aerodrome_client = AerodromeClient(quicknode_url="test_url")
    confidence_scorer = ConfidenceScorer()
    gemini_client = GeminiClient(api_key="test_key")
    
    # Create knowledge base
    knowledge_base = ProtocolKnowledgeBase(
        memory_client=memory_client,
        aerodrome_client=aerodrome_client,
        confidence_scorer=confidence_scorer,
        gemini_client=gemini_client
    )
    
    await knowledge_base.initialize()
    
    # Query the knowledge base
    query = KnowledgeQuery(
        query_text="What are the top performing pools?",
        knowledge_types=[KnowledgeType.POOL_DATA],
        confidence_threshold=0.5,
        max_results=5
    )
    
    response = await knowledge_base.query_knowledge(query)
    
    print(f"Found {len(response.items)} items with confidence {response.query_confidence:.2f}")
    for item in response.items:
        print(f"- {item.title} (confidence: {item.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())