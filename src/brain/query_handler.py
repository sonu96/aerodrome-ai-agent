"""
Aerodrome Query Handler

Advanced natural language query processing system that:
- Understands user intent and context
- Routes queries to appropriate components
- Aggregates responses from multiple sources
- Formats responses with confidence scores
- Maintains conversation history and context

This system serves as the primary interface for user interactions with the brain.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

import structlog
from pydantic import BaseModel, Field

# Internal imports
from .knowledge_base import (
    ProtocolKnowledgeBase,
    KnowledgeQuery,
    KnowledgeResponse,
    KnowledgeType,
    KnowledgeItem
)
from .confidence_scorer import ConfidenceScorer, MemoryCategory
from ..intelligence import GeminiClient, GeminiResponse

logger = structlog.get_logger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle"""
    POOL_INQUIRY = "pool_inquiry"              # Questions about specific pools
    TOKEN_INQUIRY = "token_inquiry"            # Questions about tokens
    PROTOCOL_STATUS = "protocol_status"        # Protocol metrics and health
    VOTING_INQUIRY = "voting_inquiry"          # veAERO voting questions
    MARKET_ANALYSIS = "market_analysis"        # Market trends and analysis
    TRADING_ADVICE = "trading_advice"          # Trading recommendations
    EDUCATIONAL = "educational"                # Learning about Aerodrome
    TECHNICAL = "technical"                    # Technical protocol details
    HISTORICAL = "historical"                  # Historical data queries
    PREDICTION = "prediction"                  # Future predictions
    GENERAL = "general"                        # General conversation
    UNKNOWN = "unknown"                        # Unclassified queries


class QueryIntent(Enum):
    """User intent classification"""
    INFORMATION_SEEKING = "information_seeking"
    DECISION_SUPPORT = "decision_support"
    MONITORING = "monitoring"
    LEARNING = "learning"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    EXPLORATION = "exploration"


class ResponseFormat(Enum):
    """Response formatting options"""
    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    TECHNICAL = "technical"
    SUMMARY = "summary"
    DETAILED = "detailed"


@dataclass
class QueryContext:
    """Context for a user query"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    """Analysis results for a query"""
    query_type: QueryType
    intent: QueryIntent
    entities: Dict[str, List[str]] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    confidence: float = 0.0
    requires_real_time_data: bool = False
    complexity_score: float = 0.0
    suggested_format: ResponseFormat = ResponseFormat.CONVERSATIONAL


@dataclass
class QueryResponse:
    """Response to a user query"""
    query_id: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    format: ResponseFormat = ResponseFormat.CONVERSATIONAL
    processing_time: float = 0.0


class QueryAnalyzer:
    """Analyzes queries to understand intent and extract entities"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client
        self.logger = structlog.get_logger(__name__)
        
        # Pattern-based classification (fallback if AI is not available)
        self.query_patterns = {
            QueryType.POOL_INQUIRY: [
                r'pool.*(?:performance|stats|metrics|tvl|volume|apr)',
                r'(?:best|top|worst).*pools',
                r'pool.*(?:0x[a-fA-F0-9]{40})',
                r'(?:usdc|eth|aero|weth).*pool',
                r'liquidity.*pool'
            ],
            QueryType.TOKEN_INQUIRY: [
                r'token.*(?:price|value|market|cap)',
                r'(?:aero|usdc|eth|weth).*(?:price|stats)',
                r'token.*(?:0x[a-fA-F0-9]{40})',
                r'what.*(?:aero|aerodrome.*token)'
            ],
            QueryType.PROTOCOL_STATUS: [
                r'protocol.*(?:status|health|metrics|stats)',
                r'total.*(?:tvl|volume|fees)',
                r'aerodrome.*(?:metrics|performance|status)',
                r'how.*(?:protocol|aerodrome).*doing'
            ],
            QueryType.VOTING_INQUIRY: [
                r'(?:vote|voting|veaero|gauge)',
                r'bribes?.*(?:rewards?|incentives?)',
                r'epoch.*(?:voting|rewards?)',
                r'governance.*(?:voting|proposals?)'
            ],
            QueryType.MARKET_ANALYSIS: [
                r'market.*(?:trends?|analysis|conditions?)',
                r'(?:trend|trending).*(?:up|down|bullish|bearish)',
                r'price.*(?:movement|action|trend)',
                r'market.*(?:outlook|forecast)'
            ],
            QueryType.TRADING_ADVICE: [
                r'should.*(?:buy|sell|trade|swap)',
                r'trading.*(?:advice|strategy|recommendation)',
                r'(?:investment|investing).*(?:advice|strategy)',
                r'good.*(?:entry|exit).*point'
            ],
            QueryType.EDUCATIONAL: [
                r'how.*(?:works?|does|do|to)',
                r'what.*(?:is|are|means?)',
                r'explain.*(?:aerodrome|protocol|defi)',
                r'(?:learn|teach|understand).*(?:about|aerodrome)'
            ],
            QueryType.HISTORICAL: [
                r'(?:history|historical|past|previous)',
                r'(?:yesterday|last.*(?:week|month|day))',
                r'(?:compare|comparison).*(?:historical|past)',
                r'(?:24h|7d|30d).*(?:ago|back)'
            ],
            QueryType.PREDICTION: [
                r'(?:predict|prediction|forecast|future)',
                r'what.*(?:will|expect|happen)',
                r'(?:next|upcoming).*(?:week|month|days?)',
                r'price.*(?:prediction|target|forecast)'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'pool_address': r'0x[a-fA-F0-9]{40}',
            'token_symbol': r'\b(?:USDC|ETH|WETH|AERO|BTC|WBTC|DAI|USDT)\b',
            'percentage': r'\d+(?:\.\d+)?%',
            'dollar_amount': r'\$\d+(?:,\d{3})*(?:\.\d{2})?[KMB]?',
            'time_period': r'(?:24h|7d|30d|1h|1d|1w|1m|1y)',
            'epoch_number': r'epoch\s+(\d+)'
        }
    
    async def analyze_query(self, query_context: QueryContext) -> QueryAnalysis:
        """Analyze a query to understand intent and extract information"""
        try:
            query = query_context.query.lower().strip()
            
            # Try AI-powered analysis first
            if self.gemini_client:
                ai_analysis = await self._ai_analyze_query(query_context)
                if ai_analysis:
                    return ai_analysis
            
            # Fallback to pattern-based analysis
            return await self._pattern_analyze_query(query_context)
            
        except Exception as e:
            await self.logger.aerror("Error analyzing query", error=str(e))
            # Return default analysis
            return QueryAnalysis(
                query_type=QueryType.UNKNOWN,
                intent=QueryIntent.INFORMATION_SEEKING,
                confidence=0.1
            )
    
    async def _ai_analyze_query(self, query_context: QueryContext) -> Optional[QueryAnalysis]:
        """Use AI to analyze query intent and entities"""
        try:
            analysis_prompt = f"""
            Analyze this Aerodrome DEX query and provide structured analysis:
            
            Query: "{query_context.query}"
            Context: {json.dumps(query_context.context, indent=2)}
            
            Provide analysis in JSON format with:
            1. query_type: one of {[t.value for t in QueryType]}
            2. intent: one of {[i.value for i in QueryIntent]}
            3. entities: dict with extracted entities (addresses, symbols, amounts, etc.)
            4. topics: list of main topics
            5. confidence: float 0-1
            6. requires_real_time_data: boolean
            7. complexity_score: float 0-1 (simple to complex)
            8. suggested_format: one of {[f.value for f in ResponseFormat]}
            
            Consider:
            - Aerodrome is a DEX on Base with pools, tokens, voting (veAERO), and bribes
            - Look for pool addresses (0x...), token symbols (USDC, ETH, AERO, etc.)
            - Trading advice should be marked as decision_support intent
            - Technical questions about protocol mechanics are educational
            - Real-time data needed for current prices, TVL, volumes, etc.
            """
            
            response = await self.gemini_client.generate_content(
                analysis_prompt,
                response_format="application/json"
            )
            
            if response and response.content:
                try:
                    analysis_data = json.loads(response.content)
                    
                    return QueryAnalysis(
                        query_type=QueryType(analysis_data.get('query_type', 'unknown')),
                        intent=QueryIntent(analysis_data.get('intent', 'information_seeking')),
                        entities=analysis_data.get('entities', {}),
                        topics=analysis_data.get('topics', []),
                        confidence=analysis_data.get('confidence', 0.5),
                        requires_real_time_data=analysis_data.get('requires_real_time_data', False),
                        complexity_score=analysis_data.get('complexity_score', 0.5),
                        suggested_format=ResponseFormat(
                            analysis_data.get('suggested_format', 'conversational')
                        )
                    )
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    await self.logger.awarning("Failed to parse AI analysis", error=str(e))
                    return None
            
            return None
            
        except Exception as e:
            await self.logger.awarning("AI query analysis failed", error=str(e))
            return None
    
    async def _pattern_analyze_query(self, query_context: QueryContext) -> QueryAnalysis:
        """Pattern-based query analysis as fallback"""
        query = query_context.query.lower().strip()
        
        # Classify query type
        query_type = QueryType.UNKNOWN
        max_matches = 0
        
        for qtype, patterns in self.query_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
            if matches > max_matches:
                max_matches = matches
                query_type = qtype
        
        # Determine intent
        intent = QueryIntent.INFORMATION_SEEKING
        if any(word in query for word in ['should', 'recommend', 'advice', 'suggest']):
            intent = QueryIntent.DECISION_SUPPORT
        elif any(word in query for word in ['how', 'what', 'explain', 'learn']):
            intent = QueryIntent.LEARNING
        elif any(word in query for word in ['compare', 'vs', 'versus', 'better']):
            intent = QueryIntent.COMPARISON
        elif any(word in query for word in ['monitor', 'watch', 'track', 'alert']):
            intent = QueryIntent.MONITORING
        
        # Extract entities
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Extract topics
        topics = []
        topic_keywords = {
            'pools': ['pool', 'liquidity', 'pair'],
            'voting': ['vote', 'voting', 'veaero', 'gauge', 'epoch'],
            'tokens': ['token', 'price', 'aero', 'usdc', 'eth'],
            'trading': ['trade', 'swap', 'buy', 'sell'],
            'metrics': ['tvl', 'volume', 'fees', 'apr', 'metrics'],
            'bribes': ['bribe', 'rewards', 'incentive']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query for keyword in keywords):
                topics.append(topic)
        
        # Determine if real-time data is needed
        requires_real_time_data = any(word in query for word in [
            'current', 'now', 'today', 'latest', 'real-time', 'live', 'price'
        ])
        
        # Calculate confidence based on pattern matches
        confidence = min(0.9, max_matches * 0.2 + 0.3) if max_matches > 0 else 0.3
        
        # Calculate complexity score
        complexity_indicators = ['compare', 'analyze', 'strategy', 'complex', 'multiple']
        complexity_score = min(1.0, sum(
            0.2 for indicator in complexity_indicators if indicator in query
        ) + 0.3)
        
        # Suggest format
        suggested_format = ResponseFormat.CONVERSATIONAL
        if any(word in query for word in ['data', 'stats', 'metrics', 'numbers']):
            suggested_format = ResponseFormat.STRUCTURED
        elif any(word in query for word in ['technical', 'how', 'explain']):
            suggested_format = ResponseFormat.DETAILED
        
        return QueryAnalysis(
            query_type=query_type,
            intent=intent,
            entities=entities,
            topics=topics,
            confidence=confidence,
            requires_real_time_data=requires_real_time_data,
            complexity_score=complexity_score,
            suggested_format=suggested_format
        )


class ResponseGenerator:
    """Generates formatted responses from knowledge and analysis"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client
        self.logger = structlog.get_logger(__name__)
    
    async def generate_response(
        self,
        query_context: QueryContext,
        analysis: QueryAnalysis,
        knowledge_response: KnowledgeResponse,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a formatted response based on query analysis and knowledge"""
        try:
            # Use AI for response generation if available
            if self.gemini_client and knowledge_response.items:
                return await self._ai_generate_response(
                    query_context, analysis, knowledge_response, additional_context
                )
            
            # Fallback to template-based response
            return await self._template_generate_response(
                query_context, analysis, knowledge_response, additional_context
            )
            
        except Exception as e:
            await self.logger.aerror("Error generating response", error=str(e))
            return "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
    
    async def _ai_generate_response(
        self,
        query_context: QueryContext,
        analysis: QueryAnalysis,
        knowledge_response: KnowledgeResponse,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using AI"""
        try:
            # Prepare knowledge data for AI
            knowledge_summary = []
            for item in knowledge_response.items[:5]:  # Top 5 most relevant
                knowledge_summary.append({
                    "title": item.title,
                    "content": str(item.content)[:500],  # Truncate long content
                    "confidence": item.confidence,
                    "freshness": item.freshness.value,
                    "source": item.source
                })
            
            response_prompt = f"""
            Generate a helpful response to this Aerodrome DEX query:
            
            Query: "{query_context.query}"
            Query Type: {analysis.query_type.value}
            Intent: {analysis.intent.value}
            Format: {analysis.suggested_format.value}
            
            Relevant Knowledge:
            {json.dumps(knowledge_summary, indent=2)}
            
            Additional Context:
            {json.dumps(additional_context or {}, indent=2)}
            
            Guidelines:
            1. Be helpful, accurate, and informative
            2. Use the provided knowledge to answer the query
            3. If data is stale, mention the timestamp
            4. For trading advice, always include risk disclaimers
            5. Format according to the suggested format style
            6. Include confidence indicators when appropriate
            7. Be conversational but professional
            8. If knowledge is insufficient, say so honestly
            
            For numerical data:
            - Format large numbers with appropriate units (K, M, B)
            - Show percentages to 2 decimal places
            - Include timestamps for time-sensitive data
            
            Response should be 2-4 paragraphs unless more detail is specifically needed.
            """
            
            response = await self.gemini_client.generate_content(response_prompt)
            
            if response and response.content:
                return response.content.strip()
            
            return "I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            await self.logger.aerror("AI response generation failed", error=str(e))
            return await self._template_generate_response(
                query_context, analysis, knowledge_response, additional_context
            )
    
    async def _template_generate_response(
        self,
        query_context: QueryContext,
        analysis: QueryAnalysis,
        knowledge_response: KnowledgeResponse,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using templates"""
        if not knowledge_response.items:
            return self._generate_no_data_response(analysis)
        
        # Select template based on query type
        if analysis.query_type == QueryType.POOL_INQUIRY:
            return self._generate_pool_response(knowledge_response)
        elif analysis.query_type == QueryType.PROTOCOL_STATUS:
            return self._generate_protocol_status_response(knowledge_response)
        elif analysis.query_type == QueryType.TOKEN_INQUIRY:
            return self._generate_token_response(knowledge_response)
        elif analysis.query_type == QueryType.EDUCATIONAL:
            return self._generate_educational_response(knowledge_response)
        else:
            return self._generate_general_response(knowledge_response)
    
    def _generate_no_data_response(self, analysis: QueryAnalysis) -> str:
        """Generate response when no relevant data is found"""
        if analysis.query_type == QueryType.POOL_INQUIRY:
            return "I don't have current information about the specific pool you're asking about. This could be because it's a new pool, has low activity, or the address might be incorrect. Could you provide more details or check the pool address?"
        elif analysis.query_type == QueryType.TRADING_ADVICE:
            return "I don't have enough current market data to provide trading advice. For the most up-to-date information, please check recent pool performance, liquidity levels, and market conditions on the Aerodrome interface directly."
        else:
            return "I don't have specific information about your query right now. Could you try rephrasing your question or asking about a different aspect of Aerodrome?"
    
    def _generate_pool_response(self, knowledge_response: KnowledgeResponse) -> str:
        """Generate pool-specific response"""
        pool_items = [item for item in knowledge_response.items if item.knowledge_type.value == "pool_data"]
        
        if not pool_items:
            return "I don't have current pool data available."
        
        response_parts = []
        for item in pool_items[:3]:  # Top 3 pools
            content = item.content
            if isinstance(content, dict):
                tvl = content.get('tvl_usd', 0)
                volume = content.get('volume_24h', 0)
                apr = content.get('apr', 0)
                
                response_parts.append(
                    f"{item.title}: TVL ${tvl:,.0f}, 24h Volume ${volume:,.0f}, APR {apr:.2f}%"
                )
        
        if response_parts:
            return "Here's what I found:\n\n" + "\n\n".join(response_parts) + f"\n\nConfidence: {knowledge_response.query_confidence:.1%}"
        
        return "I found pool information but couldn't format it properly."
    
    def _generate_protocol_status_response(self, knowledge_response: KnowledgeResponse) -> str:
        """Generate protocol status response"""
        metrics_items = [
            item for item in knowledge_response.items 
            if item.knowledge_type.value == "protocol_metrics"
        ]
        
        if not metrics_items:
            return "I don't have current protocol metrics available."
        
        latest_metrics = metrics_items[0]
        content = latest_metrics.content
        
        if isinstance(content, dict):
            tvl = content.get('total_tvl', 0)
            volume = content.get('total_volume_24h', 0)
            fees = content.get('total_fees_24h', 0)
            pools = content.get('active_pools', 0)
            
            return f"""Current Aerodrome Protocol Status:

📊 Total TVL: ${tvl:,.0f}
💹 24h Volume: ${volume:,.0f}
💰 24h Fees: ${fees:,.0f}
🔗 Active Pools: {pools}

Data confidence: {latest_metrics.confidence:.1%} | Last updated: {latest_metrics.freshness.value}"""
        
        return "I found protocol data but couldn't format it properly."
    
    def _generate_token_response(self, knowledge_response: KnowledgeResponse) -> str:
        """Generate token-specific response"""
        return "Token information is available but I need to implement token-specific formatting."
    
    def _generate_educational_response(self, knowledge_response: KnowledgeResponse) -> str:
        """Generate educational response"""
        return "I can help explain Aerodrome concepts, but I need to implement educational content formatting."
    
    def _generate_general_response(self, knowledge_response: KnowledgeResponse) -> str:
        """Generate general response"""
        if knowledge_response.items:
            item = knowledge_response.items[0]
            return f"I found information about {item.title}. {str(item.content)[:200]}..."
        return "I found some information but couldn't format it properly."


class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
    
    def add_exchange(
        self,
        user_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a query-response exchange to conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }
        
        self.conversations[user_id].append(exchange)
        
        # Keep only recent history
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return self.conversations.get(user_id, [])
    
    def get_context_for_query(self, user_id: str) -> Dict[str, Any]:
        """Get relevant context for a new query"""
        history = self.get_conversation_history(user_id)
        if not history:
            return {}
        
        # Extract recent topics and entities
        recent_topics = set()
        recent_entities = set()
        
        for exchange in history[-3:]:  # Last 3 exchanges
            metadata = exchange.get("metadata", {})
            if "topics" in metadata:
                recent_topics.update(metadata["topics"])
            if "entities" in metadata:
                for entity_list in metadata["entities"].values():
                    recent_entities.update(entity_list)
        
        return {
            "recent_topics": list(recent_topics),
            "recent_entities": list(recent_entities),
            "conversation_length": len(history)
        }


class QueryHandler:
    """
    Main query handler that coordinates all query processing components
    """
    
    def __init__(
        self,
        knowledge_base: ProtocolKnowledgeBase,
        gemini_client: Optional[GeminiClient] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None
    ):
        """
        Initialize the query handler.
        
        Args:
            knowledge_base: Protocol knowledge base
            gemini_client: Optional AI client for advanced processing
            confidence_scorer: Optional confidence scoring system
        """
        self.knowledge_base = knowledge_base
        self.gemini_client = gemini_client
        self.confidence_scorer = confidence_scorer
        self.logger = structlog.get_logger(__name__)
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer(gemini_client)
        self.response_generator = ResponseGenerator(gemini_client)
        self.conversation_manager = ConversationManager()
        
        # Performance tracking
        self.total_queries = 0
        self.successful_queries = 0
        self.avg_processing_time = 0.0
    
    async def process_query(self, query_context: QueryContext) -> QueryResponse:
        """
        Process a complete query through the system.
        
        Args:
            query_context: Query context with user input and metadata
            
        Returns:
            QueryResponse with formatted answer and metadata
        """
        start_time = time.time()
        query_id = f"q_{int(time.time())}_{hash(query_context.query) % 10000}"
        
        try:
            self.total_queries += 1
            
            await self.logger.ainfo(
                "Processing query",
                query_id=query_id,
                query=query_context.query[:100],
                user_id=query_context.user_id
            )
            
            # Add conversation context
            if query_context.user_id:
                conversation_context = self.conversation_manager.get_context_for_query(
                    query_context.user_id
                )
                query_context.context.update(conversation_context)
            
            # Step 1: Analyze the query
            analysis = await self.query_analyzer.analyze_query(query_context)
            
            await self.logger.adebug(
                "Query analyzed",
                query_id=query_id,
                query_type=analysis.query_type.value,
                intent=analysis.intent.value,
                confidence=analysis.confidence
            )
            
            # Step 2: Convert analysis to knowledge query
            knowledge_query = self._create_knowledge_query(query_context, analysis)
            
            # Step 3: Query the knowledge base
            knowledge_response = await self.knowledge_base.query_knowledge(knowledge_query)
            
            await self.logger.adebug(
                "Knowledge retrieved",
                query_id=query_id,
                items_found=len(knowledge_response.items),
                knowledge_confidence=knowledge_response.query_confidence
            )
            
            # Step 4: Generate formatted response
            formatted_response = await self.response_generator.generate_response(
                query_context,
                analysis,
                knowledge_response,
                additional_context={
                    "query_id": query_id,
                    "processing_stats": {
                        "total_queries": self.total_queries,
                        "success_rate": self.successful_queries / max(self.total_queries, 1)
                    }
                }
            )
            
            # Step 5: Calculate overall confidence
            overall_confidence = self._calculate_response_confidence(
                analysis, knowledge_response
            )
            
            # Step 6: Create response object
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                query_id=query_id,
                response=formatted_response,
                confidence=overall_confidence,
                sources=self._extract_sources(knowledge_response),
                metadata={
                    "query_type": analysis.query_type.value,
                    "intent": analysis.intent.value,
                    "entities": analysis.entities,
                    "topics": analysis.topics,
                    "requires_real_time": analysis.requires_real_time_data,
                    "knowledge_items": len(knowledge_response.items),
                    "analysis_confidence": analysis.confidence,
                    "knowledge_confidence": knowledge_response.query_confidence
                },
                suggestions=knowledge_response.related_suggestions,
                format=analysis.suggested_format,
                processing_time=processing_time
            )
            
            # Update conversation history
            if query_context.user_id:
                self.conversation_manager.add_exchange(
                    query_context.user_id,
                    query_context.query,
                    formatted_response,
                    response.metadata
                )
            
            # Update performance metrics
            self.successful_queries += 1
            self._update_avg_processing_time(processing_time)
            
            await self.logger.ainfo(
                "Query processed successfully",
                query_id=query_id,
                processing_time=processing_time,
                confidence=overall_confidence
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            await self.logger.aerror(
                "Query processing failed",
                query_id=query_id,
                error=str(e),
                processing_time=processing_time
            )
            
            # Return error response
            return QueryResponse(
                query_id=query_id,
                response="I apologize, but I encountered an error processing your query. Please try again or rephrase your question.",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "processing_time": processing_time
                },
                processing_time=processing_time
            )
    
    def _create_knowledge_query(
        self, 
        query_context: QueryContext, 
        analysis: QueryAnalysis
    ) -> KnowledgeQuery:
        """Convert query analysis into knowledge base query"""
        
        # Map query type to knowledge types
        knowledge_type_mapping = {
            QueryType.POOL_INQUIRY: [KnowledgeType.POOL_DATA, KnowledgeType.MARKET_DATA],
            QueryType.TOKEN_INQUIRY: [KnowledgeType.TOKEN_DATA, KnowledgeType.MARKET_DATA],
            QueryType.PROTOCOL_STATUS: [KnowledgeType.PROTOCOL_METRICS],
            QueryType.VOTING_INQUIRY: [KnowledgeType.VOTING_DATA],
            QueryType.MARKET_ANALYSIS: [KnowledgeType.MARKET_DATA, KnowledgeType.HISTORICAL_ANALYSIS],
            QueryType.TRADING_ADVICE: [KnowledgeType.POOL_DATA, KnowledgeType.MARKET_DATA, KnowledgeType.PATTERN_INSIGHT],
            QueryType.HISTORICAL: [KnowledgeType.HISTORICAL_ANALYSIS],
            QueryType.PREDICTION: [KnowledgeType.PREDICTION, KnowledgeType.PATTERN_INSIGHT],
            QueryType.EDUCATIONAL: [KnowledgeType.PROTOCOL_METRICS, KnowledgeType.POOL_DATA]
        }
        
        knowledge_types = knowledge_type_mapping.get(analysis.query_type, [])
        
        # Set time range for historical queries
        time_range = None
        if analysis.query_type == QueryType.HISTORICAL:
            # Look for time indicators in entities
            time_entities = analysis.entities.get('time_period', [])
            if time_entities:
                # Parse time period (simplified)
                period = time_entities[0]
                if '24h' in period:
                    time_range = (datetime.now() - timedelta(days=1), datetime.now())
                elif '7d' in period:
                    time_range = (datetime.now() - timedelta(days=7), datetime.now())
                elif '30d' in period:
                    time_range = (datetime.now() - timedelta(days=30), datetime.now())
        
        # Adjust confidence threshold based on query type
        confidence_threshold = 0.3
        if analysis.query_type == QueryType.TRADING_ADVICE:
            confidence_threshold = 0.6  # Higher confidence for trading advice
        elif analysis.query_type == QueryType.PROTOCOL_STATUS:
            confidence_threshold = 0.7  # High confidence for protocol metrics
        
        # Adjust max results based on complexity
        max_results = min(10, max(3, int(analysis.complexity_score * 15)))
        
        return KnowledgeQuery(
            query_text=query_context.query,
            knowledge_types=knowledge_types,
            time_range=time_range,
            confidence_threshold=confidence_threshold,
            max_results=max_results,
            include_related=True,
            user_context=query_context.context
        )
    
    def _calculate_response_confidence(
        self,
        analysis: QueryAnalysis,
        knowledge_response: KnowledgeResponse
    ) -> float:
        """Calculate overall response confidence"""
        # Weight factors
        analysis_weight = 0.3
        knowledge_weight = 0.5
        freshness_weight = 0.2
        
        # Analysis confidence
        analysis_conf = analysis.confidence
        
        # Knowledge confidence
        knowledge_conf = knowledge_response.query_confidence
        
        # Freshness confidence (average freshness of top items)
        freshness_conf = 0.5
        if knowledge_response.items:
            freshness_scores = []
            for item in knowledge_response.items[:5]:
                if item.freshness.value == "real_time":
                    freshness_scores.append(1.0)
                elif item.freshness.value == "fresh":
                    freshness_scores.append(0.8)
                elif item.freshness.value == "recent":
                    freshness_scores.append(0.6)
                elif item.freshness.value == "stale":
                    freshness_scores.append(0.4)
                else:  # expired
                    freshness_scores.append(0.2)
            
            if freshness_scores:
                freshness_conf = sum(freshness_scores) / len(freshness_scores)
        
        # Calculate weighted confidence
        overall_confidence = (
            analysis_conf * analysis_weight +
            knowledge_conf * knowledge_weight +
            freshness_conf * freshness_weight
        )
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _extract_sources(self, knowledge_response: KnowledgeResponse) -> List[Dict[str, Any]]:
        """Extract source information from knowledge response"""
        sources = []
        
        for item in knowledge_response.items:
            source = {
                "title": item.title,
                "type": item.knowledge_type.value,
                "confidence": item.confidence,
                "freshness": item.freshness.value,
                "source": item.source,
                "timestamp": item.updated_at.isoformat()
            }
            sources.append(source)
        
        return sources
    
    def _update_avg_processing_time(self, processing_time: float) -> None:
        """Update average processing time metric"""
        if self.successful_queries == 1:
            self.avg_processing_time = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.avg_processing_time
            )
    
    async def get_handler_stats(self) -> Dict[str, Any]:
        """Get query handler performance statistics"""
        success_rate = self.successful_queries / max(self.total_queries, 1)
        
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": success_rate,
            "avg_processing_time": self.avg_processing_time,
            "active_conversations": len(self.conversation_manager.conversations),
            "components_status": {
                "analyzer": "ready",
                "generator": "ready",
                "knowledge_base": "ready" if self.knowledge_base else "unavailable",
                "ai_client": "ready" if self.gemini_client else "unavailable"
            }
        }


# Example usage
async def main():
    """Example usage of the Query Handler"""
    from .knowledge_base import ProtocolKnowledgeBase
    from .confidence_scorer import ConfidenceScorer
    from ..intelligence import GeminiClient
    from ..memory import EnhancedMem0Client
    from ..protocol import AerodromeClient
    
    # Initialize components (mock for example)
    memory_client = EnhancedMem0Client(api_key="test")
    aerodrome_client = AerodromeClient(quicknode_url="test")
    confidence_scorer = ConfidenceScorer()
    gemini_client = GeminiClient(api_key="test")
    
    knowledge_base = ProtocolKnowledgeBase(
        memory_client=memory_client,
        aerodrome_client=aerodrome_client,
        confidence_scorer=confidence_scorer,
        gemini_client=gemini_client
    )
    
    await knowledge_base.initialize()
    
    # Create query handler
    query_handler = QueryHandler(
        knowledge_base=knowledge_base,
        gemini_client=gemini_client,
        confidence_scorer=confidence_scorer
    )
    
    # Test queries
    test_queries = [
        "What are the top performing pools today?",
        "How does veAERO voting work?",
        "Should I provide liquidity to the USDC/ETH pool?",
        "What's the current protocol TVL?"
    ]
    
    for query in test_queries:
        query_context = QueryContext(
            query=query,
            user_id="test_user",
            context={"user_level": "intermediate"}
        )
        
        response = await query_handler.process_query(query_context)
        
        print(f"Query: {query}")
        print(f"Response: {response.response[:200]}...")
        print(f"Confidence: {response.confidence:.2f}")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())