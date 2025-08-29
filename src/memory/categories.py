"""
Memory categorization logic and category definitions.

This module defines memory categories and provides categorization logic
to classify memories based on their content and context.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CategoryDefinition:
    """Definition of a memory category"""
    
    name: str
    description: str
    retention_days: int  # -1 for permanent
    importance: float  # 0.0 to 1.0
    auto_archive: bool = True
    pattern_eligible: bool = True
    compression_level: str = 'light'  # 'none', 'light', 'heavy', 'pattern_only'


class MemoryCategories:
    """Define and manage memory categories"""
    
    def __init__(self):
        self.categories = self._initialize_categories()
        self.category_keywords = self._build_keyword_mappings()
    
    def _initialize_categories(self) -> Dict[str, CategoryDefinition]:
        """Initialize category definitions"""
        
        return {
            'trades': CategoryDefinition(
                name='trades',
                description='Trade executions and results',
                retention_days=90,  # Extended for trading data
                importance=0.9,
                auto_archive=True,
                pattern_eligible=True,
                compression_level='light'
            ),
            
            'market_observations': CategoryDefinition(
                name='market_observations',
                description='Market state snapshots and observations',
                retention_days=14,
                importance=0.6,
                auto_archive=True,
                pattern_eligible=True,
                compression_level='heavy'
            ),
            
            'patterns': CategoryDefinition(
                name='patterns',
                description='Learned patterns and strategies',
                retention_days=-1,  # Never delete
                importance=1.0,
                auto_archive=False,
                pattern_eligible=False,  # Patterns don't create patterns
                compression_level='none'
            ),
            
            'failures': CategoryDefinition(
                name='failures',
                description='Failed actions and errors',
                retention_days=60,
                importance=0.8,
                auto_archive=True,
                pattern_eligible=True,
                compression_level='light'
            ),
            
            'opportunities': CategoryDefinition(
                name='opportunities',
                description='Identified trading opportunities',
                retention_days=7,  # Short-lived by nature
                importance=0.5,
                auto_archive=True,
                pattern_eligible=True,
                compression_level='heavy'
            ),
            
            'user_preferences': CategoryDefinition(
                name='user_preferences',
                description='User settings and preferences',
                retention_days=-1,  # Never delete
                importance=1.0,
                auto_archive=False,
                pattern_eligible=False,
                compression_level='none'
            ),
            
            'system_events': CategoryDefinition(
                name='system_events',
                description='System status and events',
                retention_days=30,
                importance=0.4,
                auto_archive=True,
                pattern_eligible=False,
                compression_level='heavy'
            ),
            
            'wallet_operations': CategoryDefinition(
                name='wallet_operations',
                description='Wallet interactions and operations',
                retention_days=30,
                importance=0.7,
                auto_archive=True,
                pattern_eligible=True,
                compression_level='light'
            ),
            
            'pool_analysis': CategoryDefinition(
                name='pool_analysis',
                description='Liquidity pool analysis and metrics',
                retention_days=21,
                importance=0.7,
                auto_archive=True,
                pattern_eligible=True,
                compression_level='light'
            ),
            
            'strategy_execution': CategoryDefinition(
                name='strategy_execution',
                description='Strategy execution and performance',
                retention_days=60,
                importance=0.9,
                auto_archive=True,
                pattern_eligible=True,
                compression_level='light'
            )
        }
    
    def _build_keyword_mappings(self) -> Dict[str, List[str]]:
        """Build keyword mappings for category detection"""
        
        return {
            'trades': [
                'trade', 'swap', 'buy', 'sell', 'exchange',
                'liquidity', 'add_liquidity', 'remove_liquidity',
                'position', 'order', 'execution'
            ],
            
            'market_observations': [
                'observation', 'market', 'price', 'volume',
                'volatility', 'trend', 'analysis', 'snapshot',
                'data', 'metrics'
            ],
            
            'patterns': [
                'pattern', 'strategy', 'behavior', 'trend',
                'correlation', 'signal', 'indicator'
            ],
            
            'failures': [
                'error', 'failed', 'failure', 'exception',
                'timeout', 'rejected', 'revert', 'slippage',
                'insufficient', 'invalid'
            ],
            
            'opportunities': [
                'opportunity', 'arbitrage', 'profit', 'signal',
                'alert', 'target', 'threshold', 'trigger'
            ],
            
            'user_preferences': [
                'preference', 'setting', 'config', 'configuration',
                'option', 'choice', 'parameter', 'limit',
                'threshold', 'target'
            ],
            
            'system_events': [
                'system', 'startup', 'shutdown', 'health',
                'status', 'connection', 'sync', 'update',
                'maintenance', 'backup'
            ],
            
            'wallet_operations': [
                'wallet', 'balance', 'transfer', 'approve',
                'allowance', 'gas', 'transaction', 'address',
                'account', 'funds'
            ],
            
            'pool_analysis': [
                'pool', 'reserve', 'tvl', 'fee', 'apr',
                'yield', 'impermanent', 'ratio', 'pair',
                'composition'
            ],
            
            'strategy_execution': [
                'strategy', 'execute', 'performance', 'backtest',
                'optimization', 'parameter', 'result', 'outcome',
                'effectiveness', 'roi'
            ]
        }
    
    def categorize_memory(self, content: Dict[str, Any]) -> str:
        """Categorize memory based on content"""
        
        # Direct type mapping
        memory_type = content.get('type', '').lower()
        direct_mappings = {
            'trade': 'trades',
            'swap': 'trades',
            'liquidity': 'trades',
            'add_liquidity': 'trades',
            'remove_liquidity': 'trades',
            'observation': 'market_observations',
            'market_observation': 'market_observations',
            'pattern': 'patterns',
            'error': 'failures',
            'failure': 'failures',
            'exception': 'failures',
            'opportunity': 'opportunities',
            'arbitrage': 'opportunities',
            'preference': 'user_preferences',
            'setting': 'user_preferences',
            'system': 'system_events',
            'wallet': 'wallet_operations',
            'pool': 'pool_analysis',
            'strategy': 'strategy_execution'
        }
        
        # Check direct mappings first
        for key, category in direct_mappings.items():
            if key in memory_type:
                return category
        
        # Content-based categorization using keywords
        category_scores = {}
        
        # Extract all text content for analysis
        text_content = self._extract_text_content(content)
        text_lower = text_content.lower()
        
        # Score each category based on keyword matches
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    
                    # Give higher weight to exact matches in type or action
                    if keyword in memory_type or keyword == content.get('action', '').lower():
                        score += 2
            
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Context-based fallback categorization
        return self._contextual_categorization(content)
    
    def _extract_text_content(self, content: Dict[str, Any]) -> str:
        """Extract all text content from memory for analysis"""
        
        text_parts = []
        
        # Add basic fields
        for field in ['type', 'action', 'description', 'message', 'error_message']:
            value = content.get(field, '')
            if isinstance(value, str) and value:
                text_parts.append(value)
        
        # Add pool information
        pool = content.get('pool', '')
        if pool:
            text_parts.append(f"pool {pool}")
        
        # Add tokens
        tokens = content.get('tokens', [])
        if isinstance(tokens, list):
            for token in tokens:
                if isinstance(token, str):
                    text_parts.append(f"token {token}")
        
        # Add market conditions
        market_conditions = content.get('market_conditions', {})
        if isinstance(market_conditions, dict):
            for key, value in market_conditions.items():
                if isinstance(value, str):
                    text_parts.append(f"{key} {value}")
        
        # Add searchable text if available
        searchable = content.get('searchable_text', '')
        if searchable:
            text_parts.append(searchable)
        
        return ' '.join(text_parts)
    
    def _contextual_categorization(self, content: Dict[str, Any]) -> str:
        """Contextual categorization based on content structure"""
        
        # Check for trading indicators
        if any(key in content for key in ['profit', 'gas_used', 'tx_hash', 'amounts']):
            return 'trades'
        
        # Check for market data
        if any(key in content for key in ['price', 'volume', 'volatility', 'reserves']):
            return 'market_observations'
        
        # Check for error indicators
        if any(key in content for key in ['error', 'failed', 'exception']) or content.get('success') is False:
            return 'failures'
        
        # Check for system indicators
        if any(key in content for key in ['status', 'health', 'connection', 'sync']):
            return 'system_events'
        
        # Check for wallet indicators
        if any(key in content for key in ['balance', 'address', 'allowance', 'approval']):
            return 'wallet_operations'
        
        # Default to market observations
        return 'market_observations'
    
    def get_category_definition(self, category: str) -> Optional[CategoryDefinition]:
        """Get category definition"""
        return self.categories.get(category)
    
    def get_retention_period(self, category: str) -> int:
        """Get retention period for category in days"""
        category_def = self.categories.get(category)
        return category_def.retention_days if category_def else 30
    
    def get_importance(self, category: str) -> float:
        """Get importance score for category"""
        category_def = self.categories.get(category)
        return category_def.importance if category_def else 0.5
    
    def get_compression_level(self, category: str) -> str:
        """Get compression level for category"""
        category_def = self.categories.get(category)
        return category_def.compression_level if category_def else 'light'
    
    def should_auto_archive(self, category: str) -> bool:
        """Check if category should auto-archive"""
        category_def = self.categories.get(category)
        return category_def.auto_archive if category_def else True
    
    def is_pattern_eligible(self, category: str) -> bool:
        """Check if category is eligible for pattern extraction"""
        category_def = self.categories.get(category)
        return category_def.pattern_eligible if category_def else True
    
    def get_all_categories(self) -> List[str]:
        """Get list of all category names"""
        return list(self.categories.keys())
    
    def get_category_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all categories"""
        
        stats = {}
        
        for category_name, category_def in self.categories.items():
            stats[category_name] = {
                'description': category_def.description,
                'retention_days': category_def.retention_days,
                'importance': category_def.importance,
                'auto_archive': category_def.auto_archive,
                'pattern_eligible': category_def.pattern_eligible,
                'compression_level': category_def.compression_level,
                'keywords_count': len(self.category_keywords.get(category_name, []))
            }
        
        return stats
    
    def suggest_category_improvements(self, content_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content samples and suggest category improvements"""
        
        suggestions = {
            'uncategorized_content': [],
            'keyword_suggestions': {},
            'new_category_suggestions': []
        }
        
        # Track categorization results
        categorization_results = {}
        
        for sample in content_samples:
            category = self.categorize_memory(sample)
            
            if category not in categorization_results:
                categorization_results[category] = 0
            categorization_results[category] += 1
            
            # Check if categorization seems uncertain
            text_content = self._extract_text_content(sample)
            if len(text_content.split()) < 3:  # Very short content
                suggestions['uncategorized_content'].append({
                    'content': sample,
                    'assigned_category': category,
                    'reason': 'insufficient_content'
                })
        
        # Identify potential new categories from uncategorized content
        if categorization_results.get('market_observations', 0) > len(content_samples) * 0.3:
            suggestions['new_category_suggestions'].append({
                'suggested_name': 'miscellaneous',
                'reason': 'high_default_categorization',
                'percentage': categorization_results['market_observations'] / len(content_samples) * 100
            })
        
        return suggestions


class SpecializedMemoryTypes:
    """Specialized memory type handlers"""
    
    @staticmethod
    def create_trade_memory(
        action: str,
        pool: str,
        tokens: List[str],
        amounts: List[float],
        success: bool,
        profit: float = 0.0,
        gas_used: float = 0.0,
        tx_hash: str = None,
        market_conditions: Dict = None,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Create structured trade memory"""
        
        return {
            'type': 'trade',
            'action': action,
            'pool': pool,
            'tokens': tokens,
            'amounts': amounts,
            'success': success,
            'profit': profit,
            'gas_used': gas_used,
            'tx_hash': tx_hash,
            'market_conditions': market_conditions or {},
            'decision_confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'searchable_text': f"{action} {' '.join(tokens)} pool {pool} {'successful' if success else 'failed'}"
        }
    
    @staticmethod
    def create_pattern_memory(
        pattern_type: str,
        conditions: Dict,
        action_sequence: List[str],
        success_rate: float,
        occurrence_count: int,
        avg_profit: float = 0.0,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Create structured pattern memory"""
        
        return {
            'type': 'pattern',
            'pattern_type': pattern_type,
            'conditions': conditions,
            'action_sequence': action_sequence,
            'success_rate': success_rate,
            'occurrence_count': occurrence_count,
            'avg_profit': avg_profit,
            'confidence': confidence,
            'discovered_at': datetime.now().isoformat(),
            'importance': success_rate * confidence,
            'searchable_text': f"pattern {pattern_type} {' '.join(action_sequence)}"
        }
    
    @staticmethod
    def create_market_observation(
        pool: str,
        observation: str,
        price: float = None,
        volume: float = None,
        volatility: float = None,
        conditions: Dict = None
    ) -> Dict[str, Any]:
        """Create structured market observation memory"""
        
        return {
            'type': 'market_observation',
            'pool': pool,
            'observation': observation,
            'price': price,
            'volume': volume,
            'volatility': volatility,
            'conditions': conditions or {},
            'timestamp': datetime.now().isoformat(),
            'searchable_text': f"market observation {pool} {observation}"
        }
    
    @staticmethod
    def create_failure_memory(
        action: str,
        error_type: str,
        error_message: str,
        context: Dict = None
    ) -> Dict[str, Any]:
        """Create structured failure memory"""
        
        return {
            'type': 'failure',
            'action': action,
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
            'searchable_text': f"failure {action} {error_type} {error_message}"
        }
    
    @staticmethod
    def create_opportunity_memory(
        opportunity_type: str,
        description: str,
        pool: str = None,
        potential_profit: float = None,
        confidence: float = 0.5,
        expires_at: datetime = None
    ) -> Dict[str, Any]:
        """Create structured opportunity memory"""
        
        return {
            'type': 'opportunity',
            'opportunity_type': opportunity_type,
            'description': description,
            'pool': pool,
            'potential_profit': potential_profit,
            'confidence': confidence,
            'expires_at': expires_at.isoformat() if expires_at else None,
            'timestamp': datetime.now().isoformat(),
            'searchable_text': f"opportunity {opportunity_type} {description} {pool or ''}"
        }