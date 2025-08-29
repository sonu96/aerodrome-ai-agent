"""
Memory recall node for the Aerodrome Brain.

This node retrieves relevant memories, patterns, and historical experiences
to inform current decision-making. It implements sophisticated memory retrieval
strategies including semantic search, temporal filtering, and pattern matching.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..state import BrainState, BrainConfig, MemoryEntry


class RecallNode:
    """
    Memory recall node that retrieves relevant experiences and patterns.
    
    This node is responsible for:
    - Retrieving relevant historical experiences
    - Identifying applicable patterns from past actions
    - Learning from failure patterns to avoid repetition
    - Contextualizing memories based on current market conditions
    """

    def __init__(self, memory_system, config: BrainConfig):
        """
        Initialize the recall node.
        
        Args:
            memory_system: Memory system for storing/retrieving experiences
            config: Brain configuration parameters
        """
        self.memory_system = memory_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Memory retrieval parameters
        self.similarity_threshold = 0.7
        self.max_age_days = config.memory_retention_days
        self.pattern_confidence_threshold = 0.6

    async def recall_memories(self, state: BrainState) -> BrainState:
        """
        Recall relevant memories and patterns based on current state.
        
        Args:
            state: Current brain state
            
        Returns:
            Updated state with recalled memories and patterns
        """
        
        self.logger.info("Starting memory recall process")
        
        try:
            # Extract context for memory retrieval
            context = self._extract_context(state)
            
            # Retrieve different types of memories in parallel
            tasks = [
                self._retrieve_similar_experiences(context),
                self._retrieve_successful_patterns(context),
                self._retrieve_failure_patterns(context), 
                self._retrieve_recent_experiences(context),
                self._retrieve_seasonal_patterns(context)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            (similar_experiences, successful_patterns, failure_patterns, 
             recent_experiences, seasonal_patterns) = results
            
            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in memory recall task {i}: {result}")
            
            # Process and filter memories
            relevant_memories = self._process_retrieved_memories(
                similar_experiences, recent_experiences, context
            )
            
            learned_patterns = self._process_learned_patterns(
                successful_patterns, seasonal_patterns, context
            )
            
            failure_patterns = self._process_failure_patterns(
                failure_patterns, context
            )
            
            # Calculate memory-based insights
            insights = self._generate_memory_insights(
                relevant_memories, learned_patterns, failure_patterns
            )
            
            # Update state with recalled information
            updated_state = {
                **state,
                'relevant_memories': relevant_memories,
                'learned_patterns': learned_patterns,
                'failure_patterns': failure_patterns,
                'recent_experiences': recent_experiences if not isinstance(recent_experiences, Exception) else [],
                'memory_insights': insights,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Recalled {len(relevant_memories)} memories, {len(learned_patterns)} patterns"
                ]
            }
            
            self.logger.info(
                f"Memory recall completed - {len(relevant_memories)} memories, "
                f"{len(learned_patterns)} patterns retrieved"
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error in memory recall: {e}")
            
            # Return state with empty memory context on error
            return {
                **state,
                'relevant_memories': [],
                'learned_patterns': [],
                'failure_patterns': [],
                'recent_experiences': [],
                'warnings': state.get('warnings', []) + [f"Memory recall failed: {str(e)}"]
            }

    def _extract_context(self, state: BrainState) -> Dict[str, Any]:
        """Extract relevant context for memory retrieval."""
        
        context = {
            'timestamp': state.get('timestamp', datetime.now()),
            'market_conditions': self._extract_market_context(state),
            'portfolio_state': self._extract_portfolio_context(state),
            'network_conditions': self._extract_network_context(state),
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'cycle_count': state.get('cycle_count', 0)
        }
        
        return context

    def _extract_market_context(self, state: BrainState) -> Dict[str, Any]:
        """Extract market-related context for memory matching."""
        
        market_data = state.get('market_data', {})
        
        return {
            'total_tvl': state.get('total_value_locked', 0),
            'gas_price': state.get('gas_price', 0),
            'network_congestion': state.get('network_congestion', 0),
            'pool_count': len(market_data.get('pools', {})),
            'average_apr': market_data.get('average_apr', 0),
            'market_sentiment': state.get('market_sentiment', {}),
            'base_fee_trend': state.get('base_fee_trend', 'stable')
        }

    def _extract_portfolio_context(self, state: BrainState) -> Dict[str, Any]:
        """Extract portfolio-related context."""
        
        portfolio_perf = state.get('portfolio_performance', {})
        
        return {
            'total_value': portfolio_perf.get('total_value', 0),
            'position_count': portfolio_perf.get('position_count', 0),
            'wallet_balance_count': len(state.get('wallet_balance', {})),
            'pending_tx_count': len(state.get('pending_transactions', []))
        }

    def _extract_network_context(self, state: BrainState) -> Dict[str, Any]:
        """Extract network-related context."""
        
        return {
            'gas_price': state.get('gas_price', 0),
            'congestion': state.get('network_congestion', 0),
            'base_fee_trend': state.get('base_fee_trend', 'stable')
        }

    async def _retrieve_similar_experiences(self, context: Dict[str, Any]) -> List[MemoryEntry]:
        """Retrieve experiences similar to current context."""
        
        if not self.memory_system:
            return []
        
        try:
            # Query memory system for similar experiences
            query_params = {
                'memory_type': 'experience',
                'similarity_threshold': self.similarity_threshold,
                'max_results': self.config.max_memories_per_query,
                'context': context
            }
            
            memories = await self.memory_system.query_memories(query_params)
            
            # Filter by relevance and recency
            filtered_memories = self._filter_memories_by_relevance(memories, context)
            
            return filtered_memories[:self.config.max_memories_per_query]
            
        except Exception as e:
            self.logger.error(f"Error retrieving similar experiences: {e}")
            return []

    async def _retrieve_successful_patterns(self, context: Dict[str, Any]) -> List[MemoryEntry]:
        """Retrieve patterns that led to successful outcomes."""
        
        if not self.memory_system:
            return []
        
        try:
            query_params = {
                'memory_type': 'pattern',
                'outcome_filter': 'successful',
                'confidence_threshold': self.pattern_confidence_threshold,
                'max_results': self.config.max_memories_per_query,
                'context': context
            }
            
            patterns = await self.memory_system.query_memories(query_params)
            
            # Score patterns by applicability to current context
            scored_patterns = self._score_pattern_applicability(patterns, context)
            
            return scored_patterns[:self.config.max_memories_per_query // 2]
            
        except Exception as e:
            self.logger.error(f"Error retrieving successful patterns: {e}")
            return []

    async def _retrieve_failure_patterns(self, context: Dict[str, Any]) -> List[MemoryEntry]:
        """Retrieve patterns that led to failures (to avoid)."""
        
        if not self.memory_system:
            return []
        
        try:
            query_params = {
                'memory_type': 'failure',
                'max_results': self.config.max_memories_per_query // 2,
                'context': context,
                'recency_weight': 2.0  # Emphasize recent failures
            }
            
            failures = await self.memory_system.query_memories(query_params)
            
            # Filter highly relevant failures
            relevant_failures = [
                f for f in failures 
                if self._calculate_context_similarity(f['market_conditions'], context['market_conditions']) > 0.6
            ]
            
            return relevant_failures
            
        except Exception as e:
            self.logger.error(f"Error retrieving failure patterns: {e}")
            return []

    async def _retrieve_recent_experiences(self, context: Dict[str, Any]) -> List[MemoryEntry]:
        """Retrieve recent experiences for trend analysis."""
        
        if not self.memory_system:
            return []
        
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Last week
            
            query_params = {
                'memory_type': 'experience',
                'time_filter': {'after': cutoff_time},
                'max_results': 20,  # More recent experiences
                'sort_by': 'timestamp',
                'sort_order': 'desc'
            }
            
            recent_memories = await self.memory_system.query_memories(query_params)
            
            return recent_memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent experiences: {e}")
            return []

    async def _retrieve_seasonal_patterns(self, context: Dict[str, Any]) -> List[MemoryEntry]:
        """Retrieve seasonal or time-based patterns."""
        
        if not self.memory_system:
            return []
        
        try:
            current_hour = context['time_of_day']
            current_dow = context['day_of_week']
            
            query_params = {
                'memory_type': 'pattern',
                'temporal_filter': {
                    'hour_range': (current_hour - 2, current_hour + 2),
                    'day_of_week': current_dow
                },
                'max_results': 10
            }
            
            seasonal_patterns = await self.memory_system.query_memories(query_params)
            
            return seasonal_patterns
            
        except Exception as e:
            self.logger.error(f"Error retrieving seasonal patterns: {e}")
            return []

    def _process_retrieved_memories(
        self, 
        similar_experiences: List[MemoryEntry],
        recent_experiences: List[MemoryEntry], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process and rank retrieved memories by relevance."""
        
        if isinstance(similar_experiences, Exception):
            similar_experiences = []
        if isinstance(recent_experiences, Exception):
            recent_experiences = []
        
        all_memories = similar_experiences + recent_experiences
        
        # Remove duplicates
        seen_ids = set()
        unique_memories = []
        
        for memory in all_memories:
            memory_id = (memory['timestamp'].isoformat(), memory['memory_type'])
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_memories.append(memory)
        
        # Score memories by relevance to current context
        scored_memories = []
        for memory in unique_memories:
            relevance_score = self._calculate_memory_relevance(memory, context)
            
            memory_dict = {
                **memory,
                'relevance_score': relevance_score,
                'context_similarity': self._calculate_context_similarity(
                    memory.get('market_conditions', {}), 
                    context.get('market_conditions', {})
                )
            }
            scored_memories.append(memory_dict)
        
        # Sort by relevance and return top memories
        scored_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_memories[:self.config.max_memories_per_query]

    def _process_learned_patterns(
        self, 
        successful_patterns: List[MemoryEntry],
        seasonal_patterns: List[MemoryEntry], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process and rank learned patterns."""
        
        if isinstance(successful_patterns, Exception):
            successful_patterns = []
        if isinstance(seasonal_patterns, Exception):
            seasonal_patterns = []
        
        all_patterns = successful_patterns + seasonal_patterns
        
        # Remove duplicates and score patterns
        pattern_scores = {}
        
        for pattern in all_patterns:
            pattern_key = pattern.get('pattern_features', {}).get('signature', str(hash(str(pattern))))
            
            if pattern_key not in pattern_scores:
                applicability_score = self._calculate_pattern_applicability(pattern, context)
                confidence_score = pattern.get('confidence_level', 0.5)
                
                combined_score = (applicability_score * 0.7) + (confidence_score * 0.3)
                
                pattern_scores[pattern_key] = {
                    **pattern,
                    'applicability_score': applicability_score,
                    'combined_score': combined_score
                }
        
        # Sort by combined score
        ranked_patterns = sorted(
            pattern_scores.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        return ranked_patterns[:self.config.max_memories_per_query // 2]

    def _process_failure_patterns(
        self, 
        failure_patterns: List[MemoryEntry], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process failure patterns to identify risks."""
        
        if isinstance(failure_patterns, Exception):
            failure_patterns = []
        
        processed_failures = []
        
        for failure in failure_patterns:
            risk_score = self._calculate_failure_risk_score(failure, context)
            
            if risk_score > 0.5:  # Only include significant risks
                failure_dict = {
                    **failure,
                    'risk_score': risk_score,
                    'warning_level': self._categorize_failure_risk(risk_score)
                }
                processed_failures.append(failure_dict)
        
        # Sort by risk score (highest first)
        processed_failures.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return processed_failures[:10]  # Limit to top 10 risks

    def _generate_memory_insights(
        self,
        relevant_memories: List[Dict[str, Any]],
        learned_patterns: List[Dict[str, Any]],
        failure_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate insights from retrieved memories and patterns."""
        
        insights = {
            'success_rate': self._calculate_historical_success_rate(relevant_memories),
            'common_success_factors': self._identify_success_factors(learned_patterns),
            'key_risk_factors': self._identify_risk_factors(failure_patterns),
            'recommended_actions': self._generate_action_recommendations(relevant_memories, learned_patterns),
            'warnings': self._generate_warnings(failure_patterns),
            'confidence_modifiers': self._calculate_confidence_modifiers(relevant_memories)
        }
        
        return insights

    # Helper methods for memory processing and scoring

    def _filter_memories_by_relevance(
        self, 
        memories: List[MemoryEntry], 
        context: Dict[str, Any]
    ) -> List[MemoryEntry]:
        """Filter memories by relevance to current context."""
        
        relevant_memories = []
        
        for memory in memories:
            # Check age
            age_days = (context['timestamp'] - memory['timestamp']).days
            if age_days > self.max_age_days:
                continue
            
            # Check context similarity
            similarity = self._calculate_context_similarity(
                memory.get('market_conditions', {}),
                context.get('market_conditions', {})
            )
            
            if similarity >= self.similarity_threshold:
                relevant_memories.append(memory)
        
        return relevant_memories

    def _score_pattern_applicability(
        self, 
        patterns: List[MemoryEntry], 
        context: Dict[str, Any]
    ) -> List[MemoryEntry]:
        """Score patterns by their applicability to current context."""
        
        scored_patterns = []
        
        for pattern in patterns:
            applicability = self._calculate_pattern_applicability(pattern, context)
            
            if applicability > 0.4:  # Minimum applicability threshold
                pattern_copy = pattern.copy()
                pattern_copy['applicability_score'] = applicability
                scored_patterns.append(pattern_copy)
        
        # Sort by applicability
        scored_patterns.sort(key=lambda x: x['applicability_score'], reverse=True)
        
        return scored_patterns

    def _calculate_memory_relevance(
        self, 
        memory: MemoryEntry, 
        context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for a memory."""
        
        # Time decay factor
        age_days = (context['timestamp'] - memory['timestamp']).days
        time_factor = max(0.1, 1.0 - (age_days / self.max_age_days))
        
        # Context similarity factor
        context_factor = self._calculate_context_similarity(
            memory.get('market_conditions', {}),
            context.get('market_conditions', {})
        )
        
        # Usage frequency factor (if available)
        usage_factor = min(1.0, memory.get('usage_count', 1) / 10.0)
        
        # Outcome quality factor
        outcome_factor = self._assess_outcome_quality(memory.get('outcome', {}))
        
        # Weighted combination
        relevance = (
            time_factor * 0.2 +
            context_factor * 0.4 +
            usage_factor * 0.2 +
            outcome_factor * 0.2
        )
        
        return relevance

    def _calculate_context_similarity(
        self, 
        memory_context: Dict[str, Any], 
        current_context: Dict[str, Any]
    ) -> float:
        """Calculate similarity between memory context and current context."""
        
        if not memory_context or not current_context:
            return 0.0
        
        # Compare key numerical features
        numerical_features = ['total_tvl', 'gas_price', 'network_congestion', 'average_apr']
        similarities = []
        
        for feature in numerical_features:
            if feature in memory_context and feature in current_context:
                mem_val = memory_context[feature]
                cur_val = current_context[feature]
                
                if mem_val == 0 and cur_val == 0:
                    sim = 1.0
                elif mem_val == 0 or cur_val == 0:
                    sim = 0.0
                else:
                    # Use relative similarity
                    ratio = min(mem_val, cur_val) / max(mem_val, cur_val)
                    sim = ratio
                
                similarities.append(sim)
        
        # Compare categorical features
        categorical_features = ['base_fee_trend']
        for feature in categorical_features:
            if (feature in memory_context and feature in current_context):
                if memory_context[feature] == current_context[feature]:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0

    def _calculate_pattern_applicability(
        self, 
        pattern: MemoryEntry, 
        context: Dict[str, Any]
    ) -> float:
        """Calculate how applicable a pattern is to current context."""
        
        pattern_features = pattern.get('pattern_features', {})
        
        # Market condition match
        market_match = self._calculate_context_similarity(
            pattern_features.get('market_conditions', {}),
            context.get('market_conditions', {})
        )
        
        # Temporal match (time of day, day of week)
        temporal_match = self._calculate_temporal_match(pattern_features, context)
        
        # Pattern confidence
        pattern_confidence = pattern.get('confidence_level', 0.5)
        
        # Success rate of this pattern
        success_rate = pattern.get('success_rate', 0.5)
        
        # Weighted applicability score
        applicability = (
            market_match * 0.4 +
            temporal_match * 0.2 +
            pattern_confidence * 0.2 +
            success_rate * 0.2
        )
        
        return applicability

    def _calculate_temporal_match(
        self, 
        pattern_features: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate temporal similarity between pattern and context."""
        
        # Hour of day match
        pattern_hour = pattern_features.get('hour_of_day')
        context_hour = context.get('time_of_day')
        
        hour_match = 0.5  # Default
        if pattern_hour is not None and context_hour is not None:
            hour_diff = abs(pattern_hour - context_hour)
            # Circular distance (handle wrap around at 24/0)
            hour_diff = min(hour_diff, 24 - hour_diff)
            hour_match = max(0, 1.0 - hour_diff / 12.0)  # Normalize to 0-1
        
        # Day of week match
        pattern_dow = pattern_features.get('day_of_week')
        context_dow = context.get('day_of_week')
        
        dow_match = 0.5  # Default
        if pattern_dow is not None and context_dow is not None:
            if pattern_dow == context_dow:
                dow_match = 1.0
            elif abs(pattern_dow - context_dow) == 1:  # Adjacent days
                dow_match = 0.7
            else:
                dow_match = 0.3
        
        return (hour_match + dow_match) / 2.0

    def _calculate_failure_risk_score(
        self, 
        failure: MemoryEntry, 
        context: Dict[str, Any]
    ) -> float:
        """Calculate risk score for a failure pattern."""
        
        # Context similarity (higher = more relevant risk)
        context_sim = self._calculate_context_similarity(
            failure.get('market_conditions', {}),
            context.get('market_conditions', {})
        )
        
        # Recency factor (more recent failures are more concerning)
        age_days = (context['timestamp'] - failure['timestamp']).days
        recency_factor = max(0.1, 1.0 - (age_days / 30.0))  # 30-day decay
        
        # Severity of the failure
        outcome = failure.get('outcome', {})
        severity = abs(outcome.get('net_value_change', 0)) / 1000.0  # Normalize loss
        severity = min(severity, 1.0)
        
        risk_score = (context_sim * 0.5) + (recency_factor * 0.3) + (severity * 0.2)
        
        return risk_score

    def _categorize_failure_risk(self, risk_score: float) -> str:
        """Categorize failure risk level."""
        
        if risk_score > 0.8:
            return "critical"
        elif risk_score > 0.6:
            return "high" 
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"

    def _calculate_historical_success_rate(self, memories: List[Dict[str, Any]]) -> float:
        """Calculate historical success rate from memories."""
        
        if not memories:
            return 0.5  # Default neutral success rate
        
        successes = 0
        total = len(memories)
        
        for memory in memories:
            outcome = memory.get('outcome', {})
            if outcome.get('success', False):
                successes += 1
        
        return successes / total if total > 0 else 0.5

    def _identify_success_factors(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Identify common success factors from patterns."""
        
        factors = defaultdict(int)
        
        for pattern in patterns:
            success_factors = pattern.get('success_factors', [])
            for factor in success_factors:
                factors[factor] += 1
        
        # Return most common factors
        return [factor for factor, count in factors.most_common(5)]

    def _identify_risk_factors(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Identify key risk factors from failure patterns."""
        
        risk_factors = defaultdict(int)
        
        for failure in failures:
            factors = failure.get('failure_reasons', [])
            for factor in factors:
                risk_factors[factor] += 1
        
        return [factor for factor, count in risk_factors.most_common(5)]

    def _generate_action_recommendations(
        self, 
        memories: List[Dict[str, Any]], 
        patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate action recommendations based on memories and patterns."""
        
        recommendations = []
        
        # From successful patterns
        for pattern in patterns[:3]:  # Top 3 patterns
            success_factors = pattern.get('success_factors', [])
            if success_factors:
                recommendations.append(f"Apply pattern: {success_factors[0]}")
        
        # From high-relevance memories
        high_relevance_memories = [m for m in memories if m.get('relevance_score', 0) > 0.7]
        
        if high_relevance_memories:
            memory = high_relevance_memories[0]
            action_taken = memory.get('action_taken', {})
            if action_taken.get('action_type'):
                recommendations.append(f"Consider action: {action_taken['action_type']}")
        
        return recommendations[:5]  # Limit to 5 recommendations

    def _generate_warnings(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Generate warnings based on failure patterns."""
        
        warnings = []
        
        high_risk_failures = [f for f in failures if f.get('risk_score', 0) > 0.7]
        
        for failure in high_risk_failures[:3]:  # Top 3 risks
            failure_reasons = failure.get('failure_reasons', [])
            if failure_reasons:
                warnings.append(f"Avoid: {failure_reasons[0]}")
        
        return warnings

    def _calculate_confidence_modifiers(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence modifiers based on historical performance."""
        
        if not memories:
            return {'historical': 1.0, 'recency': 1.0, 'consistency': 1.0}
        
        # Historical success rate modifier
        success_rate = self._calculate_historical_success_rate(memories)
        historical_modifier = 0.5 + success_rate  # 0.5 to 1.5
        
        # Recency modifier (recent memories more important)
        recent_memories = [m for m in memories if m.get('relevance_score', 0) > 0.6]
        recency_modifier = 1.0 + (len(recent_memories) / len(memories)) * 0.3
        
        # Consistency modifier (consistent patterns increase confidence)
        action_types = [m.get('action_taken', {}).get('action_type') for m in memories]
        if action_types:
            most_common_action = max(set(action_types), key=action_types.count)
            consistency = action_types.count(most_common_action) / len(action_types)
            consistency_modifier = 0.7 + consistency * 0.6  # 0.7 to 1.3
        else:
            consistency_modifier = 1.0
        
        return {
            'historical': min(historical_modifier, 1.5),
            'recency': min(recency_modifier, 1.3),
            'consistency': min(consistency_modifier, 1.3)
        }

    def _assess_outcome_quality(self, outcome: Dict[str, Any]) -> float:
        """Assess the quality of an outcome."""
        
        if not outcome:
            return 0.5
        
        success = outcome.get('success', False)
        net_change = outcome.get('net_value_change', 0)
        
        if success and net_change > 0:
            # Positive outcome, scale by magnitude
            return min(0.8 + net_change / 10000.0, 1.0)
        elif success and net_change >= 0:
            return 0.6  # Neutral successful outcome
        elif success:
            return 0.4  # Successful but negative outcome
        else:
            # Failed outcome, worse if large loss
            return max(0.1, 0.3 + net_change / 10000.0)  # net_change is negative