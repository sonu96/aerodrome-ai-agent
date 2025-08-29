"""
Pattern extraction and recognition from memory history.

This module implements sophisticated pattern recognition algorithms to identify
recurring behaviors, strategies, and market conditions from historical memories.
"""

import json
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from .system import MemorySystem


@dataclass
class PatternConfig:
    """Configuration for pattern extraction"""
    
    min_occurrences: int = 3
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.8
    time_window_days: int = 30
    max_patterns_per_type: int = 50
    pattern_decay_days: int = 90


@dataclass
class PatternMetrics:
    """Metrics for a discovered pattern"""
    
    occurrence_count: int
    success_rate: float
    avg_profit: float
    profit_variance: float
    confidence: float
    last_occurrence: datetime
    first_occurrence: datetime
    frequency_per_day: float


class PatternExtractor:
    """Extract patterns from memory history"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.config = PatternConfig()
        self.discovered_patterns = {}
        self.pattern_performance = {}
    
    async def extract_patterns(self) -> List[Dict]:
        """Extract patterns from memory history"""
        
        patterns = []
        
        print("Starting pattern extraction...")
        
        # Extract different pattern types
        trade_patterns = await self.extract_trade_patterns()
        patterns.extend(trade_patterns)
        
        timing_patterns = await self.extract_timing_patterns()
        patterns.extend(timing_patterns)
        
        market_patterns = await self.extract_market_patterns()
        patterns.extend(market_patterns)
        
        sequence_patterns = await self.extract_sequence_patterns()
        patterns.extend(sequence_patterns)
        
        failure_patterns = await self.extract_failure_patterns()
        patterns.extend(failure_patterns)
        
        # Store patterns and update pattern index
        for pattern in patterns:
            pattern_id = await self._store_pattern(pattern)
            self.discovered_patterns[pattern_id] = pattern
        
        print(f"Extracted {len(patterns)} patterns")
        
        return patterns
    
    async def extract_trade_patterns(self) -> List[Dict]:
        """Extract patterns from trading history"""
        
        print("Extracting trade patterns...")
        
        # Get all trade memories
        trades = await self.memory.search_memories(
            query="trade swap liquidity",
            limit=1000,
            filters={'category': 'trades'}
        )
        
        if len(trades) < self.config.min_occurrences:
            return []
        
        patterns = []
        
        # Group trades by similar conditions
        trade_groups = self._group_trades_by_conditions(trades)
        
        # Extract patterns from each group
        for group_key, group_trades in trade_groups.items():
            if len(group_trades) >= self.config.min_occurrences:
                pattern = await self._create_trade_pattern(group_key, group_trades)
                
                if pattern and pattern['confidence'] >= self.config.confidence_threshold:
                    patterns.append(pattern)
        
        # Find correlation patterns
        correlation_patterns = await self._find_trade_correlations(trades)
        patterns.extend(correlation_patterns)
        
        return patterns
    
    async def extract_timing_patterns(self) -> List[Dict]:
        """Extract timing-based patterns"""
        
        print("Extracting timing patterns...")
        
        # Get time-stamped memories
        memories = await self.memory.search_memories(
            query="trade opportunity",
            limit=500,
            filters={'max_age_days': self.config.time_window_days}
        )
        
        patterns = []
        
        # Analyze hourly patterns
        hourly_patterns = await self._analyze_hourly_patterns(memories)
        patterns.extend(hourly_patterns)
        
        # Analyze weekly patterns
        weekly_patterns = await self._analyze_weekly_patterns(memories)
        patterns.extend(weekly_patterns)
        
        # Analyze seasonal patterns
        seasonal_patterns = await self._analyze_seasonal_patterns(memories)
        patterns.extend(seasonal_patterns)
        
        return patterns
    
    async def extract_market_patterns(self) -> List[Dict]:
        """Extract market condition patterns"""
        
        print("Extracting market patterns...")
        
        # Get market observations
        market_memories = await self.memory.search_memories(
            query="market observation price volume",
            limit=500,
            filters={'category': 'market_observations'}
        )
        
        patterns = []
        
        # Price movement patterns
        price_patterns = await self._analyze_price_patterns(market_memories)
        patterns.extend(price_patterns)
        
        # Volume patterns
        volume_patterns = await self._analyze_volume_patterns(market_memories)
        patterns.extend(volume_patterns)
        
        # Volatility patterns
        volatility_patterns = await self._analyze_volatility_patterns(market_memories)
        patterns.extend(volatility_patterns)
        
        return patterns
    
    async def extract_sequence_patterns(self) -> List[Dict]:
        """Extract sequential action patterns"""
        
        print("Extracting sequence patterns...")
        
        # Get sequential memories (trades and observations)
        memories = await self.memory.search_memories(
            query="trade observation",
            limit=1000,
            filters={'max_age_days': self.config.time_window_days}
        )
        
        # Sort by timestamp
        sorted_memories = self._sort_memories_by_time(memories)
        
        patterns = []
        
        # Find action sequences
        action_sequences = await self._find_action_sequences(sorted_memories)
        patterns.extend(action_sequences)
        
        # Find decision trees
        decision_trees = await self._build_decision_trees(sorted_memories)
        patterns.extend(decision_trees)
        
        return patterns
    
    async def extract_failure_patterns(self) -> List[Dict]:
        """Extract patterns from failures and errors"""
        
        print("Extracting failure patterns...")
        
        # Get failure memories
        failures = await self.memory.search_memories(
            query="error failed failure exception",
            limit=200,
            filters={'category': 'failures'}
        )
        
        patterns = []
        
        if len(failures) >= self.config.min_occurrences:
            # Group by error type
            error_groups = self._group_failures_by_type(failures)
            
            for error_type, error_memories in error_groups.items():
                if len(error_memories) >= self.config.min_occurrences:
                    pattern = await self._create_failure_pattern(error_type, error_memories)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _group_trades_by_conditions(self, trades: List[Dict]) -> Dict[str, List[Dict]]:
        """Group trades by similar market conditions"""
        
        groups = defaultdict(list)
        
        for trade in trades:
            content = trade.get('content', {})
            
            # Create condition key
            conditions = {
                'action': content.get('action'),
                'pool': content.get('pool'),
                'market_condition': self._classify_market_condition(content),
                'time_bucket': self._get_time_bucket(content.get('timestamp'))
            }
            
            # Create hashable key
            key = json.dumps(conditions, sort_keys=True)
            groups[key].append(trade)
        
        return dict(groups)
    
    def _classify_market_condition(self, content: Dict) -> str:
        """Classify market condition from trade content"""
        
        market_context = content.get('market_context', {})
        
        # Simple classification based on available data
        volatility = market_context.get('volatility', 'medium')
        volume = market_context.get('volume', 'medium')
        trend = market_context.get('trend', 'neutral')
        
        # Create compound classification
        return f"{volatility}_{volume}_{trend}"
    
    def _get_time_bucket(self, timestamp_str: Optional[str]) -> str:
        """Get time bucket for timestamp"""
        
        if not timestamp_str:
            return 'unknown'
        
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            hour = dt.hour
            
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 24:
                return 'evening'
            else:
                return 'night'
        except ValueError:
            return 'unknown'
    
    async def _create_trade_pattern(self, pattern_key: str, group_trades: List[Dict]) -> Optional[Dict]:
        """Create trade pattern from group of similar trades"""
        
        if not group_trades:
            return None
        
        # Parse pattern key
        try:
            conditions = json.loads(pattern_key)
        except json.JSONDecodeError:
            return None
        
        # Calculate metrics
        metrics = self._calculate_pattern_metrics(group_trades)
        
        # Create pattern
        pattern = {
            'type': 'pattern',
            'pattern_type': 'trade_pattern',
            'conditions': conditions,
            'metrics': {
                'occurrence_count': metrics.occurrence_count,
                'success_rate': metrics.success_rate,
                'avg_profit': metrics.avg_profit,
                'profit_variance': metrics.profit_variance,
                'confidence': metrics.confidence,
                'frequency_per_day': metrics.frequency_per_day
            },
            'time_range': {
                'first_occurrence': metrics.first_occurrence.isoformat(),
                'last_occurrence': metrics.last_occurrence.isoformat()
            },
            'action_recommendation': self._get_action_recommendation(metrics),
            'risk_assessment': self._assess_pattern_risk(metrics),
            'discovered_at': datetime.now().isoformat()
        }
        
        return pattern
    
    def _calculate_pattern_metrics(self, memories: List[Dict]) -> PatternMetrics:
        """Calculate metrics for a pattern"""
        
        if not memories:
            return PatternMetrics(0, 0.0, 0.0, 0.0, 0.0, datetime.now(), datetime.now(), 0.0)
        
        # Extract data
        successes = []
        profits = []
        timestamps = []
        
        for memory in memories:
            content = memory.get('content', {})
            metadata = memory.get('metadata', {})
            
            successes.append(content.get('success', False))
            profits.append(content.get('profit', 0))
            
            timestamp_str = metadata.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except ValueError:
                    pass
        
        # Calculate metrics
        occurrence_count = len(memories)
        success_rate = sum(successes) / len(successes) if successes else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        profit_variance = np.var(profits) if profits else 0.0
        
        # Time-based metrics
        if timestamps:
            first_occurrence = min(timestamps)
            last_occurrence = max(timestamps)
            time_span = (last_occurrence - first_occurrence).days + 1
            frequency_per_day = occurrence_count / time_span if time_span > 0 else 0.0
        else:
            first_occurrence = datetime.now()
            last_occurrence = datetime.now()
            frequency_per_day = 0.0
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            occurrence_count, success_rate, profit_variance, frequency_per_day
        )
        
        return PatternMetrics(
            occurrence_count=occurrence_count,
            success_rate=success_rate,
            avg_profit=avg_profit,
            profit_variance=profit_variance,
            confidence=confidence,
            first_occurrence=first_occurrence,
            last_occurrence=last_occurrence,
            frequency_per_day=frequency_per_day
        )
    
    def _calculate_pattern_confidence(
        self,
        occurrence_count: int,
        success_rate: float,
        profit_variance: float,
        frequency: float
    ) -> float:
        """Calculate confidence score for pattern"""
        
        # Occurrence factor (more occurrences = higher confidence)
        occurrence_factor = min(occurrence_count / 10.0, 1.0)
        
        # Success factor
        success_factor = success_rate
        
        # Consistency factor (lower variance = higher confidence)
        consistency_factor = 1.0 / (1.0 + profit_variance) if profit_variance >= 0 else 0.5
        
        # Frequency factor (regular patterns = higher confidence)
        frequency_factor = min(frequency * 10, 1.0)
        
        # Combined confidence
        confidence = (
            occurrence_factor * 0.3 +
            success_factor * 0.4 +
            consistency_factor * 0.2 +
            frequency_factor * 0.1
        )
        
        return min(confidence, 1.0)
    
    def _get_action_recommendation(self, metrics: PatternMetrics) -> str:
        """Get action recommendation based on pattern metrics"""
        
        if metrics.success_rate > 0.8 and metrics.avg_profit > 0:
            return "strongly_recommended"
        elif metrics.success_rate > 0.6 and metrics.avg_profit > 0:
            return "recommended"
        elif metrics.success_rate > 0.4:
            return "neutral"
        else:
            return "avoid"
    
    def _assess_pattern_risk(self, metrics: PatternMetrics) -> Dict[str, Any]:
        """Assess risk level of pattern"""
        
        # Risk factors
        variance_risk = "high" if metrics.profit_variance > 100 else "medium" if metrics.profit_variance > 10 else "low"
        frequency_risk = "high" if metrics.frequency_per_day < 0.1 else "medium" if metrics.frequency_per_day < 0.5 else "low"
        success_risk = "high" if metrics.success_rate < 0.5 else "medium" if metrics.success_rate < 0.7 else "low"
        
        # Overall risk
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        avg_risk = (risk_scores[variance_risk] + risk_scores[frequency_risk] + risk_scores[success_risk]) / 3
        
        if avg_risk < 1.5:
            overall_risk = "low"
        elif avg_risk < 2.5:
            overall_risk = "medium"
        else:
            overall_risk = "high"
        
        return {
            'overall': overall_risk,
            'variance_risk': variance_risk,
            'frequency_risk': frequency_risk,
            'success_risk': success_risk,
            'risk_score': avg_risk
        }
    
    async def _find_trade_correlations(self, trades: List[Dict]) -> List[Dict]:
        """Find correlations between different trade types"""
        
        correlations = []
        
        # Group trades by pool
        pool_groups = defaultdict(list)
        for trade in trades:
            pool = trade.get('content', {}).get('pool', 'unknown')
            pool_groups[pool].append(trade)
        
        # Find correlations within each pool
        for pool, pool_trades in pool_groups.items():
            if len(pool_trades) >= 10:  # Need enough data for correlation
                pool_correlations = await self._analyze_pool_correlations(pool, pool_trades)
                correlations.extend(pool_correlations)
        
        return correlations
    
    async def _analyze_pool_correlations(self, pool: str, trades: List[Dict]) -> List[Dict]:
        """Analyze correlations within a specific pool"""
        
        correlations = []
        
        # Sort trades by time
        sorted_trades = self._sort_memories_by_time(trades)
        
        # Look for sequences of trades
        for i in range(len(sorted_trades) - 1):
            current_trade = sorted_trades[i]['content']
            next_trade = sorted_trades[i + 1]['content']
            
            # Check if trades are close in time (within 1 hour)
            current_time = self._parse_timestamp(sorted_trades[i]['metadata'].get('timestamp'))
            next_time = self._parse_timestamp(sorted_trades[i + 1]['metadata'].get('timestamp'))
            
            if current_time and next_time:
                time_diff = (next_time - current_time).total_seconds() / 3600
                
                if 0 < time_diff <= 1:  # Within 1 hour
                    correlation = {
                        'type': 'pattern',
                        'pattern_type': 'trade_correlation',
                        'pool': pool,
                        'first_action': current_trade.get('action'),
                        'second_action': next_trade.get('action'),
                        'time_gap_hours': time_diff,
                        'combined_success_rate': (
                            current_trade.get('success', False) and next_trade.get('success', False)
                        ),
                        'combined_profit': current_trade.get('profit', 0) + next_trade.get('profit', 0),
                        'discovered_at': datetime.now().isoformat()
                    }
                    
                    correlations.append(correlation)
        
        return correlations
    
    async def _analyze_hourly_patterns(self, memories: List[Dict]) -> List[Dict]:
        """Analyze patterns by hour of day"""
        
        hourly_data = defaultdict(list)
        
        for memory in memories:
            timestamp_str = memory.get('metadata', {}).get('timestamp')
            if timestamp_str:
                dt = self._parse_timestamp(timestamp_str)
                if dt:
                    hour = dt.hour
                    hourly_data[hour].append(memory)
        
        patterns = []
        
        for hour, hour_memories in hourly_data.items():
            if len(hour_memories) >= self.config.min_occurrences:
                metrics = self._calculate_pattern_metrics(hour_memories)
                
                if metrics.confidence >= self.config.confidence_threshold:
                    pattern = {
                        'type': 'pattern',
                        'pattern_type': 'hourly_pattern',
                        'hour': hour,
                        'time_period': self._get_time_period_name(hour),
                        'metrics': {
                            'occurrence_count': metrics.occurrence_count,
                            'success_rate': metrics.success_rate,
                            'avg_profit': metrics.avg_profit,
                            'confidence': metrics.confidence
                        },
                        'discovered_at': datetime.now().isoformat()
                    }
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_weekly_patterns(self, memories: List[Dict]) -> List[Dict]:
        """Analyze patterns by day of week"""
        
        daily_data = defaultdict(list)
        
        for memory in memories:
            timestamp_str = memory.get('metadata', {}).get('timestamp')
            if timestamp_str:
                dt = self._parse_timestamp(timestamp_str)
                if dt:
                    day_of_week = dt.weekday()  # 0 = Monday
                    daily_data[day_of_week].append(memory)
        
        patterns = []
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day_num, day_memories in daily_data.items():
            if len(day_memories) >= self.config.min_occurrences:
                metrics = self._calculate_pattern_metrics(day_memories)
                
                if metrics.confidence >= self.config.confidence_threshold:
                    pattern = {
                        'type': 'pattern',
                        'pattern_type': 'weekly_pattern',
                        'day_of_week': day_num,
                        'day_name': day_names[day_num],
                        'metrics': {
                            'occurrence_count': metrics.occurrence_count,
                            'success_rate': metrics.success_rate,
                            'avg_profit': metrics.avg_profit,
                            'confidence': metrics.confidence
                        },
                        'discovered_at': datetime.now().isoformat()
                    }
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_seasonal_patterns(self, memories: List[Dict]) -> List[Dict]:
        """Analyze seasonal patterns"""
        
        # For simplicity, analyze by month
        monthly_data = defaultdict(list)
        
        for memory in memories:
            timestamp_str = memory.get('metadata', {}).get('timestamp')
            if timestamp_str:
                dt = self._parse_timestamp(timestamp_str)
                if dt:
                    month = dt.month
                    monthly_data[month].append(memory)
        
        patterns = []
        
        month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        for month_num, month_memories in monthly_data.items():
            if len(month_memories) >= self.config.min_occurrences:
                metrics = self._calculate_pattern_metrics(month_memories)
                
                if metrics.confidence >= self.config.confidence_threshold:
                    pattern = {
                        'type': 'pattern',
                        'pattern_type': 'seasonal_pattern',
                        'month': month_num,
                        'month_name': month_names[month_num - 1],
                        'season': self._get_season(month_num),
                        'metrics': {
                            'occurrence_count': metrics.occurrence_count,
                            'success_rate': metrics.success_rate,
                            'avg_profit': metrics.avg_profit,
                            'confidence': metrics.confidence
                        },
                        'discovered_at': datetime.now().isoformat()
                    }
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_price_patterns(self, market_memories: List[Dict]) -> List[Dict]:
        """Analyze price movement patterns"""
        
        patterns = []
        
        # Extract price data
        price_data = []
        for memory in market_memories:
            content = memory.get('content', {})
            price = content.get('price')
            if price is not None:
                price_data.append({
                    'price': price,
                    'timestamp': content.get('timestamp'),
                    'pool': content.get('pool'),
                    'memory': memory
                })
        
        if len(price_data) < 10:
            return patterns
        
        # Group by pool
        pool_prices = defaultdict(list)
        for item in price_data:
            pool = item['pool']
            if pool:
                pool_prices[pool].append(item)
        
        # Analyze each pool
        for pool, prices in pool_prices.items():
            if len(prices) >= 5:
                pool_patterns = await self._analyze_pool_price_patterns(pool, prices)
                patterns.extend(pool_patterns)
        
        return patterns
    
    async def _analyze_pool_price_patterns(self, pool: str, price_data: List[Dict]) -> List[Dict]:
        """Analyze price patterns for a specific pool"""
        
        patterns = []
        
        # Sort by timestamp
        sorted_prices = sorted(price_data, key=lambda x: x.get('timestamp', ''))
        
        # Calculate price movements
        movements = []
        for i in range(1, len(sorted_prices)):
            prev_price = sorted_prices[i-1]['price']
            curr_price = sorted_prices[i]['price']
            
            if prev_price > 0:
                change_pct = (curr_price - prev_price) / prev_price * 100
                movements.append(change_pct)
        
        if movements:
            # Analyze movement patterns
            avg_movement = np.mean(movements)
            movement_std = np.std(movements)
            
            pattern = {
                'type': 'pattern',
                'pattern_type': 'price_pattern',
                'pool': pool,
                'metrics': {
                    'avg_movement_pct': avg_movement,
                    'volatility': movement_std,
                    'data_points': len(movements),
                    'trend': 'bullish' if avg_movement > 1 else 'bearish' if avg_movement < -1 else 'neutral'
                },
                'discovered_at': datetime.now().isoformat()
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _analyze_volume_patterns(self, market_memories: List[Dict]) -> List[Dict]:
        """Analyze volume patterns"""
        
        patterns = []
        
        # Extract volume data
        volume_data = []
        for memory in market_memories:
            content = memory.get('content', {})
            volume = content.get('volume')
            if volume is not None:
                volume_data.append({
                    'volume': volume,
                    'pool': content.get('pool'),
                    'timestamp': content.get('timestamp')
                })
        
        if len(volume_data) >= 10:
            # Analyze volume trends
            volumes = [item['volume'] for item in volume_data]
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            
            pattern = {
                'type': 'pattern',
                'pattern_type': 'volume_pattern',
                'metrics': {
                    'avg_volume': avg_volume,
                    'volume_volatility': volume_std,
                    'data_points': len(volumes),
                    'volume_trend': self._classify_volume_trend(volumes)
                },
                'discovered_at': datetime.now().isoformat()
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _analyze_volatility_patterns(self, market_memories: List[Dict]) -> List[Dict]:
        """Analyze volatility patterns"""
        
        patterns = []
        
        # Extract volatility data
        volatility_data = []
        for memory in market_memories:
            content = memory.get('content', {})
            volatility = content.get('volatility')
            if volatility is not None:
                volatility_data.append(volatility)
        
        if len(volatility_data) >= 10:
            avg_volatility = np.mean(volatility_data)
            volatility_trend = self._classify_volatility_trend(volatility_data)
            
            pattern = {
                'type': 'pattern',
                'pattern_type': 'volatility_pattern',
                'metrics': {
                    'avg_volatility': avg_volatility,
                    'volatility_trend': volatility_trend,
                    'data_points': len(volatility_data)
                },
                'discovered_at': datetime.now().isoformat()
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _find_action_sequences(self, sorted_memories: List[Dict]) -> List[Dict]:
        """Find sequential action patterns"""
        
        patterns = []
        
        # Extract action sequences
        sequences = []
        current_sequence = []
        
        for memory in sorted_memories:
            content = memory.get('content', {})
            action = content.get('action')
            
            if action:
                current_sequence.append(action)
                
                # Limit sequence length
                if len(current_sequence) > 5:
                    current_sequence = current_sequence[-5:]
            else:
                # Break sequence on non-action memory
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence.copy())
                current_sequence = []
        
        # Add final sequence
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        # Find common sequences
        sequence_counts = Counter()
        for seq in sequences:
            for length in range(2, len(seq) + 1):
                for start in range(len(seq) - length + 1):
                    subseq = tuple(seq[start:start + length])
                    sequence_counts[subseq] += 1
        
        # Create patterns from frequent sequences
        for sequence, count in sequence_counts.items():
            if count >= self.config.min_occurrences:
                pattern = {
                    'type': 'pattern',
                    'pattern_type': 'action_sequence',
                    'sequence': list(sequence),
                    'occurrence_count': count,
                    'sequence_length': len(sequence),
                    'discovered_at': datetime.now().isoformat()
                }
                patterns.append(pattern)
        
        return patterns
    
    async def _build_decision_trees(self, sorted_memories: List[Dict]) -> List[Dict]:
        """Build decision tree patterns"""
        
        patterns = []
        
        # This is a simplified decision tree implementation
        # In practice, you would use a proper decision tree algorithm
        
        # Group memories by conditions and outcomes
        decision_data = []
        
        for memory in sorted_memories:
            content = memory.get('content', {})
            
            if content.get('type') == 'trade':
                decision_point = {
                    'conditions': {
                        'action': content.get('action'),
                        'pool': content.get('pool'),
                        'market_condition': content.get('market_conditions', {})
                    },
                    'outcome': {
                        'success': content.get('success', False),
                        'profit': content.get('profit', 0)
                    }
                }
                decision_data.append(decision_point)
        
        if len(decision_data) >= 10:
            # Create simple decision tree pattern
            pattern = {
                'type': 'pattern',
                'pattern_type': 'decision_tree',
                'data_points': len(decision_data),
                'rules': await self._extract_simple_rules(decision_data),
                'discovered_at': datetime.now().isoformat()
            }
            patterns.append(pattern)
        
        return patterns
    
    def _group_failures_by_type(self, failures: List[Dict]) -> Dict[str, List[Dict]]:
        """Group failures by error type"""
        
        groups = defaultdict(list)
        
        for failure in failures:
            content = failure.get('content', {})
            error_type = content.get('error_type', content.get('type', 'unknown'))
            groups[error_type].append(failure)
        
        return dict(groups)
    
    async def _create_failure_pattern(self, error_type: str, error_memories: List[Dict]) -> Optional[Dict]:
        """Create failure pattern from error memories"""
        
        if not error_memories:
            return None
        
        # Analyze failure context
        contexts = []
        for memory in error_memories:
            content = memory.get('content', {})
            context = content.get('context', {})
            contexts.append(context)
        
        # Find common failure conditions
        common_conditions = self._find_common_conditions(contexts)
        
        pattern = {
            'type': 'pattern',
            'pattern_type': 'failure_pattern',
            'error_type': error_type,
            'occurrence_count': len(error_memories),
            'common_conditions': common_conditions,
            'prevention_recommendations': self._generate_prevention_recommendations(error_type, common_conditions),
            'discovered_at': datetime.now().isoformat()
        }
        
        return pattern
    
    def _sort_memories_by_time(self, memories: List[Dict]) -> List[Dict]:
        """Sort memories by timestamp"""
        
        def get_timestamp(memory):
            timestamp_str = memory.get('metadata', {}).get('timestamp')
            if timestamp_str:
                try:
                    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    pass
            return datetime.min
        
        return sorted(memories, key=get_timestamp)
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime"""
        
        if not timestamp_str:
            return None
        
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            return None
    
    def _get_time_period_name(self, hour: int) -> str:
        """Get time period name from hour"""
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    
    def _get_season(self, month: int) -> str:
        """Get season from month number"""
        
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _classify_volume_trend(self, volumes: List[float]) -> str:
        """Classify volume trend"""
        
        if len(volumes) < 2:
            return 'insufficient_data'
        
        # Simple trend analysis
        first_half = volumes[:len(volumes)//2]
        second_half = volumes[len(volumes)//2:]
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        change_pct = (avg_second - avg_first) / avg_first * 100 if avg_first > 0 else 0
        
        if change_pct > 10:
            return 'increasing'
        elif change_pct < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def _classify_volatility_trend(self, volatilities: List[float]) -> str:
        """Classify volatility trend"""
        
        if len(volatilities) < 2:
            return 'insufficient_data'
        
        # Calculate trend
        recent_avg = np.mean(volatilities[-5:]) if len(volatilities) >= 5 else np.mean(volatilities)
        overall_avg = np.mean(volatilities)
        
        if recent_avg > overall_avg * 1.2:
            return 'increasing'
        elif recent_avg < overall_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _extract_simple_rules(self, decision_data: List[Dict]) -> List[Dict]:
        """Extract simple decision rules from data"""
        
        rules = []
        
        # Group by action
        action_groups = defaultdict(list)
        for data in decision_data:
            action = data['conditions'].get('action', 'unknown')
            action_groups[action].append(data)
        
        # Create rules for each action
        for action, data_points in action_groups.items():
            if len(data_points) >= 3:
                successes = [d for d in data_points if d['outcome']['success']]
                success_rate = len(successes) / len(data_points)
                avg_profit = np.mean([d['outcome']['profit'] for d in data_points])
                
                rule = {
                    'condition': f"action == '{action}'",
                    'success_rate': success_rate,
                    'avg_profit': avg_profit,
                    'sample_size': len(data_points),
                    'recommendation': 'execute' if success_rate > 0.6 and avg_profit > 0 else 'avoid'
                }
                rules.append(rule)
        
        return rules
    
    def _find_common_conditions(self, contexts: List[Dict]) -> Dict[str, Any]:
        """Find common conditions across contexts"""
        
        if not contexts:
            return {}
        
        common = {}
        
        # Find keys that appear in most contexts
        all_keys = set()
        for context in contexts:
            all_keys.update(context.keys())
        
        for key in all_keys:
            values = [context.get(key) for context in contexts if key in context]
            
            if len(values) >= len(contexts) * 0.5:  # Appears in at least 50% of contexts
                # Find most common value
                if values:
                    value_counts = Counter(values)
                    most_common = value_counts.most_common(1)[0]
                    if most_common[1] >= len(values) * 0.6:  # Most common value appears in 60%+ of cases
                        common[key] = most_common[0]
        
        return common
    
    def _generate_prevention_recommendations(self, error_type: str, common_conditions: Dict) -> List[str]:
        """Generate prevention recommendations for failure patterns"""
        
        recommendations = []
        
        # Generic recommendations based on error type
        if 'timeout' in error_type.lower():
            recommendations.append("Increase timeout duration")
            recommendations.append("Check network connectivity before operation")
        
        elif 'slippage' in error_type.lower():
            recommendations.append("Reduce trade size during high volatility")
            recommendations.append("Use more conservative slippage tolerance")
        
        elif 'insufficient' in error_type.lower():
            recommendations.append("Check balance before trade execution")
            recommendations.append("Implement better fund management")
        
        # Condition-based recommendations
        if common_conditions:
            for condition, value in common_conditions.items():
                if condition == 'gas_price' and isinstance(value, (int, float)):
                    recommendations.append(f"Avoid operations when gas price > {value}")
                elif condition == 'time_of_day':
                    recommendations.append(f"Be cautious during {value} hours")
        
        return recommendations
    
    async def _store_pattern(self, pattern: Dict) -> str:
        """Store pattern in memory system"""
        
        pattern_id = await self.memory.add_memory(
            content=pattern,
            metadata={
                'type': 'pattern',
                'category': 'patterns',
                'pattern_type': pattern.get('pattern_type'),
                'importance': 1.0,
                'auto_expire': False
            }
        )
        
        return pattern_id
    
    async def find_matching_patterns(self, query_conditions: Dict) -> List[Dict]:
        """Find patterns matching given conditions"""
        
        matching_patterns = []
        
        for pattern_id, pattern in self.discovered_patterns.items():
            if self._pattern_matches_conditions(pattern, query_conditions):
                matching_patterns.append(pattern)
        
        # Sort by confidence
        matching_patterns.sort(
            key=lambda p: p.get('metrics', {}).get('confidence', 0),
            reverse=True
        )
        
        return matching_patterns
    
    def _pattern_matches_conditions(self, pattern: Dict, query_conditions: Dict) -> bool:
        """Check if pattern matches query conditions"""
        
        pattern_conditions = pattern.get('conditions', {})
        
        for key, value in query_conditions.items():
            if key in pattern_conditions:
                if pattern_conditions[key] != value:
                    return False
        
        return True
    
    async def get_pattern_performance(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a pattern"""
        
        return self.pattern_performance.get(pattern_id)
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all discovered patterns"""
        
        summary = {
            'total_patterns': len(self.discovered_patterns),
            'by_type': {},
            'high_confidence_patterns': 0,
            'recommended_patterns': 0
        }
        
        for pattern in self.discovered_patterns.values():
            pattern_type = pattern.get('pattern_type', 'unknown')
            summary['by_type'][pattern_type] = summary['by_type'].get(pattern_type, 0) + 1
            
            confidence = pattern.get('metrics', {}).get('confidence', 0)
            if confidence >= 0.8:
                summary['high_confidence_patterns'] += 1
            
            recommendation = pattern.get('action_recommendation')
            if recommendation in ['recommended', 'strongly_recommended']:
                summary['recommended_patterns'] += 1
        
        return summary