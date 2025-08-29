"""
Memory system monitoring and performance metrics.

This module provides comprehensive monitoring, performance tracking,
and alerting capabilities for the memory system.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import os

from .system import MemorySystem


@dataclass
class MemorySystemAlert:
    """Alert for memory system issues"""
    
    level: str  # 'info', 'warning', 'error', 'critical'
    category: str  # 'performance', 'capacity', 'health', 'security'
    message: str
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for memory operations"""
    
    operation_count: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    error_count: int = 0
    timeout_count: int = 0
    cache_hit_ratio: float = 0.0
    throughput_ops_per_second: float = 0.0


@dataclass
class CapacityMetrics:
    """Capacity metrics for memory tiers"""
    
    total_memories: int = 0
    tier_distribution: Dict[str, int] = None
    storage_utilization: Dict[str, float] = None
    compression_ratio: float = 0.0
    growth_rate: float = 0.0
    projected_capacity_days: int = 0
    
    def __post_init__(self):
        if self.tier_distribution is None:
            self.tier_distribution = {}
        if self.storage_utilization is None:
            self.storage_utilization = {}


@dataclass
class QualityMetrics:
    """Data quality metrics"""
    
    pattern_count: int = 0
    pattern_confidence_avg: float = 0.0
    successful_trades_ratio: float = 0.0
    data_completeness: float = 0.0
    duplicate_ratio: float = 0.0
    corruption_count: int = 0


class MemoryMetrics:
    """Monitor memory system performance and health"""
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        
        # Metrics storage
        self.performance_history = deque(maxlen=1000)  # Last 1000 operations
        self.capacity_history = deque(maxlen=144)  # Last 24 hours (10min intervals)
        self.quality_history = deque(maxlen=144)
        
        # Real-time tracking
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.alert_history = []
        
        # Thresholds for alerting
        self.alert_thresholds = {
            'max_response_time': 5.0,  # seconds
            'max_error_rate': 0.1,  # 10%
            'min_cache_hit_ratio': 0.7,  # 70%
            'max_tier_utilization': 0.9,  # 90%
            'min_pattern_confidence': 0.5,  # 50%
            'max_memory_growth_rate': 0.2  # 20% per day
        }
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.collection_interval = 600  # 10 minutes
        self.retention_days = 30
        
        # Background monitoring task
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start background monitoring"""
        
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            print("Memory metrics monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            print("Memory metrics monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        
        try:
            while self.monitoring_enabled:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.collection_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in monitoring loop: {str(e)}")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics manually"""
        
        return await self._collect_metrics()
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Internal metrics collection"""
        
        start_time = time.time()
        
        try:
            # Collect performance metrics
            performance = await self._collect_performance_metrics()
            
            # Collect capacity metrics
            capacity = await self._collect_capacity_metrics()
            
            # Collect quality metrics
            quality = await self._collect_quality_metrics()
            
            # Collect system health metrics
            health = await self._collect_health_metrics()
            
            # Store in history
            timestamp = datetime.now()
            self.performance_history.append((timestamp, performance))
            self.capacity_history.append((timestamp, capacity))
            self.quality_history.append((timestamp, quality))
            
            collection_time = time.time() - start_time
            
            metrics_summary = {
                'timestamp': timestamp.isoformat(),
                'collection_time_seconds': collection_time,
                'performance': asdict(performance),
                'capacity': asdict(capacity),
                'quality': asdict(quality),
                'health': health,
                'alerts_active': len([a for a in self.alert_history if not a.resolved])
            }
            
            return metrics_summary
        
        except Exception as e:
            error_alert = MemorySystemAlert(
                level='error',
                category='health',
                message=f'Metrics collection failed: {str(e)}',
                timestamp=datetime.now(),
                details={'error': str(e), 'collection_time': time.time() - start_time}
            )
            self.alert_history.append(error_alert)
            
            raise
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics"""
        
        # Initialize memory system components if needed
        await self.memory.initialize_components()
        
        metrics = PerformanceMetrics()
        
        # Get cache statistics if available
        if hasattr(self.memory, 'optimized_access'):
            cache_stats = self.memory.optimized_access.get_cache_stats()
            
            search_cache = cache_stats.get('search_cache', {})
            memory_cache = cache_stats.get('memory_cache', {})
            
            # Calculate overall cache hit ratio
            total_hits = search_cache.get('hits', 0) + memory_cache.get('hits', 0)
            total_requests = (search_cache.get('total_requests', 0) + 
                            memory_cache.get('total_requests', 0))
            
            metrics.cache_hit_ratio = total_hits / total_requests if total_requests > 0 else 0
        
        # Calculate operation metrics from recent history
        if self.operation_times:
            all_times = []
            for op_list in self.operation_times.values():
                all_times.extend([op['duration'] for op in op_list])
            
            if all_times:
                metrics.avg_response_time = sum(all_times) / len(all_times)
                metrics.max_response_time = max(all_times)
                metrics.min_response_time = min(all_times)
                metrics.operation_count = len(all_times)
        
        # Calculate error metrics
        metrics.error_count = sum(self.error_counts.values())
        
        # Calculate throughput (operations per second)
        current_time = time.time()
        recent_ops = []
        
        for op_list in self.operation_times.values():
            recent_ops.extend([
                op for op in op_list 
                if current_time - op.get('timestamp', 0) < 60  # Last minute
            ])
        
        metrics.throughput_ops_per_second = len(recent_ops) / 60 if recent_ops else 0
        
        return metrics
    
    async def _collect_capacity_metrics(self) -> CapacityMetrics:
        """Collect capacity metrics"""
        
        metrics = CapacityMetrics()
        
        # Get tier statistics
        if hasattr(self.memory, 'tiers_manager'):
            tier_stats = await self.memory.tiers_manager.get_tier_statistics()
            
            # Total memories
            metrics.total_memories = sum(
                stats.get('current_size', 0) 
                for stats in tier_stats.values()
                if isinstance(stats, dict)
            )
            
            # Tier distribution
            metrics.tier_distribution = {
                tier_name: stats.get('current_size', 0)
                for tier_name, stats in tier_stats.items()
                if isinstance(stats, dict)
            }
            
            # Storage utilization
            metrics.storage_utilization = {
                tier_name: stats.get('utilization', 0)
                for tier_name, stats in tier_stats.items()
                if isinstance(stats, dict)
            }
            
            # Compression ratio
            total_compressed = sum(
                stats.get('compressed_memories', 0)
                for stats in tier_stats.values()
                if isinstance(stats, dict)
            )
            
            metrics.compression_ratio = (
                total_compressed / metrics.total_memories
                if metrics.total_memories > 0 else 0
            )
        
        else:
            # Fallback to direct tier inspection
            for tier_name, tier_data in self.memory.tiers.items():
                metrics.tier_distribution[tier_name] = len(tier_data)
                metrics.total_memories += len(tier_data)
        
        # Calculate growth rate from history
        if len(self.capacity_history) >= 2:
            current_capacity = metrics.total_memories
            prev_timestamp, prev_metrics = self.capacity_history[-2]
            prev_capacity = prev_metrics.total_memories
            
            time_diff_hours = (datetime.now() - prev_timestamp).total_seconds() / 3600
            
            if time_diff_hours > 0 and prev_capacity > 0:
                growth = (current_capacity - prev_capacity) / prev_capacity
                metrics.growth_rate = growth / time_diff_hours * 24  # Daily growth rate
        
        # Project capacity (simplified)
        if metrics.growth_rate > 0:
            # Assume some maximum capacity
            max_capacity = 50000  # Configurable
            remaining_capacity = max_capacity - metrics.total_memories
            
            if remaining_capacity > 0:
                metrics.projected_capacity_days = int(
                    remaining_capacity / (metrics.total_memories * metrics.growth_rate)
                )
        
        return metrics
    
    async def _collect_quality_metrics(self) -> QualityMetrics:
        """Collect data quality metrics"""
        
        metrics = QualityMetrics()
        
        # Pattern analysis
        if hasattr(self.memory, 'pattern_extractor'):
            pattern_summary = self.memory.pattern_extractor.get_pattern_summary()
            metrics.pattern_count = pattern_summary.get('total_patterns', 0)
            
            # Calculate average pattern confidence
            total_confidence = 0
            pattern_count = 0
            
            for pattern in self.memory.pattern_extractor.discovered_patterns.values():
                confidence = pattern.get('metrics', {}).get('confidence', 0)
                if confidence > 0:
                    total_confidence += confidence
                    pattern_count += 1
            
            metrics.pattern_confidence_avg = (
                total_confidence / pattern_count if pattern_count > 0 else 0
            )
        
        # Trade success analysis
        try:
            # Get recent trades
            recent_trades = await self.memory.search_memories(
                query="trade",
                limit=100,
                filters={'category': 'trades', 'max_age_days': 7}
            )
            
            if recent_trades:
                successful = sum(
                    1 for trade in recent_trades
                    if trade.get('content', {}).get('success', False)
                )
                metrics.successful_trades_ratio = successful / len(recent_trades)
            
        except Exception as e:
            print(f"Error calculating trade success ratio: {str(e)}")
        
        # Data completeness check
        total_memories = sum(len(tier) for tier in self.memory.tiers.values())
        complete_memories = 0
        
        for tier_data in self.memory.tiers.values():
            for memory_data in tier_data.values():
                content = memory_data.get('content', {})
                metadata = memory_data.get('metadata', {})
                
                # Check for essential fields
                essential_fields = ['type', 'timestamp']
                if all(field in content or field in metadata for field in essential_fields):
                    complete_memories += 1
        
        metrics.data_completeness = (
            complete_memories / total_memories if total_memories > 0 else 0
        )
        
        return metrics
    
    async def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect system health metrics"""
        
        health = {
            'system_status': 'healthy',
            'component_status': {},
            'last_health_check': datetime.now().isoformat(),
            'uptime_hours': 0,  # Would track actual uptime
            'memory_usage_mb': 0,  # Would track actual memory usage
            'disk_usage_mb': 0  # Would track actual disk usage
        }
        
        # Check memory system health
        try:
            system_health = await self.memory.health_check()
            health['component_status']['memory_system'] = system_health['status']
            health['mem0_connected'] = system_health.get('mem0_connected', False)
        except Exception as e:
            health['component_status']['memory_system'] = 'unhealthy'
            health['system_status'] = 'degraded'
        
        # Check cache health
        if hasattr(self.memory, 'optimized_access'):
            try:
                cache_stats = self.memory.optimized_access.get_cache_stats()
                health['component_status']['cache'] = 'healthy'
                health['cache_performance'] = cache_stats
            except Exception as e:
                health['component_status']['cache'] = 'unhealthy'
        
        # Check pattern extractor health
        if hasattr(self.memory, 'pattern_extractor'):
            health['component_status']['pattern_extractor'] = 'healthy'
        
        # Overall health assessment
        unhealthy_components = [
            comp for comp, status in health['component_status'].items()
            if status == 'unhealthy'
        ]
        
        if unhealthy_components:
            if len(unhealthy_components) > 1:
                health['system_status'] = 'critical'
            else:
                health['system_status'] = 'degraded'
        
        return health
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        
        if not self.performance_history or not self.capacity_history:
            return
        
        # Get latest metrics
        _, latest_performance = self.performance_history[-1]
        _, latest_capacity = self.capacity_history[-1]
        
        alerts = []
        
        # Performance alerts
        if latest_performance.max_response_time > self.alert_thresholds['max_response_time']:
            alerts.append(MemorySystemAlert(
                level='warning',
                category='performance',
                message=f'High response time: {latest_performance.max_response_time:.2f}s',
                timestamp=datetime.now(),
                details={'response_time': latest_performance.max_response_time}
            ))
        
        if latest_performance.cache_hit_ratio < self.alert_thresholds['min_cache_hit_ratio']:
            alerts.append(MemorySystemAlert(
                level='warning',
                category='performance',
                message=f'Low cache hit ratio: {latest_performance.cache_hit_ratio:.2%}',
                timestamp=datetime.now(),
                details={'cache_hit_ratio': latest_performance.cache_hit_ratio}
            ))
        
        # Capacity alerts
        for tier, utilization in latest_capacity.storage_utilization.items():
            if utilization > self.alert_thresholds['max_tier_utilization']:
                alerts.append(MemorySystemAlert(
                    level='warning',
                    category='capacity',
                    message=f'High {tier} tier utilization: {utilization:.1%}',
                    timestamp=datetime.now(),
                    details={'tier': tier, 'utilization': utilization}
                ))
        
        if latest_capacity.growth_rate > self.alert_thresholds['max_memory_growth_rate']:
            alerts.append(MemorySystemAlert(
                level='info',
                category='capacity',
                message=f'High memory growth rate: {latest_capacity.growth_rate:.1%}/day',
                timestamp=datetime.now(),
                details={'growth_rate': latest_capacity.growth_rate}
            ))
        
        # Add new alerts
        for alert in alerts:
            # Check if similar alert already exists
            similar_exists = any(
                existing.category == alert.category and
                existing.message == alert.message and
                not existing.resolved
                for existing in self.alert_history
            )
            
            if not similar_exists:
                self.alert_history.append(alert)
                await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: MemorySystemAlert):
        """Handle new alert"""
        
        print(f"[{alert.level.upper()}] {alert.category}: {alert.message}")
        
        # Auto-resolution for some alerts
        if alert.category == 'performance' and 'cache' in alert.message.lower():
            # Try to optimize cache performance
            if hasattr(self.memory, 'optimized_access'):
                try:
                    await self.memory.optimized_access.optimize_cache_performance()
                    alert.resolved = True
                    alert.resolution_timestamp = datetime.now()
                except Exception as e:
                    print(f"Failed to auto-resolve cache alert: {str(e)}")
    
    def track_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool = True,
        error: str = None
    ):
        """Track individual operation performance"""
        
        operation_data = {
            'duration': duration,
            'timestamp': time.time(),
            'success': success,
            'error': error
        }
        
        self.operation_times[operation_type].append(operation_data)
        
        # Keep only recent operations
        cutoff_time = time.time() - 3600  # 1 hour
        self.operation_times[operation_type] = [
            op for op in self.operation_times[operation_type]
            if op['timestamp'] > cutoff_time
        ]
        
        # Track errors
        if not success:
            self.error_counts[operation_type] += 1
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        
        if not self.performance_history:
            return {'status': 'no_data'}
        
        # Latest metrics
        _, latest_performance = self.performance_history[-1]
        _, latest_capacity = self.capacity_history[-1]
        
        # Active alerts
        active_alerts = [a for a in self.alert_history if not a.resolved]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'avg_response_time': latest_performance.avg_response_time,
                'cache_hit_ratio': latest_performance.cache_hit_ratio,
                'throughput_ops_per_second': latest_performance.throughput_ops_per_second,
                'error_count': latest_performance.error_count
            },
            'capacity': {
                'total_memories': latest_capacity.total_memories,
                'tier_distribution': latest_capacity.tier_distribution,
                'compression_ratio': latest_capacity.compression_ratio,
                'growth_rate_daily': latest_capacity.growth_rate
            },
            'alerts': {
                'active_count': len(active_alerts),
                'by_level': {
                    level: len([a for a in active_alerts if a.level == level])
                    for level in ['info', 'warning', 'error', 'critical']
                }
            },
            'health_status': 'healthy' if not active_alerts else 'issues_detected'
        }
    
    def get_historical_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical trends for specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter historical data
        recent_performance = [
            (timestamp, metrics) for timestamp, metrics in self.performance_history
            if timestamp > cutoff_time
        ]
        
        recent_capacity = [
            (timestamp, metrics) for timestamp, metrics in self.capacity_history
            if timestamp > cutoff_time
        ]
        
        if not recent_performance or not recent_capacity:
            return {'status': 'insufficient_data'}
        
        # Calculate trends
        performance_trend = self._calculate_performance_trend(recent_performance)
        capacity_trend = self._calculate_capacity_trend(recent_capacity)
        
        return {
            'time_period_hours': hours,
            'data_points': len(recent_performance),
            'performance_trend': performance_trend,
            'capacity_trend': capacity_trend,
            'alert_frequency': len([
                a for a in self.alert_history 
                if a.timestamp > cutoff_time
            ]),
            'system_stability': self._assess_system_stability(recent_performance)
        }
    
    def _calculate_performance_trend(
        self, 
        performance_history: List[Tuple[datetime, PerformanceMetrics]]
    ) -> Dict[str, Any]:
        """Calculate performance trends"""
        
        if len(performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Extract time series data
        response_times = [metrics.avg_response_time for _, metrics in performance_history]
        cache_ratios = [metrics.cache_hit_ratio for _, metrics in performance_history]
        throughputs = [metrics.throughput_ops_per_second for _, metrics in performance_history]
        
        return {
            'response_time': {
                'trend': 'improving' if response_times[-1] < response_times[0] else 'degrading',
                'change_percent': ((response_times[-1] - response_times[0]) / response_times[0] * 100
                                 if response_times[0] > 0 else 0)
            },
            'cache_hit_ratio': {
                'trend': 'improving' if cache_ratios[-1] > cache_ratios[0] else 'degrading',
                'current_ratio': cache_ratios[-1]
            },
            'throughput': {
                'trend': 'increasing' if throughputs[-1] > throughputs[0] else 'decreasing',
                'avg_ops_per_second': sum(throughputs) / len(throughputs)
            }
        }
    
    def _calculate_capacity_trend(
        self,
        capacity_history: List[Tuple[datetime, CapacityMetrics]]
    ) -> Dict[str, Any]:
        """Calculate capacity trends"""
        
        if len(capacity_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Extract capacity data
        memory_counts = [metrics.total_memories for _, metrics in capacity_history]
        compression_ratios = [metrics.compression_ratio for _, metrics in capacity_history]
        
        return {
            'memory_growth': {
                'trend': 'increasing' if memory_counts[-1] > memory_counts[0] else 'stable',
                'total_change': memory_counts[-1] - memory_counts[0],
                'current_count': memory_counts[-1]
            },
            'compression_efficiency': {
                'trend': 'improving' if compression_ratios[-1] > compression_ratios[0] else 'stable',
                'current_ratio': compression_ratios[-1]
            }
        }
    
    def _assess_system_stability(
        self,
        performance_history: List[Tuple[datetime, PerformanceMetrics]]
    ) -> str:
        """Assess overall system stability"""
        
        if len(performance_history) < 5:
            return 'insufficient_data'
        
        # Calculate variance in key metrics
        response_times = [metrics.avg_response_time for _, metrics in performance_history]
        cache_ratios = [metrics.cache_hit_ratio for _, metrics in performance_history]
        
        # Simple stability assessment based on variance
        import statistics
        
        response_time_cv = (statistics.stdev(response_times) / statistics.mean(response_times)
                          if statistics.mean(response_times) > 0 else 0)
        cache_ratio_cv = (statistics.stdev(cache_ratios) / statistics.mean(cache_ratios)
                         if statistics.mean(cache_ratios) > 0 else 0)
        
        # Classify stability
        if response_time_cv < 0.2 and cache_ratio_cv < 0.1:
            return 'stable'
        elif response_time_cv < 0.5 and cache_ratio_cv < 0.2:
            return 'moderate'
        else:
            return 'unstable'
    
    async def export_metrics(self, file_path: str, format: str = 'json'):
        """Export metrics to file"""
        
        metrics_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_info': {
                'monitoring_enabled': self.monitoring_enabled,
                'collection_interval': self.collection_interval,
                'retention_days': self.retention_days
            },
            'performance_history': [
                {
                    'timestamp': timestamp.isoformat(),
                    'metrics': asdict(metrics)
                }
                for timestamp, metrics in self.performance_history
            ],
            'capacity_history': [
                {
                    'timestamp': timestamp.isoformat(),
                    'metrics': asdict(metrics)
                }
                for timestamp, metrics in self.capacity_history
            ],
            'quality_history': [
                {
                    'timestamp': timestamp.isoformat(),
                    'metrics': asdict(metrics)
                }
                for timestamp, metrics in self.quality_history
            ],
            'alerts': [
                {
                    'level': alert.level,
                    'category': alert.category,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'details': alert.details,
                    'resolved': alert.resolved,
                    'resolution_timestamp': (
                        alert.resolution_timestamp.isoformat()
                        if alert.resolution_timestamp else None
                    )
                }
                for alert in self.alert_history
            ]
        }
        
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"Metrics exported to {file_path}")
    
    async def cleanup_old_data(self, days: int = None):
        """Clean up old metrics data"""
        
        cleanup_days = days or self.retention_days
        cutoff_time = datetime.now() - timedelta(days=cleanup_days)
        
        # Clean up performance history
        self.performance_history = deque([
            (timestamp, metrics) for timestamp, metrics in self.performance_history
            if timestamp > cutoff_time
        ], maxlen=self.performance_history.maxlen)
        
        # Clean up capacity history
        self.capacity_history = deque([
            (timestamp, metrics) for timestamp, metrics in self.capacity_history
            if timestamp > cutoff_time
        ], maxlen=self.capacity_history.maxlen)
        
        # Clean up quality history
        self.quality_history = deque([
            (timestamp, metrics) for timestamp, metrics in self.quality_history
            if timestamp > cutoff_time
        ], maxlen=self.quality_history.maxlen)
        
        # Clean up old alerts (keep resolved alerts for some time)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time or not alert.resolved
        ]
        
        print(f"Cleaned up metrics data older than {cleanup_days} days")