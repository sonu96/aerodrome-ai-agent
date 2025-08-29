"""
Aerodrome Pool Monitoring System

Advanced pool monitoring system that provides:
- Real-time pool state tracking
- TVL, volume, and APR monitoring  
- Fee tracking and analysis
- Liquidity depth analysis
- Performance metrics and alerts
- Historical data collection

Integrates with AerodromeClient for real-time data access.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque, defaultdict
from statistics import mean, median, stdev

import structlog
from .aerodrome_client import AerodromeClient, PoolInfo, PoolType

logger = structlog.get_logger(__name__)


class AlertType(Enum):
    """Types of pool monitoring alerts"""
    TVL_CHANGE = "tvl_change"
    VOLUME_SPIKE = "volume_spike"  
    LIQUIDITY_DROP = "liquidity_drop"
    FEE_ANOMALY = "fee_anomaly"
    APR_CHANGE = "apr_change"
    PRICE_IMPACT = "price_impact"
    IL_RISK = "impermanent_loss_risk"


class MonitoringStatus(Enum):
    """Pool monitoring status"""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class PoolMetrics:
    """Comprehensive pool metrics at a point in time"""
    timestamp: datetime
    pool_address: str
    tvl_usd: float
    volume_24h: float
    fees_24h: float
    apr: float
    token0_reserve: float
    token1_reserve: float
    token0_price: float
    token1_price: float
    liquidity_depth: Dict[str, float] = field(default_factory=dict)
    active_positions: int = 0
    unique_traders_24h: int = 0


@dataclass
class PoolAlert:
    """Pool monitoring alert"""
    timestamp: datetime
    pool_address: str
    alert_type: AlertType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    current_value: float
    previous_value: Optional[float] = None
    threshold_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for pool monitoring"""
    # Update intervals
    update_interval: int = 30  # seconds
    metrics_retention_hours: int = 168  # 7 days
    
    # TVL monitoring
    tvl_change_threshold: float = 0.10  # 10% change
    tvl_alert_enabled: bool = True
    
    # Volume monitoring  
    volume_spike_multiplier: float = 3.0  # 3x normal volume
    volume_alert_enabled: bool = True
    
    # Liquidity monitoring
    liquidity_drop_threshold: float = 0.20  # 20% drop
    liquidity_alert_enabled: bool = True
    
    # Fee monitoring
    fee_anomaly_threshold: float = 2.0  # 2 std devs from mean
    fee_alert_enabled: bool = True
    
    # APR monitoring
    apr_change_threshold: float = 0.15  # 15% change
    apr_alert_enabled: bool = True
    
    # Price impact monitoring
    max_price_impact_threshold: float = 0.05  # 5%
    price_impact_alert_enabled: bool = True
    
    # Impermanent loss risk
    il_risk_threshold: float = 0.10  # 10% divergence
    il_risk_alert_enabled: bool = True


class PoolAnalyzer:
    """Analyzes pool data for patterns and anomalies"""
    
    def __init__(self, lookback_periods: int = 24):
        self.lookback_periods = lookback_periods
        
    def calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation) of values"""
        if len(values) < 2:
            return 0.0
        return stdev(values)
    
    def detect_trend(self, values: List[float], min_periods: int = 6) -> str:
        """Detect trend direction in values"""
        if len(values) < min_periods:
            return "insufficient_data"
        
        # Simple trend detection using linear fit
        recent = values[-min_periods:]
        if len(set(recent)) == 1:  # All values same
            return "sideways"
        
        # Calculate slope
        x = list(range(len(recent)))
        y = recent
        n = len(recent)
        
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        if slope > 0.01:
            return "upward"
        elif slope < -0.01:
            return "downward" 
        else:
            return "sideways"
    
    def calculate_impermanent_loss(
        self,
        initial_price_ratio: float,
        current_price_ratio: float
    ) -> float:
        """Calculate impermanent loss percentage"""
        if initial_price_ratio <= 0 or current_price_ratio <= 0:
            return 0.0
        
        price_change_ratio = current_price_ratio / initial_price_ratio
        
        # IL formula for constant product pools
        il = 2 * (price_change_ratio**0.5) / (1 + price_change_ratio) - 1
        return abs(il)
    
    def analyze_liquidity_distribution(self, metrics: List[PoolMetrics]) -> Dict[str, Any]:
        """Analyze liquidity distribution patterns"""
        if not metrics:
            return {}
        
        recent_metrics = metrics[-self.lookback_periods:]
        
        tvl_values = [m.tvl_usd for m in recent_metrics]
        volume_values = [m.volume_24h for m in recent_metrics]
        
        return {
            "avg_tvl": mean(tvl_values),
            "tvl_volatility": self.calculate_volatility(tvl_values),
            "tvl_trend": self.detect_trend(tvl_values),
            "avg_volume": mean(volume_values),
            "volume_volatility": self.calculate_volatility(volume_values),
            "volume_trend": self.detect_trend(volume_values),
            "volume_to_tvl_ratio": mean(volume_values) / mean(tvl_values) if mean(tvl_values) > 0 else 0
        }


class PoolMonitor:
    """
    Advanced pool monitoring system
    
    Provides real-time monitoring of Aerodrome pools with:
    - Continuous metrics collection
    - Anomaly detection and alerting
    - Historical data analysis
    - Performance insights
    """
    
    def __init__(
        self,
        aerodrome_client: AerodromeClient,
        config: Optional[MonitoringConfig] = None
    ):
        """
        Initialize pool monitor
        
        Args:
            aerodrome_client: AerodromeClient instance
            config: Optional monitoring configuration
        """
        self.client = aerodrome_client
        self.config = config or MonitoringConfig()
        self.analyzer = PoolAnalyzer()
        
        # Monitoring state
        self.monitored_pools: Set[str] = set()
        self.monitoring_status: Dict[str, MonitoringStatus] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Data storage
        self.pool_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.metrics_retention_hours * 2))
        self.pool_alerts: Dict[str, List[PoolAlert]] = defaultdict(list)
        self.pool_info_cache: Dict[str, PoolInfo] = {}
        
        # Callbacks
        self.alert_callbacks: List[Callable[[PoolAlert], None]] = []
        self.metrics_callbacks: List[Callable[[str, PoolMetrics], None]] = []
        
        logger.info("Pool monitor initialized",
                   update_interval=config.update_interval if config else 30)
    
    def add_alert_callback(self, callback: Callable[[PoolAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
        logger.debug("Added alert callback")
    
    def add_metrics_callback(self, callback: Callable[[str, PoolMetrics], None]):
        """Add callback for metrics updates"""
        self.metrics_callbacks.append(callback)
        logger.debug("Added metrics callback")
    
    async def start_monitoring(self, pool_address: str):
        """
        Start monitoring a specific pool
        
        Args:
            pool_address: Pool contract address to monitor
        """
        if pool_address in self.monitored_pools:
            logger.warning("Pool already being monitored", pool=pool_address)
            return
        
        try:
            # Get initial pool info
            pool_info = await self.client.get_pool_info(pool_address)
            self.pool_info_cache[pool_address] = pool_info
            
            # Add to monitored pools
            self.monitored_pools.add(pool_address)
            self.monitoring_status[pool_address] = MonitoringStatus.ACTIVE
            
            # Start monitoring task
            task = asyncio.create_task(self._monitor_pool(pool_address))
            self.monitoring_tasks[pool_address] = task
            
            logger.info("Started monitoring pool",
                       pool=pool_address,
                       symbol0=pool_info.token0.symbol,
                       symbol1=pool_info.token1.symbol)
            
        except Exception as e:
            logger.error("Failed to start monitoring pool",
                        pool=pool_address,
                        error=str(e))
            self.monitoring_status[pool_address] = MonitoringStatus.ERROR
    
    async def stop_monitoring(self, pool_address: str):
        """
        Stop monitoring a specific pool
        
        Args:
            pool_address: Pool contract address to stop monitoring
        """
        if pool_address not in self.monitored_pools:
            logger.warning("Pool not being monitored", pool=pool_address)
            return
        
        # Cancel monitoring task
        if pool_address in self.monitoring_tasks:
            self.monitoring_tasks[pool_address].cancel()
            try:
                await self.monitoring_tasks[pool_address]
            except asyncio.CancelledError:
                pass
            del self.monitoring_tasks[pool_address]
        
        # Update status
        self.monitored_pools.remove(pool_address)
        self.monitoring_status[pool_address] = MonitoringStatus.STOPPED
        
        logger.info("Stopped monitoring pool", pool=pool_address)
    
    async def pause_monitoring(self, pool_address: str):
        """Pause monitoring for a pool"""
        if pool_address in self.monitoring_status:
            self.monitoring_status[pool_address] = MonitoringStatus.PAUSED
            logger.info("Paused monitoring pool", pool=pool_address)
    
    async def resume_monitoring(self, pool_address: str):
        """Resume monitoring for a paused pool"""
        if pool_address in self.monitoring_status:
            self.monitoring_status[pool_address] = MonitoringStatus.ACTIVE
            logger.info("Resumed monitoring pool", pool=pool_address)
    
    async def _monitor_pool(self, pool_address: str):
        """
        Main monitoring loop for a pool
        
        Args:
            pool_address: Pool address to monitor
        """
        logger.debug("Starting pool monitoring loop", pool=pool_address)
        
        while pool_address in self.monitored_pools:
            try:
                # Check if monitoring is paused
                if self.monitoring_status.get(pool_address) != MonitoringStatus.ACTIVE:
                    await asyncio.sleep(self.config.update_interval)
                    continue
                
                # Collect metrics
                metrics = await self._collect_pool_metrics(pool_address)
                if metrics:
                    # Store metrics
                    self.pool_metrics[pool_address].append(metrics)
                    
                    # Run analysis and check for alerts
                    await self._analyze_pool_metrics(pool_address, metrics)
                    
                    # Notify callbacks
                    for callback in self.metrics_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(pool_address, metrics)
                            else:
                                callback(pool_address, metrics)
                        except Exception as e:
                            logger.error("Metrics callback error", error=str(e))
                
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                logger.debug("Pool monitoring cancelled", pool=pool_address)
                break
            except Exception as e:
                logger.error("Error in pool monitoring loop",
                           pool=pool_address,
                           error=str(e))
                self.monitoring_status[pool_address] = MonitoringStatus.ERROR
                await asyncio.sleep(self.config.update_interval * 2)  # Back off on error
    
    async def _collect_pool_metrics(self, pool_address: str) -> Optional[PoolMetrics]:
        """Collect current pool metrics"""
        try:
            # Get current pool info
            pool_info = await self.client.get_pool_info(pool_address, use_cache=False)
            
            # Get token prices
            token_prices = await self.client.get_token_prices([
                pool_info.token0.address,
                pool_info.token1.address
            ])
            
            # Get additional analytics
            analytics = await self.client.get_pool_analytics(pool_address, "1h")
            
            metrics = PoolMetrics(
                timestamp=datetime.now(),
                pool_address=pool_address,
                tvl_usd=pool_info.tvl_usd,
                volume_24h=pool_info.volume_24h,
                fees_24h=pool_info.fees_24h,
                apr=pool_info.apr,
                token0_reserve=pool_info.reserves["token0"],
                token1_reserve=pool_info.reserves["token1"],
                token0_price=token_prices.get(pool_info.token0.address, 0.0),
                token1_price=token_prices.get(pool_info.token1.address, 0.0),
                liquidity_depth=analytics.get("liquidity_depth", {}),
                active_positions=analytics.get("active_positions", 0),
                unique_traders_24h=analytics.get("unique_traders_24h", 0)
            )
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect pool metrics",
                        pool=pool_address,
                        error=str(e))
            return None
    
    async def _analyze_pool_metrics(self, pool_address: str, current_metrics: PoolMetrics):
        """Analyze pool metrics and generate alerts"""
        historical_metrics = list(self.pool_metrics[pool_address])
        if len(historical_metrics) < 2:
            return  # Need at least 2 data points
        
        previous_metrics = historical_metrics[-2]  # Get previous metrics
        
        # Check TVL changes
        if self.config.tvl_alert_enabled:
            await self._check_tvl_changes(pool_address, current_metrics, previous_metrics)
        
        # Check volume spikes
        if self.config.volume_alert_enabled:
            await self._check_volume_spikes(pool_address, current_metrics, historical_metrics)
        
        # Check liquidity drops
        if self.config.liquidity_alert_enabled:
            await self._check_liquidity_drops(pool_address, current_metrics, previous_metrics)
        
        # Check fee anomalies
        if self.config.fee_alert_enabled:
            await self._check_fee_anomalies(pool_address, current_metrics, historical_metrics)
        
        # Check APR changes
        if self.config.apr_alert_enabled:
            await self._check_apr_changes(pool_address, current_metrics, previous_metrics)
        
        # Check impermanent loss risk
        if self.config.il_risk_alert_enabled:
            await self._check_il_risk(pool_address, current_metrics, historical_metrics)
    
    async def _check_tvl_changes(self, pool_address: str, current: PoolMetrics, previous: PoolMetrics):
        """Check for significant TVL changes"""
        if previous.tvl_usd == 0:
            return
        
        change_ratio = abs(current.tvl_usd - previous.tvl_usd) / previous.tvl_usd
        
        if change_ratio >= self.config.tvl_change_threshold:
            severity = "high" if change_ratio >= 0.25 else "medium"
            direction = "increased" if current.tvl_usd > previous.tvl_usd else "decreased"
            
            alert = PoolAlert(
                timestamp=current.timestamp,
                pool_address=pool_address,
                alert_type=AlertType.TVL_CHANGE,
                severity=severity,
                message=f"TVL {direction} by {change_ratio:.1%}",
                current_value=current.tvl_usd,
                previous_value=previous.tvl_usd,
                threshold_value=self.config.tvl_change_threshold
            )
            
            await self._send_alert(alert)
    
    async def _check_volume_spikes(self, pool_address: str, current: PoolMetrics, historical: List[PoolMetrics]):
        """Check for volume spikes"""
        if len(historical) < 24:  # Need at least 24 periods for baseline
            return
        
        recent_volumes = [m.volume_24h for m in historical[-24:]]
        avg_volume = mean(recent_volumes)
        
        if avg_volume == 0:
            return
        
        spike_ratio = current.volume_24h / avg_volume
        
        if spike_ratio >= self.config.volume_spike_multiplier:
            severity = "critical" if spike_ratio >= 5.0 else "high"
            
            alert = PoolAlert(
                timestamp=current.timestamp,
                pool_address=pool_address,
                alert_type=AlertType.VOLUME_SPIKE,
                severity=severity,
                message=f"Volume spike: {spike_ratio:.1f}x normal volume",
                current_value=current.volume_24h,
                previous_value=avg_volume,
                threshold_value=self.config.volume_spike_multiplier
            )
            
            await self._send_alert(alert)
    
    async def _check_liquidity_drops(self, pool_address: str, current: PoolMetrics, previous: PoolMetrics):
        """Check for liquidity drops"""
        if previous.tvl_usd == 0:
            return
        
        drop_ratio = (previous.tvl_usd - current.tvl_usd) / previous.tvl_usd
        
        if drop_ratio >= self.config.liquidity_drop_threshold:
            severity = "critical" if drop_ratio >= 0.5 else "high"
            
            alert = PoolAlert(
                timestamp=current.timestamp,
                pool_address=pool_address,
                alert_type=AlertType.LIQUIDITY_DROP,
                severity=severity,
                message=f"Liquidity dropped by {drop_ratio:.1%}",
                current_value=current.tvl_usd,
                previous_value=previous.tvl_usd,
                threshold_value=self.config.liquidity_drop_threshold
            )
            
            await self._send_alert(alert)
    
    async def _check_fee_anomalies(self, pool_address: str, current: PoolMetrics, historical: List[PoolMetrics]):
        """Check for fee anomalies using statistical analysis"""
        if len(historical) < 24:
            return
        
        recent_fees = [m.fees_24h for m in historical[-24:]]
        if not any(recent_fees):  # All zeros
            return
        
        mean_fees = mean(recent_fees)
        fee_std = stdev(recent_fees) if len(recent_fees) > 1 else 0
        
        if fee_std == 0:
            return
        
        z_score = abs(current.fees_24h - mean_fees) / fee_std
        
        if z_score >= self.config.fee_anomaly_threshold:
            severity = "high" if z_score >= 3.0 else "medium"
            anomaly_type = "high" if current.fees_24h > mean_fees else "low"
            
            alert = PoolAlert(
                timestamp=current.timestamp,
                pool_address=pool_address,
                alert_type=AlertType.FEE_ANOMALY,
                severity=severity,
                message=f"Unusually {anomaly_type} fees: {z_score:.1f}σ from mean",
                current_value=current.fees_24h,
                previous_value=mean_fees,
                threshold_value=self.config.fee_anomaly_threshold,
                metadata={"z_score": z_score}
            )
            
            await self._send_alert(alert)
    
    async def _check_apr_changes(self, pool_address: str, current: PoolMetrics, previous: PoolMetrics):
        """Check for significant APR changes"""
        if previous.apr == 0:
            return
        
        change_ratio = abs(current.apr - previous.apr) / previous.apr
        
        if change_ratio >= self.config.apr_change_threshold:
            severity = "medium" if change_ratio < 0.5 else "high"
            direction = "increased" if current.apr > previous.apr else "decreased"
            
            alert = PoolAlert(
                timestamp=current.timestamp,
                pool_address=pool_address,
                alert_type=AlertType.APR_CHANGE,
                severity=severity,
                message=f"APR {direction} by {change_ratio:.1%}",
                current_value=current.apr,
                previous_value=previous.apr,
                threshold_value=self.config.apr_change_threshold
            )
            
            await self._send_alert(alert)
    
    async def _check_il_risk(self, pool_address: str, current: PoolMetrics, historical: List[PoolMetrics]):
        """Check for impermanent loss risk"""
        if len(historical) < 24 or current.token0_price == 0 or current.token1_price == 0:
            return
        
        # Use 24 periods ago as baseline
        baseline = historical[-24]
        if baseline.token0_price == 0 or baseline.token1_price == 0:
            return
        
        initial_ratio = baseline.token0_price / baseline.token1_price
        current_ratio = current.token0_price / current.token1_price
        
        il_risk = self.analyzer.calculate_impermanent_loss(initial_ratio, current_ratio)
        
        if il_risk >= self.config.il_risk_threshold:
            severity = "critical" if il_risk >= 0.25 else "high"
            
            alert = PoolAlert(
                timestamp=current.timestamp,
                pool_address=pool_address,
                alert_type=AlertType.IL_RISK,
                severity=severity,
                message=f"High impermanent loss risk: {il_risk:.1%}",
                current_value=il_risk,
                threshold_value=self.config.il_risk_threshold,
                metadata={
                    "initial_price_ratio": initial_ratio,
                    "current_price_ratio": current_ratio
                }
            )
            
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: PoolAlert):
        """Send alert to all registered callbacks"""
        # Store alert
        self.pool_alerts[alert.pool_address].append(alert)
        
        # Limit stored alerts per pool
        if len(self.pool_alerts[alert.pool_address]) > 100:
            self.pool_alerts[alert.pool_address] = self.pool_alerts[alert.pool_address][-50:]
        
        logger.warning("Pool alert generated",
                      pool=alert.pool_address,
                      type=alert.alert_type.value,
                      severity=alert.severity,
                      message=alert.message)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error("Alert callback error", error=str(e))
    
    def get_pool_status(self, pool_address: str) -> Dict[str, Any]:
        """Get current monitoring status for a pool"""
        if pool_address not in self.monitored_pools:
            return {"status": "not_monitored"}
        
        metrics_count = len(self.pool_metrics[pool_address])
        alerts_count = len(self.pool_alerts[pool_address])
        recent_alerts = [a for a in self.pool_alerts[pool_address] 
                        if a.timestamp > datetime.now() - timedelta(hours=24)]
        
        latest_metrics = self.pool_metrics[pool_address][-1] if metrics_count > 0 else None
        
        return {
            "status": self.monitoring_status.get(pool_address, MonitoringStatus.ERROR).value,
            "metrics_count": metrics_count,
            "total_alerts": alerts_count,
            "alerts_24h": len(recent_alerts),
            "latest_update": latest_metrics.timestamp if latest_metrics else None,
            "pool_info": self.pool_info_cache.get(pool_address)
        }
    
    def get_pool_analysis(self, pool_address: str) -> Dict[str, Any]:
        """Get comprehensive analysis for a pool"""
        if pool_address not in self.pool_metrics:
            return {}
        
        metrics = list(self.pool_metrics[pool_address])
        if not metrics:
            return {}
        
        # Basic statistics
        analysis = self.analyzer.analyze_liquidity_distribution(metrics)
        
        # Recent performance
        if len(metrics) >= 24:
            recent_24h = metrics[-24:]
            analysis["recent_24h"] = {
                "avg_tvl": mean([m.tvl_usd for m in recent_24h]),
                "avg_volume": mean([m.volume_24h for m in recent_24h]),
                "avg_apr": mean([m.apr for m in recent_24h]),
                "total_fees": sum([m.fees_24h for m in recent_24h])
            }
        
        # Alert summary
        recent_alerts = [a for a in self.pool_alerts[pool_address] 
                        if a.timestamp > datetime.now() - timedelta(hours=24)]
        
        alert_summary = defaultdict(int)
        for alert in recent_alerts:
            alert_summary[alert.alert_type.value] += 1
        
        analysis["alert_summary_24h"] = dict(alert_summary)
        analysis["total_alerts"] = len(self.pool_alerts[pool_address])
        
        return analysis
    
    async def stop_all_monitoring(self):
        """Stop monitoring all pools"""
        pools_to_stop = list(self.monitored_pools)
        for pool_address in pools_to_stop:
            await self.stop_monitoring(pool_address)
        
        logger.info("Stopped all pool monitoring")
    
    def get_monitored_pools(self) -> List[Dict[str, Any]]:
        """Get list of all monitored pools with their status"""
        return [
            {
                "address": pool_address,
                "status": self.get_pool_status(pool_address)
            }
            for pool_address in self.monitored_pools
        ]


# ===== EXAMPLE USAGE =====

async def example_pool_monitoring():
    """Example usage of PoolMonitor"""
    from .aerodrome_client import AerodromeClient
    
    # Initialize client and monitor
    quicknode_url = "https://your-quicknode-endpoint.quiknode.pro/your-api-key/"
    
    async with AerodromeClient(quicknode_url) as client:
        
        # Create monitoring configuration
        config = MonitoringConfig(
            update_interval=60,  # Update every minute
            tvl_change_threshold=0.05,  # 5% TVL change alert
            volume_spike_multiplier=2.0,  # 2x volume spike alert
            liquidity_drop_threshold=0.15,  # 15% liquidity drop alert
        )
        
        # Create monitor
        monitor = PoolMonitor(client, config)
        
        # Add alert callback
        def alert_handler(alert: PoolAlert):
            print(f"🚨 ALERT: {alert.message}")
            print(f"   Pool: {alert.pool_address}")
            print(f"   Severity: {alert.severity}")
            print(f"   Type: {alert.alert_type.value}")
        
        monitor.add_alert_callback(alert_handler)
        
        # Start monitoring some popular pools
        popular_pools = await client.search_pools(min_tvl=1000000, limit=5)
        
        for pool in popular_pools:
            await monitor.start_monitoring(pool.address)
            print(f"Monitoring {pool.token0.symbol}/{pool.token1.symbol}")
        
        # Run monitoring for a while
        print("Monitoring started. Collecting data...")
        await asyncio.sleep(300)  # Monitor for 5 minutes
        
        # Get analysis for monitored pools
        for pool_address in monitor.monitored_pools:
            status = monitor.get_pool_status(pool_address)
            analysis = monitor.get_pool_analysis(pool_address)
            
            print(f"\nPool Analysis: {pool_address}")
            print(f"Status: {status['status']}")
            print(f"Metrics collected: {status['metrics_count']}")
            print(f"Alerts (24h): {status['alerts_24h']}")
            
            if analysis:
                print(f"Avg TVL: ${analysis.get('avg_tvl', 0):,.2f}")
                print(f"Avg Volume: ${analysis.get('avg_volume', 0):,.2f}")
                print(f"TVL Trend: {analysis.get('tvl_trend', 'unknown')}")
        
        # Stop all monitoring
        await monitor.stop_all_monitoring()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_pool_monitoring())