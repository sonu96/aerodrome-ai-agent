"""
System Monitoring and Health Checks for Aerodrome AI Agent

This module provides comprehensive monitoring capabilities including:
- System health checks
- Performance metrics collection
- Resource usage monitoring  
- Component status tracking
- Alert and notification management
- Emergency stop conditions
"""

import asyncio
import psutil
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from .utils.logger import get_logger
from .config.settings import Settings


logger = get_logger("monitoring")


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical" 
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable
    interval: int  # seconds
    timeout: int = 30  # seconds
    critical: bool = False
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class Alert:
    """Alert definition"""
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """
    Comprehensive system monitoring for the Aerodrome AI Agent
    
    Provides:
    - Health checks for all components
    - Performance metrics collection
    - Resource usage monitoring
    - Alert management
    - Emergency stop detection
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # State tracking
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: List[Alert] = []
        self.metrics_history: List[PerformanceMetric] = []
        self.component_status: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = {}
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        logger.info("System Monitor initialized")
    
    def _setup_default_health_checks(self):
        """Setup default system health checks"""
        
        # System resource checks
        self.add_health_check(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval=30,
            critical=False
        )
        
        self.add_health_check(
            name="memory_usage", 
            check_function=self._check_memory_usage,
            interval=30,
            critical=True
        )
        
        self.add_health_check(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval=60,
            critical=False
        )
        
        # Network connectivity
        self.add_health_check(
            name="network_connectivity",
            check_function=self._check_network_connectivity,
            interval=60,
            critical=True
        )
        
        # Component health checks (to be registered by components)
        self.add_health_check(
            name="brain_health",
            check_function=self._check_component_health,
            interval=30,
            critical=True
        )
        
        self.add_health_check(
            name="memory_system_health",
            check_function=self._check_memory_system_health,
            interval=60,
            critical=True
        )
        
        self.add_health_check(
            name="cdp_connection_health",
            check_function=self._check_cdp_health,
            interval=30,
            critical=True
        )
    
    def add_health_check(self, name: str, check_function: Callable, interval: int, 
                        critical: bool = False, timeout: int = 30, enabled: bool = True):
        """Add a custom health check"""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical,
            enabled=enabled
        )
        logger.info(f"Health check added: {name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Health check removed: {name}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def register_emergency_callback(self, callback: Callable):
        """Register callback for emergency situations"""
        self.emergency_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        logger.info("Starting system monitoring")
        
        # Start monitoring in separate thread to avoid blocking
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Monitoring loop started")
        
        while self.running:
            try:
                # Run health checks
                self._run_health_checks()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for emergency conditions
                self._check_emergency_conditions()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep between cycles
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _run_health_checks(self):
        """Run all enabled health checks"""
        current_time = datetime.now()
        
        for check in self.health_checks.values():
            if not check.enabled:
                continue
            
            # Check if it's time to run this check
            if (check.last_run is None or 
                (current_time - check.last_run).total_seconds() >= check.interval):
                
                try:
                    # Run the health check with timeout
                    result = self._run_single_health_check(check)
                    
                    check.last_run = current_time
                    check.last_result = result
                    
                    # Handle result
                    if result.get("status") == HealthStatus.HEALTHY:
                        check.consecutive_failures = 0
                        # Resolve any existing alerts for this check
                        self._resolve_alerts(check.name)
                    else:
                        check.consecutive_failures += 1
                        
                        # Create alert if threshold reached
                        if check.consecutive_failures >= check.max_failures:
                            self._create_alert(
                                component=check.name,
                                level=AlertLevel.CRITICAL if check.critical else AlertLevel.WARNING,
                                message=result.get("message", f"Health check {check.name} failing"),
                                metadata=result
                            )
                
                except Exception as e:
                    logger.error(f"Health check {check.name} failed: {e}")
                    check.consecutive_failures += 1
                    
                    if check.consecutive_failures >= check.max_failures:
                        self._create_alert(
                            component=check.name,
                            level=AlertLevel.CRITICAL,
                            message=f"Health check {check.name} exception: {str(e)}",
                            metadata={"exception": str(e)}
                        )
    
    def _run_single_health_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a single health check with timeout"""
        try:
            # Simple timeout mechanism
            start_time = time.time()
            result = check.check_function(check.name)
            duration = time.time() - start_time
            
            if duration > check.timeout:
                return {
                    "status": HealthStatus.WARNING,
                    "message": f"Health check {check.name} timed out",
                    "duration": duration
                }
            
            return result
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Health check {check.name} failed: {str(e)}",
                "exception": str(e)
            }
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        current_time = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self._add_metric("cpu_usage_percent", cpu_percent, "%", "system", current_time)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric("memory_usage_percent", memory.percent, "%", "system", current_time)
        self._add_metric("memory_available_gb", memory.available / (1024**3), "GB", "system", current_time)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric("disk_usage_percent", disk_percent, "%", "system", current_time)
        
        # Network metrics
        network = psutil.net_io_counters()
        self._add_metric("network_bytes_sent", network.bytes_sent, "bytes", "system", current_time)
        self._add_metric("network_bytes_recv", network.bytes_recv, "bytes", "system", current_time)
    
    def _add_metric(self, name: str, value: float, unit: str, component: str, 
                   timestamp: datetime, **metadata):
        """Add a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            component=component,
            metadata=metadata
        )
        
        self.metrics_history.append(metric)
    
    def track_operation(self, operation_name: str, duration: float):
        """Track operation performance"""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        
        self.operation_times[operation_name].append(duration)
        
        # Keep only recent measurements
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
        
        # Add as metric
        self._add_metric(
            f"operation_{operation_name}_duration",
            duration * 1000,  # Convert to ms
            "ms",
            "performance",
            datetime.now()
        )
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        # High memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            self._trigger_emergency("high_memory_usage", {
                "memory_percent": memory.percent
            })
        
        # Too many critical alerts
        critical_alerts = [a for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.resolved]
        if len(critical_alerts) > 5:
            self._trigger_emergency("too_many_critical_alerts", {
                "alert_count": len(critical_alerts)
            })
        
        # Component health failures
        critical_failures = sum(1 for check in self.health_checks.values()
                               if check.critical and check.consecutive_failures >= check.max_failures)
        if critical_failures > 2:
            self._trigger_emergency("multiple_critical_failures", {
                "failure_count": critical_failures
            })
    
    def _trigger_emergency(self, reason: str, details: Dict[str, Any]):
        """Trigger emergency procedures"""
        logger.critical(f"EMERGENCY TRIGGERED: {reason}")
        
        # Create emergency alert
        self._create_alert(
            component="system",
            level=AlertLevel.EMERGENCY,
            message=f"Emergency condition: {reason}",
            metadata=details
        )
        
        # Execute emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(reason, details)
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
    
    def _create_alert(self, component: str, level: AlertLevel, message: str, 
                     metadata: Dict[str, Any] = None):
        """Create a new alert"""
        alert = Alert(
            id=f"{component}_{int(time.time())}",
            level=level,
            component=component,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {level.value} - {component} - {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _resolve_alerts(self, component: str):
        """Resolve alerts for a component"""
        resolved_count = 0
        for alert in self.alerts:
            if alert.component == component and not alert.resolved:
                alert.resolved = True
                resolved_count += 1
        
        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} alerts for {component}")
    
    def _cleanup_old_data(self):
        """Cleanup old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Cleanup old metrics
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        # Cleanup old resolved alerts
        self.alerts = [
            a for a in self.alerts
            if not a.resolved or a.timestamp > cutoff_time
        ]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        current_time = datetime.now()
        
        # Calculate overall health
        healthy_checks = 0
        total_checks = 0
        critical_failures = 0
        
        for check in self.health_checks.values():
            if check.enabled:
                total_checks += 1
                if check.consecutive_failures == 0:
                    healthy_checks += 1
                elif check.critical and check.consecutive_failures >= check.max_failures:
                    critical_failures += 1
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = HealthStatus.CRITICAL
        elif healthy_checks / total_checks < 0.8:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "timestamp": current_time.isoformat(),
            "checks_passing": healthy_checks,
            "total_checks": total_checks,
            "critical_failures": critical_failures,
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "system_uptime": self._get_system_uptime(),
            "component_status": self.component_status
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        # Operation performance
        for operation, times in self.operation_times.items():
            if times:
                stats[f"{operation}_avg_ms"] = sum(times) / len(times) * 1000
                stats[f"{operation}_max_ms"] = max(times) * 1000
                stats[f"{operation}_min_ms"] = min(times) * 1000
        
        # Recent metrics
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        for metric in recent_metrics:
            stats[f"recent_{metric.name}"] = metric.value
        
        return stats
    
    def get_alerts(self, resolved: bool = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering"""
        alerts = self.alerts
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return [
            {
                "id": a.id,
                "level": a.level.value,
                "component": a.component,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "resolved": a.resolved,
                "metadata": a.metadata
            }
            for a in alerts
        ]
    
    def update_component_status(self, component: str, status: Dict[str, Any]):
        """Update status for a component"""
        self.component_status[component] = {
            **status,
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_system_uptime(self) -> float:
        """Get system uptime in seconds"""
        try:
            return time.time() - psutil.boot_time()
        except:
            return 0.0
    
    # Default health check implementations
    def _check_cpu_usage(self, check_name: str) -> Dict[str, Any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
            message = f"CPU usage critical: {cpu_percent}%"
        elif cpu_percent > 75:
            status = HealthStatus.WARNING
            message = f"CPU usage high: {cpu_percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent}%"
        
        return {
            "status": status,
            "message": message,
            "cpu_percent": cpu_percent
        }
    
    def _check_memory_usage(self, check_name: str) -> Dict[str, Any]:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        
        if memory.percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Memory usage critical: {memory.percent}%"
        elif memory.percent > 85:
            status = HealthStatus.WARNING
            message = f"Memory usage high: {memory.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory.percent}%"
        
        return {
            "status": status,
            "message": message,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3)
        }
    
    def _check_disk_usage(self, check_name: str) -> Dict[str, Any]:
        """Check disk usage"""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Disk usage critical: {disk_percent:.1f}%"
        elif disk_percent > 85:
            status = HealthStatus.WARNING
            message = f"Disk usage high: {disk_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {disk_percent:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "disk_percent": disk_percent
        }
    
    def _check_network_connectivity(self, check_name: str) -> Dict[str, Any]:
        """Check network connectivity"""
        # This is a simplified check - in practice you'd test specific endpoints
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Network connectivity OK"
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Network connectivity failed: {str(e)}"
            }
    
    def _check_component_health(self, check_name: str) -> Dict[str, Any]:
        """Check registered component health"""
        # This would be overridden by actual components
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Component health check not implemented"
        }
    
    def _check_memory_system_health(self, check_name: str) -> Dict[str, Any]:
        """Check memory system health"""
        # This would check the memory system status
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Memory system health check not implemented"
        }
    
    def _check_cdp_health(self, check_name: str) -> Dict[str, Any]:
        """Check CDP connection health"""
        # This would check CDP SDK connectivity
        return {
            "status": HealthStatus.HEALTHY,
            "message": "CDP health check not implemented"
        }


# Performance measurement decorator
def measure_performance(monitor: SystemMonitor, operation_name: str):
    """Decorator to measure operation performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.track_operation(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.track_operation(f"{operation_name}_error", duration)
                raise
        return wrapper
    return decorator


async def measure_async_performance(monitor: SystemMonitor, operation_name: str, coro):
    """Measure async operation performance"""
    start_time = time.time()
    try:
        result = await coro
        duration = time.time() - start_time
        monitor.track_operation(operation_name, duration)
        return result
    except Exception as e:
        duration = time.time() - start_time
        monitor.track_operation(f"{operation_name}_error", duration)
        raise