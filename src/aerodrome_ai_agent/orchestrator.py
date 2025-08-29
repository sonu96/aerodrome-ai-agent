"""
Aerodrome AI Agent Orchestrator

The main orchestrator that coordinates all components of the agent:
- Brain (LangGraph cognitive system)
- Memory (Mem0 learning system)  
- CDP (Coinbase Developer Platform SDK)
- Monitoring (health checks and metrics)

Provides:
- Graceful startup/shutdown
- Component lifecycle management
- Health monitoring and emergency stops
- Configuration management
- Different operating modes (simulation, testnet, mainnet)
- Safety checks and risk management
"""

import asyncio
import signal
import sys
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import threading

from .utils.logger import get_logger, initialize_logging, log_performance, log_security_event
from .config.settings import Settings
from .brain import AerodromeBrain, BrainConfig
from .memory import MemorySystem, MemoryConfig
from .cdp import CDPManager, CDPConfig
from .monitoring import SystemMonitor, AlertLevel, HealthStatus, measure_async_performance


logger = get_logger("orchestrator")


class OperationMode(Enum):
    """Agent operation modes"""
    SIMULATION = "simulation"  # No real transactions, mock everything
    TESTNET = "testnet"       # Use testnet for real testing
    MAINNET = "mainnet"       # Production mode with real funds
    MANUAL = "manual"         # Manual approval for all actions


class AgentState(Enum):
    """Agent state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class OrchestatorConfig:
    """Configuration for the orchestrator"""
    operation_mode: OperationMode = OperationMode.SIMULATION
    auto_start: bool = False
    max_startup_time: int = 60  # seconds
    max_shutdown_time: int = 30  # seconds
    health_check_interval: int = 30  # seconds
    enable_emergency_stop: bool = True
    max_consecutive_errors: int = 5
    error_recovery_delay: int = 30  # seconds
    backup_interval: int = 3600  # seconds (1 hour)


class AerodromeOrchestrator:
    """
    Main orchestrator for the Aerodrome AI Agent
    
    Coordinates all components and manages the complete lifecycle of the agent,
    including startup, operation, monitoring, and shutdown procedures.
    """
    
    def __init__(self, settings: Settings, config: OrchestatorConfig = None):
        self.settings = settings
        self.config = config or OrchestatorConfig()
        
        # Initialize logging first
        self.logger_system = initialize_logging(settings)
        
        # State management
        self.state = AgentState.STOPPED
        self.start_time: Optional[datetime] = None
        self.last_error: Optional[Exception] = None
        self.consecutive_errors = 0
        self.emergency_stop_active = False
        
        # Component instances
        self.monitor: Optional[SystemMonitor] = None
        self.memory_system: Optional[MemorySystem] = None
        self.cdp_manager: Optional[CDPManager] = None
        self.brain: Optional[AerodromeBrain] = None
        
        # Runtime tracking
        self.cycle_count = 0
        self.last_cycle_time: Optional[datetime] = None
        self.performance_stats: Dict[str, Any] = {}
        
        # Async control
        self.main_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Signal handling
        self._setup_signal_handlers()
        
        logger.info("Aerodrome Orchestrator initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Signals not available (e.g., in threads)
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        if self.main_task:
            self.main_task.cancel()
        self.shutdown_event.set()
    
    async def start(self) -> bool:
        """
        Start the Aerodrome AI Agent
        
        Returns:
            True if startup successful, False otherwise
        """
        if self.state != AgentState.STOPPED:
            logger.warning(f"Cannot start agent in state: {self.state}")
            return False
        
        logger.info("Starting Aerodrome AI Agent...")
        self.state = AgentState.STARTING
        self.start_time = datetime.now()
        
        try:
            # Initialize components in order
            startup_start = time.time()
            
            # 1. Initialize monitoring system
            await self._initialize_monitoring()
            
            # 2. Initialize memory system
            await self._initialize_memory_system()
            
            # 3. Initialize CDP manager
            await self._initialize_cdp_manager()
            
            # 4. Initialize brain
            await self._initialize_brain()
            
            # 5. Perform safety checks
            if not await self._perform_safety_checks():
                raise RuntimeError("Safety checks failed")
            
            # 6. Start main operation
            self.state = AgentState.RUNNING
            self.main_task = asyncio.create_task(self._main_operation_loop())
            
            startup_duration = time.time() - startup_start
            log_performance("agent_startup", startup_duration)
            
            logger.info(f"Aerodrome AI Agent started successfully in {startup_duration:.2f}s")
            logger.info(f"Operation mode: {self.config.operation_mode.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            self.last_error = e
            self.state = AgentState.ERROR
            await self._cleanup_on_error()
            return False
    
    async def stop(self, emergency: bool = False) -> bool:
        """
        Stop the Aerodrome AI Agent
        
        Args:
            emergency: Whether this is an emergency stop
            
        Returns:
            True if shutdown successful, False otherwise
        """
        if self.state == AgentState.STOPPED:
            return True
        
        logger.info(f"Stopping Aerodrome AI Agent (emergency={emergency})")
        
        if emergency:
            self.state = AgentState.EMERGENCY_STOP
            self.emergency_stop_active = True
        else:
            self.state = AgentState.STOPPING
        
        shutdown_start = time.time()
        
        try:
            # Cancel main operation
            if self.main_task and not self.main_task.done():
                self.main_task.cancel()
                try:
                    await asyncio.wait_for(self.main_task, timeout=10)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Shutdown components in reverse order
            await self._shutdown_brain()
            await self._shutdown_cdp_manager()
            await self._shutdown_memory_system()
            await self._shutdown_monitoring()
            
            self.state = AgentState.STOPPED
            
            shutdown_duration = time.time() - shutdown_start
            log_performance("agent_shutdown", shutdown_duration)
            
            logger.info(f"Aerodrome AI Agent stopped successfully in {shutdown_duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.state = AgentState.ERROR
            return False
    
    async def pause(self) -> bool:
        """Pause agent operations"""
        if self.state != AgentState.RUNNING:
            logger.warning(f"Cannot pause agent in state: {self.state}")
            return False
        
        logger.info("Pausing agent operations")
        self.state = AgentState.PAUSED
        
        # Stop brain continuous operation
        if self.brain:
            self.brain.stop()
        
        return True
    
    async def resume(self) -> bool:
        """Resume agent operations"""
        if self.state != AgentState.PAUSED:
            logger.warning(f"Cannot resume agent in state: {self.state}")
            return False
        
        logger.info("Resuming agent operations")
        self.state = AgentState.RUNNING
        
        # Restart brain operation if needed
        if self.brain and not self.brain.running:
            brain_config = self.settings.get_brain_config()
            asyncio.create_task(
                self.brain.start_continuous_operation(brain_config.observation_interval)
            )
        
        return True
    
    async def emergency_stop(self) -> bool:
        """Trigger emergency stop procedures"""
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        log_security_event("emergency_stop", {
            "timestamp": datetime.now().isoformat(),
            "state": self.state.value,
            "cycle_count": self.cycle_count
        })
        
        # Immediate shutdown
        return await self.stop(emergency=True)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        status = {
            "state": self.state.value,
            "operation_mode": self.config.operation_mode.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "cycle_count": self.cycle_count,
            "last_cycle_time": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "consecutive_errors": self.consecutive_errors,
            "emergency_stop_active": self.emergency_stop_active,
            "components": {
                "monitoring": self.monitor is not None,
                "memory_system": self.memory_system is not None,
                "cdp_manager": self.cdp_manager is not None,
                "brain": self.brain is not None
            }
        }
        
        # Add component-specific status
        if self.monitor:
            try:
                status["health"] = self.monitor.get_system_health()
                status["performance"] = self.monitor.get_performance_stats()
                status["alerts"] = len(self.monitor.get_alerts(resolved=False))
            except Exception as e:
                logger.error(f"Error getting monitor status: {e}")
        
        if self.memory_system:
            try:
                status["memory_stats"] = await self.memory_system.get_memory_stats()
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
        
        if self.cdp_manager:
            try:
                status["wallet_info"] = await self.cdp_manager.get_wallet_info()
            except Exception as e:
                logger.error(f"Error getting wallet info: {e}")
        
        if self.brain:
            try:
                status["brain_status"] = self.brain.get_status()
            except Exception as e:
                logger.error(f"Error getting brain status: {e}")
        
        return status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "cycle_count": self.cycle_count,
            "consecutive_errors": self.consecutive_errors,
            "state": self.state.value
        }
        
        if self.monitor:
            try:
                metrics.update(self.monitor.get_performance_stats())
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
        
        return metrics
    
    async def _initialize_monitoring(self):
        """Initialize the monitoring system"""
        logger.info("Initializing monitoring system...")
        
        self.monitor = SystemMonitor(self.settings)
        
        # Register emergency callback
        self.monitor.register_emergency_callback(self._handle_emergency)
        
        # Start monitoring
        await self.monitor.start_monitoring()
        
        logger.info("Monitoring system initialized")
    
    async def _initialize_memory_system(self):
        """Initialize the memory system"""
        logger.info("Initializing memory system...")
        
        memory_config = self.settings.get_memory_config()
        self.memory_system = MemorySystem(memory_config)
        
        # Register health check
        if self.monitor:
            self.monitor.add_health_check(
                name="memory_system",
                check_function=self._check_memory_health,
                interval=60,
                critical=True
            )
        
        logger.info("Memory system initialized")
    
    async def _initialize_cdp_manager(self):
        """Initialize the CDP manager"""
        logger.info("Initializing CDP manager...")
        
        cdp_config = self.settings.get_cdp_config()
        
        # Override for simulation mode
        if self.config.operation_mode == OperationMode.SIMULATION:
            cdp_config.network_id = "base-sepolia"  # Use testnet for simulation
        
        self.cdp_manager = CDPManager(cdp_config)
        
        # Initialize wallet
        wallet_info = await self.cdp_manager.initialize_wallet()
        logger.info(f"Wallet initialized: {wallet_info.get('default_address', 'Unknown')}")
        
        # Register health check
        if self.monitor:
            self.monitor.add_health_check(
                name="cdp_manager",
                check_function=self._check_cdp_health,
                interval=30,
                critical=True
            )
        
        logger.info("CDP manager initialized")
    
    async def _initialize_brain(self):
        """Initialize the brain system"""
        logger.info("Initializing brain system...")
        
        brain_config = self.settings.get_brain_config()
        
        # Override settings for different modes
        if self.config.operation_mode == OperationMode.SIMULATION:
            brain_config.simulation_mode = True
            brain_config.confidence_threshold = 0.5  # Lower threshold for testing
        elif self.config.operation_mode == OperationMode.MANUAL:
            brain_config.require_manual_approval = True
        
        self.brain = AerodromeBrain(
            config=brain_config,
            memory_system=self.memory_system,
            cdp_manager=self.cdp_manager
        )
        
        # Register health check
        if self.monitor:
            self.monitor.add_health_check(
                name="brain",
                check_function=self._check_brain_health,
                interval=30,
                critical=True
            )
        
        logger.info("Brain system initialized")
    
    async def _perform_safety_checks(self) -> bool:
        """Perform comprehensive safety checks before operation"""
        logger.info("Performing safety checks...")
        
        try:
            # Check wallet balance
            if self.cdp_manager:
                balances = await self.cdp_manager.get_balances()
                if balances.get("eth", 0) < 0.01:  # Minimum ETH for gas
                    logger.warning("Low ETH balance for gas fees")
                    if self.config.operation_mode == OperationMode.MAINNET:
                        return False
            
            # Check memory system
            if self.memory_system:
                stats = await self.memory_system.get_memory_stats()
                if stats.get("error"):
                    logger.error(f"Memory system error: {stats['error']}")
                    return False
            
            # Check brain configuration
            if self.brain:
                brain_status = self.brain.get_status()
                if not brain_status.get("config"):
                    logger.error("Brain configuration invalid")
                    return False
            
            # Check monitoring system
            if self.monitor:
                health = self.monitor.get_system_health()
                if health.get("overall_status") == HealthStatus.CRITICAL.value:
                    logger.error("Critical system health issues detected")
                    return False
            
            logger.info("All safety checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return False
    
    async def _main_operation_loop(self):
        """Main operation loop for the agent"""
        logger.info("Starting main operation loop")
        
        brain_config = self.settings.get_brain_config()
        
        while self.state == AgentState.RUNNING and not self.shutdown_event.is_set():
            try:
                cycle_start = time.time()
                
                # Run brain cycle
                result = await measure_async_performance(
                    self.monitor,
                    "brain_cycle",
                    self.brain.run_cycle()
                )
                
                self.cycle_count += 1
                self.last_cycle_time = datetime.now()
                self.consecutive_errors = 0
                
                # Update component status
                if self.monitor:
                    self.monitor.update_component_status("orchestrator", {
                        "last_cycle": self.last_cycle_time.isoformat(),
                        "cycle_count": self.cycle_count,
                        "status": "running"
                    })
                
                # Log cycle results
                execution_status = result.get("execution_status", "unknown")
                logger.info(f"Cycle {self.cycle_count} completed: {execution_status}")
                
                # Check for errors in brain cycle
                if result.get("errors"):
                    logger.warning(f"Brain cycle had {len(result['errors'])} errors")
                
                # Wait for next cycle
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, brain_config.observation_interval - cycle_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info("Main operation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main operation loop: {e}")
                self.consecutive_errors += 1
                self.last_error = e
                
                # Handle too many consecutive errors
                if self.consecutive_errors >= self.config.max_consecutive_errors:
                    logger.critical(f"Too many consecutive errors ({self.consecutive_errors}), stopping")
                    await self.emergency_stop()
                    break
                
                # Wait before retrying
                await asyncio.sleep(self.config.error_recovery_delay)
        
        logger.info("Main operation loop ended")
    
    async def _handle_emergency(self, reason: str, details: Dict[str, Any]):
        """Handle emergency conditions from monitoring"""
        logger.critical(f"Emergency condition detected: {reason}")
        
        log_security_event("monitoring_emergency", {
            "reason": reason,
            "details": details
        })
        
        # Trigger emergency stop
        await self.emergency_stop()
    
    def _check_memory_health(self, check_name: str) -> Dict[str, Any]:
        """Health check for memory system"""
        if not self.memory_system:
            return {
                "status": HealthStatus.CRITICAL,
                "message": "Memory system not initialized"
            }
        
        try:
            # This would be implemented to check memory system health
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Memory system operational"
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Memory system error: {str(e)}"
            }
    
    def _check_cdp_health(self, check_name: str) -> Dict[str, Any]:
        """Health check for CDP manager"""
        if not self.cdp_manager:
            return {
                "status": HealthStatus.CRITICAL,
                "message": "CDP manager not initialized"
            }
        
        try:
            # This would implement actual CDP health checks
            return {
                "status": HealthStatus.HEALTHY,
                "message": "CDP manager operational"
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"CDP manager error: {str(e)}"
            }
    
    def _check_brain_health(self, check_name: str) -> Dict[str, Any]:
        """Health check for brain system"""
        if not self.brain:
            return {
                "status": HealthStatus.CRITICAL,
                "message": "Brain not initialized"
            }
        
        try:
            brain_status = self.brain.get_status()
            if brain_status.get("running"):
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Brain operational"
                }
            else:
                return {
                    "status": HealthStatus.WARNING,
                    "message": "Brain not running"
                }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Brain error: {str(e)}"
            }
    
    async def _cleanup_on_error(self):
        """Cleanup resources when error occurs during startup"""
        logger.info("Cleaning up after startup error")
        
        if self.brain:
            try:
                await self._shutdown_brain()
            except:
                pass
        
        if self.cdp_manager:
            try:
                await self._shutdown_cdp_manager()
            except:
                pass
        
        if self.memory_system:
            try:
                await self._shutdown_memory_system()
            except:
                pass
        
        if self.monitor:
            try:
                await self._shutdown_monitoring()
            except:
                pass
    
    async def _shutdown_brain(self):
        """Shutdown brain system"""
        if self.brain:
            logger.info("Shutting down brain system...")
            self.brain.stop()
            self.brain = None
    
    async def _shutdown_cdp_manager(self):
        """Shutdown CDP manager"""
        if self.cdp_manager:
            logger.info("Shutting down CDP manager...")
            await self.cdp_manager.close()
            self.cdp_manager = None
    
    async def _shutdown_memory_system(self):
        """Shutdown memory system"""
        if self.memory_system:
            logger.info("Shutting down memory system...")
            await self.memory_system.close()
            self.memory_system = None
    
    async def _shutdown_monitoring(self):
        """Shutdown monitoring system"""
        if self.monitor:
            logger.info("Shutting down monitoring system...")
            self.monitor.stop_monitoring()
            self.monitor = None
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if self.state == AgentState.RUNNING:
                asyncio.create_task(self.stop())
        except:
            pass


class OrchestratorManager:
    """Manager class for running the orchestrator in different contexts"""
    
    def __init__(self, settings: Settings, config: OrchestatorConfig = None):
        self.settings = settings
        self.config = config or OrchestatorConfig()
        self.orchestrator: Optional[AerodromeOrchestrator] = None
    
    async def run(self, operation_mode: OperationMode = None, max_cycles: int = None):
        """Run the orchestrator"""
        if operation_mode:
            self.config.operation_mode = operation_mode
        
        self.orchestrator = AerodromeOrchestrator(self.settings, self.config)
        
        try:
            # Start the orchestrator
            success = await self.orchestrator.start()
            if not success:
                raise RuntimeError("Failed to start orchestrator")
            
            # Run for specified cycles or indefinitely
            if max_cycles:
                logger.info(f"Running for {max_cycles} cycles")
                for i in range(max_cycles):
                    await asyncio.sleep(10)  # Wait between checks
                    status = await self.orchestrator.get_status()
                    if status["state"] != "running":
                        break
            else:
                # Run until stopped
                try:
                    while True:
                        await asyncio.sleep(1)
                        status = await self.orchestrator.get_status()
                        if status["state"] not in ["running", "paused"]:
                            break
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt")
        
        finally:
            # Ensure graceful shutdown
            if self.orchestrator:
                await self.orchestrator.stop()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        if self.orchestrator:
            return await self.orchestrator.get_status()
        else:
            return {"state": "not_initialized"}
    
    async def stop(self):
        """Stop the orchestrator"""
        if self.orchestrator:
            await self.orchestrator.stop()