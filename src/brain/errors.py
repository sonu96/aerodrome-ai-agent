"""
Error handling and recovery strategies for the Aerodrome Brain.

This module implements comprehensive error handling including:
- Error classification and categorization
- Context-aware recovery strategies
- Circuit breaker patterns
- Graceful degradation
- Emergency stop mechanisms
- Error learning and adaptation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque

from .state import BrainState, BrainConfig


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    TRANSACTION = "transaction"
    VALIDATION = "validation"
    CALCULATION = "calculation"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    SLIPPAGE = "slippage"
    GAS = "gas"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Structured error information."""
    timestamp: datetime
    error_type: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""
    name: str
    max_retries: int
    base_delay: float
    max_delay: float
    exponential_backoff: bool
    jitter: bool
    circuit_breaker_threshold: int
    recovery_function: Optional[Callable] = None


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class BrainErrorHandler:
    """
    Comprehensive error handling system for the Aerodrome Brain.
    
    Provides:
    - Error classification and categorization
    - Context-aware recovery strategies
    - Circuit breaker patterns for external services
    - Graceful degradation mechanisms
    - Error learning and adaptation
    - Emergency stop capabilities
    """

    def __init__(self, config: BrainConfig):
        """Initialize the error handler."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.error_patterns = defaultdict(list)
        
        # Circuit breakers for external services
        self.circuit_breakers = {
            'cdp_api': CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'memory_system': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'external_data': CircuitBreaker(failure_threshold=5, recovery_timeout=120)
        }
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Emergency state tracking
        self.emergency_stop_active = False
        self.emergency_triggers = 0
        self.last_emergency_time = None

    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, RecoveryStrategy]:
        """Initialize recovery strategies for different error categories."""
        
        strategies = {
            ErrorCategory.NETWORK: RecoveryStrategy(
                name="network_retry",
                max_retries=3,
                base_delay=2.0,
                max_delay=30.0,
                exponential_backoff=True,
                jitter=True,
                circuit_breaker_threshold=5
            ),
            
            ErrorCategory.TRANSACTION: RecoveryStrategy(
                name="transaction_recovery",
                max_retries=2,
                base_delay=5.0,
                max_delay=60.0,
                exponential_backoff=True,
                jitter=False,
                circuit_breaker_threshold=3
            ),
            
            ErrorCategory.SLIPPAGE: RecoveryStrategy(
                name="slippage_adjustment",
                max_retries=2,
                base_delay=1.0,
                max_delay=10.0,
                exponential_backoff=False,
                jitter=False,
                circuit_breaker_threshold=3
            ),
            
            ErrorCategory.GAS: RecoveryStrategy(
                name="gas_optimization",
                max_retries=3,
                base_delay=10.0,
                max_delay=300.0,
                exponential_backoff=True,
                jitter=True,
                circuit_breaker_threshold=5
            ),
            
            ErrorCategory.INSUFFICIENT_FUNDS: RecoveryStrategy(
                name="balance_recovery",
                max_retries=1,
                base_delay=0.0,
                max_delay=0.0,
                exponential_backoff=False,
                jitter=False,
                circuit_breaker_threshold=2
            ),
            
            ErrorCategory.TIMEOUT: RecoveryStrategy(
                name="timeout_retry",
                max_retries=2,
                base_delay=5.0,
                max_delay=120.0,
                exponential_backoff=True,
                jitter=True,
                circuit_breaker_threshold=4
            ),
            
            ErrorCategory.VALIDATION: RecoveryStrategy(
                name="validation_fix",
                max_retries=1,
                base_delay=0.0,
                max_delay=0.0,
                exponential_backoff=False,
                jitter=False,
                circuit_breaker_threshold=3
            )
        }
        
        return strategies

    async def handle_error(
        self, 
        error: Exception, 
        state: BrainState,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main error handling entry point.
        
        Args:
            error: The exception that occurred
            state: Current brain state
            context: Additional context about the error
            
        Returns:
            Recovery result with actions taken
        """
        
        try:
            # Classify the error
            error_info = self._classify_error(error, state, context)
            
            # Record the error
            self._record_error(error_info)
            
            # Check for emergency conditions
            if self._should_trigger_emergency_stop(error_info):
                return await self._trigger_emergency_stop(state, error_info)
            
            # Attempt recovery
            recovery_result = await self._attempt_recovery(error_info, state)
            
            # Learn from the error
            await self._learn_from_error(error_info, recovery_result)
            
            return recovery_result
            
        except Exception as e:
            # Meta-error: error in error handling
            self.logger.critical(f"Error in error handler: {e}")
            
            return {
                'success': False,
                'action': 'emergency_stop',
                'reason': 'Error handler failure',
                'emergency_triggered': True
            }

    def _classify_error(
        self, 
        error: Exception, 
        state: BrainState,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """Classify and categorize the error."""
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # Determine category
        category = self._determine_error_category(error_type, error_message)
        
        # Determine severity
        severity = self._determine_error_severity(category, error_message, state)
        
        # Build context
        error_context = {
            'node': context.get('node') if context else 'unknown',
            'cycle_id': state.get('cycle_id', ''),
            'action_type': state.get('selected_action', {}).get('action_type', ''),
            'gas_price': state.get('gas_price', 0),
            'network_congestion': state.get('network_congestion', 0),
            **(context or {})
        }
        
        return ErrorInfo(
            timestamp=datetime.now(),
            error_type=error_type,
            severity=severity,
            category=category,
            message=error_message,
            context=error_context
        )

    def _determine_error_category(self, error_type: str, error_message: str) -> ErrorCategory:
        """Determine error category based on type and message."""
        
        error_message_lower = error_message.lower()
        
        # Network-related errors
        network_indicators = ['timeout', 'connection', 'network', 'dns', 'unreachable']
        if any(indicator in error_message_lower for indicator in network_indicators):
            return ErrorCategory.NETWORK
        
        # Transaction-related errors
        transaction_indicators = ['transaction', 'revert', 'failed', 'nonce', 'underpriced']
        if any(indicator in error_message_lower for indicator in transaction_indicators):
            return ErrorCategory.TRANSACTION
        
        # Gas-related errors
        gas_indicators = ['gas', 'out of gas', 'intrinsic gas too low']
        if any(indicator in error_message_lower for indicator in gas_indicators):
            return ErrorCategory.GAS
        
        # Slippage errors
        slippage_indicators = ['slippage', 'price impact', 'too little received']
        if any(indicator in error_message_lower for indicator in slippage_indicators):
            return ErrorCategory.SLIPPAGE
        
        # Insufficient funds
        funds_indicators = ['insufficient', 'balance', 'not enough']
        if any(indicator in error_message_lower for indicator in funds_indicators):
            return ErrorCategory.INSUFFICIENT_FUNDS
        
        # Validation errors
        validation_indicators = ['validation', 'invalid', 'malformed', 'parsing']
        if any(indicator in error_message_lower for indicator in validation_indicators):
            return ErrorCategory.VALIDATION
        
        # Timeout errors
        if 'timeout' in error_type.lower() or 'timeout' in error_message_lower:
            return ErrorCategory.TIMEOUT
        
        # Memory errors
        memory_indicators = ['memory', 'storage', 'database']
        if any(indicator in error_message_lower for indicator in memory_indicators):
            return ErrorCategory.MEMORY
        
        # Calculation errors
        calculation_indicators = ['division by zero', 'overflow', 'underflow', 'nan', 'infinity']
        if any(indicator in error_message_lower for indicator in calculation_indicators):
            return ErrorCategory.CALCULATION
        
        return ErrorCategory.UNKNOWN

    def _determine_error_severity(
        self, 
        category: ErrorCategory, 
        error_message: str, 
        state: BrainState
    ) -> ErrorSeverity:
        """Determine error severity based on category and context."""
        
        # Critical errors that require immediate emergency stop
        critical_indicators = [
            'critical', 'fatal', 'emergency', 'panic', 'security',
            'unauthorized', 'hack', 'exploit'
        ]
        
        if any(indicator in error_message.lower() for indicator in critical_indicators):
            return ErrorSeverity.CRITICAL
        
        # Category-based severity
        if category in [ErrorCategory.INSUFFICIENT_FUNDS, ErrorCategory.TRANSACTION]:
            # Financial errors are generally high severity
            return ErrorSeverity.HIGH
        
        elif category in [ErrorCategory.GAS, ErrorCategory.SLIPPAGE]:
            # Execution errors are medium to high severity
            gas_price = state.get('gas_price', 0)
            if gas_price > self.config.max_gas_price * 2:
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM
        
        elif category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
            # Network issues are usually medium severity
            return ErrorSeverity.MEDIUM
        
        elif category in [ErrorCategory.VALIDATION, ErrorCategory.CALCULATION]:
            # Logic errors can be high severity
            return ErrorSeverity.HIGH
        
        else:
            # Default to medium severity
            return ErrorSeverity.MEDIUM

    def _record_error(self, error_info: ErrorInfo):
        """Record error for tracking and analysis."""
        
        self.error_history.append(error_info)
        self.error_counts[error_info.category] += 1
        
        # Track error patterns
        pattern_key = f"{error_info.category}:{error_info.error_type}"
        self.error_patterns[pattern_key].append(error_info.timestamp)
        
        self.logger.error(
            f"Error recorded: {error_info.category.value} - {error_info.error_type} - {error_info.message}"
        )

    def _should_trigger_emergency_stop(self, error_info: ErrorInfo) -> bool:
        """Determine if emergency stop should be triggered."""
        
        # Always trigger on critical errors
        if error_info.severity == ErrorSeverity.CRITICAL:
            return True
        
        # Check error frequency patterns
        if self._is_error_pattern_dangerous(error_info):
            return True
        
        # Check if too many high-severity errors recently
        recent_high_errors = self._count_recent_errors(
            severity_filter=ErrorSeverity.HIGH,
            time_window_minutes=10
        )
        
        if recent_high_errors >= 3:
            return True
        
        # Check specific error categories
        if error_info.category == ErrorCategory.INSUFFICIENT_FUNDS:
            # Multiple fund errors indicate serious issue
            recent_fund_errors = self._count_recent_errors(
                category_filter=ErrorCategory.INSUFFICIENT_FUNDS,
                time_window_minutes=5
            )
            
            if recent_fund_errors >= 2:
                return True
        
        return False

    def _is_error_pattern_dangerous(self, error_info: ErrorInfo) -> bool:
        """Check if error pattern indicates dangerous situation."""
        
        pattern_key = f"{error_info.category}:{error_info.error_type}"
        pattern_timestamps = self.error_patterns[pattern_key]
        
        # Check for rapid succession of same error
        if len(pattern_timestamps) >= 5:
            recent_occurrences = [
                ts for ts in pattern_timestamps 
                if (datetime.now() - ts).total_seconds() < 300  # 5 minutes
            ]
            
            if len(recent_occurrences) >= 3:
                return True
        
        return False

    def _count_recent_errors(
        self, 
        time_window_minutes: int,
        severity_filter: Optional[ErrorSeverity] = None,
        category_filter: Optional[ErrorCategory] = None
    ) -> int:
        """Count recent errors matching criteria."""
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        count = 0
        for error_info in self.error_history:
            if error_info.timestamp < cutoff_time:
                continue
            
            if severity_filter and error_info.severity != severity_filter:
                continue
                
            if category_filter and error_info.category != category_filter:
                continue
            
            count += 1
        
        return count

    async def _trigger_emergency_stop(
        self, 
        state: BrainState, 
        error_info: ErrorInfo
    ) -> Dict[str, Any]:
        """Trigger emergency stop procedures."""
        
        self.emergency_stop_active = True
        self.emergency_triggers += 1
        self.last_emergency_time = datetime.now()
        
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {error_info.message}")
        
        try:
            # Cancel pending transactions
            await self._cancel_pending_transactions(state)
            
            # Close risky positions if possible
            await self._close_risky_positions(state)
            
            # Notify monitoring systems
            await self._send_emergency_alerts(error_info, state)
            
            # Set recovery parameters
            recovery_time = self._calculate_emergency_recovery_time(error_info)
            
            return {
                'success': False,
                'action': 'emergency_stop',
                'reason': error_info.message,
                'emergency_triggered': True,
                'recovery_time_minutes': recovery_time,
                'error_category': error_info.category.value,
                'error_severity': error_info.severity.value
            }
            
        except Exception as e:
            self.logger.critical(f"Error during emergency stop: {e}")
            
            return {
                'success': False,
                'action': 'emergency_stop',
                'reason': 'Emergency stop failed',
                'emergency_triggered': True,
                'critical_failure': True
            }

    async def _attempt_recovery(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Attempt to recover from the error."""
        
        strategy = self.recovery_strategies.get(error_info.category)
        
        if not strategy:
            return {
                'success': False,
                'action': 'no_recovery',
                'reason': f'No recovery strategy for {error_info.category.value}'
            }
        
        # Check circuit breaker
        service_key = self._get_service_key_for_category(error_info.category)
        circuit_breaker = self.circuit_breakers.get(service_key)
        
        if circuit_breaker and circuit_breaker.state == "open":
            return {
                'success': False,
                'action': 'circuit_breaker_open',
                'reason': f'Circuit breaker open for {service_key}'
            }
        
        # Attempt recovery with retries
        for attempt in range(strategy.max_retries + 1):
            try:
                if attempt > 0:
                    # Wait before retry
                    delay = self._calculate_retry_delay(strategy, attempt)
                    await asyncio.sleep(delay)
                
                # Execute recovery strategy
                recovery_result = await self._execute_recovery_strategy(
                    strategy, error_info, state, attempt
                )
                
                if recovery_result['success']:
                    error_info.recovery_attempted = True
                    error_info.recovery_successful = True
                    error_info.retry_count = attempt
                    
                    return recovery_result
                
            except Exception as e:
                self.logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")
                
                if circuit_breaker:
                    circuit_breaker._on_failure()
        
        # All recovery attempts failed
        error_info.recovery_attempted = True
        error_info.recovery_successful = False
        error_info.retry_count = strategy.max_retries
        
        return {
            'success': False,
            'action': 'recovery_failed',
            'reason': f'All {strategy.max_retries} recovery attempts failed',
            'strategy': strategy.name
        }

    async def _execute_recovery_strategy(
        self, 
        strategy: RecoveryStrategy, 
        error_info: ErrorInfo, 
        state: BrainState, 
        attempt: int
    ) -> Dict[str, Any]:
        """Execute specific recovery strategy."""
        
        if strategy.name == "network_retry":
            return await self._recover_network_error(error_info, state)
        
        elif strategy.name == "transaction_recovery":
            return await self._recover_transaction_error(error_info, state)
        
        elif strategy.name == "slippage_adjustment":
            return await self._recover_slippage_error(error_info, state)
        
        elif strategy.name == "gas_optimization":
            return await self._recover_gas_error(error_info, state)
        
        elif strategy.name == "balance_recovery":
            return await self._recover_balance_error(error_info, state)
        
        elif strategy.name == "timeout_retry":
            return await self._recover_timeout_error(error_info, state)
        
        elif strategy.name == "validation_fix":
            return await self._recover_validation_error(error_info, state)
        
        else:
            return {
                'success': False,
                'action': 'unknown_strategy',
                'reason': f'Unknown recovery strategy: {strategy.name}'
            }

    # Recovery strategy implementations

    async def _recover_network_error(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Recover from network errors."""
        
        # Simple retry - network errors are often transient
        return {
            'success': True,
            'action': 'network_retry',
            'modifications': {},
            'reason': 'Network error - retry recommended'
        }

    async def _recover_transaction_error(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Recover from transaction errors."""
        
        error_message = error_info.message.lower()
        
        if 'nonce' in error_message:
            return {
                'success': True,
                'action': 'nonce_retry',
                'modifications': {'refresh_nonce': True},
                'reason': 'Nonce error - refresh and retry'
            }
        
        elif 'underpriced' in error_message:
            return {
                'success': True,
                'action': 'gas_increase',
                'modifications': {'gas_price_multiplier': 1.2},
                'reason': 'Underpriced transaction - increase gas price'
            }
        
        else:
            # Generic transaction retry
            return {
                'success': True,
                'action': 'transaction_retry',
                'modifications': {'transaction_refresh': True},
                'reason': 'Transaction error - rebuild and retry'
            }

    async def _recover_slippage_error(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Recover from slippage errors."""
        
        current_slippage = state.get('execution_plan', {}).get('slippage_tolerance', 0.01)
        
        if current_slippage < 0.05:  # Less than 5%
            new_slippage = min(current_slippage * 1.5, 0.05)
            
            return {
                'success': True,
                'action': 'increase_slippage',
                'modifications': {'slippage_tolerance': new_slippage},
                'reason': f'Slippage error - increase tolerance to {new_slippage:.3f}'
            }
        else:
            return {
                'success': False,
                'action': 'slippage_too_high',
                'reason': 'Slippage tolerance already at maximum'
            }

    async def _recover_gas_error(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Recover from gas-related errors."""
        
        current_gas = state.get('gas_price', 0.001)
        
        if current_gas < self.config.max_gas_price:
            new_gas = min(current_gas * 1.3, self.config.max_gas_price)
            
            return {
                'success': True,
                'action': 'increase_gas',
                'modifications': {
                    'gas_price': new_gas,
                    'gas_strategy': 'aggressive'
                },
                'reason': f'Gas error - increase price to {new_gas:.6f}'
            }
        else:
            return {
                'success': False,
                'action': 'gas_too_high',
                'reason': 'Gas price already at maximum - wait for better conditions'
            }

    async def _recover_balance_error(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Recover from insufficient balance errors."""
        
        # Balance errors usually can't be recovered automatically
        return {
            'success': False,
            'action': 'insufficient_funds',
            'reason': 'Insufficient balance - manual intervention required',
            'suggestions': [
                'Check wallet balances',
                'Ensure sufficient token allowances',
                'Verify network and token addresses'
            ]
        }

    async def _recover_timeout_error(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Recover from timeout errors."""
        
        return {
            'success': True,
            'action': 'timeout_retry',
            'modifications': {
                'timeout_multiplier': 1.5,
                'retry_with_backoff': True
            },
            'reason': 'Timeout error - retry with longer timeout'
        }

    async def _recover_validation_error(
        self, 
        error_info: ErrorInfo, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Recover from validation errors."""
        
        return {
            'success': False,
            'action': 'validation_failed',
            'reason': 'Validation error - requires code fix',
            'error_details': error_info.message
        }

    async def _learn_from_error(
        self, 
        error_info: ErrorInfo, 
        recovery_result: Dict[str, Any]
    ):
        """Learn from error and recovery for future improvement."""
        
        try:
            # Store error pattern for analysis
            pattern_key = f"{error_info.category}:{error_info.error_type}"
            
            learning_entry = {
                'error_info': error_info,
                'recovery_result': recovery_result,
                'context_factors': self._extract_context_factors(error_info),
                'learning_timestamp': datetime.now()
            }
            
            # This would integrate with the memory system
            # to store error learning data
            
            self.logger.info(f"Learning from error: {pattern_key}")
            
        except Exception as e:
            self.logger.error(f"Error in error learning: {e}")

    def _extract_context_factors(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Extract context factors that might contribute to errors."""
        
        context = error_info.context
        
        factors = {
            'gas_price_high': context.get('gas_price', 0) > 0.02,
            'network_congestion_high': context.get('network_congestion', 0) > 0.7,
            'action_type': context.get('action_type', 'unknown'),
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
        
        return factors

    # Helper methods

    def _calculate_retry_delay(self, strategy: RecoveryStrategy, attempt: int) -> float:
        """Calculate delay before retry attempt."""
        
        if not strategy.exponential_backoff:
            delay = strategy.base_delay
        else:
            delay = strategy.base_delay * (2 ** attempt)
        
        delay = min(delay, strategy.max_delay)
        
        if strategy.jitter:
            # Add random jitter to prevent thundering herd
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay

    def _get_service_key_for_category(self, category: ErrorCategory) -> str:
        """Get service key for circuit breaker based on error category."""
        
        service_mapping = {
            ErrorCategory.NETWORK: 'external_data',
            ErrorCategory.TRANSACTION: 'cdp_api',
            ErrorCategory.EXTERNAL_API: 'external_data',
            ErrorCategory.MEMORY: 'memory_system'
        }
        
        return service_mapping.get(category, 'general')

    def _calculate_emergency_recovery_time(self, error_info: ErrorInfo) -> int:
        """Calculate recovery time for emergency stop."""
        
        base_time = 30  # 30 minutes base
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            return base_time * 4  # 2 hours for critical errors
        elif error_info.severity == ErrorSeverity.HIGH:
            return base_time * 2  # 1 hour for high severity
        else:
            return base_time  # 30 minutes for others

    async def _cancel_pending_transactions(self, state: BrainState):
        """Cancel pending transactions during emergency stop."""
        try:
            pending_txs = state.get('pending_transactions', [])
            for tx_hash in pending_txs:
                # Would implement actual transaction cancellation
                self.logger.info(f"Cancelling transaction: {tx_hash}")
        except Exception as e:
            self.logger.error(f"Error cancelling transactions: {e}")

    async def _close_risky_positions(self, state: BrainState):
        """Close risky positions during emergency stop."""
        try:
            positions = state.get('active_positions', [])
            risky_positions = [
                pos for pos in positions 
                if pos.get('risk_score', 0) > 0.7
            ]
            
            for position in risky_positions:
                # Would implement actual position closing
                self.logger.info(f"Closing risky position: {position.get('pool_address')}")
                
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")

    async def _send_emergency_alerts(self, error_info: ErrorInfo, state: BrainState):
        """Send emergency alerts to monitoring systems."""
        try:
            alert_data = {
                'timestamp': error_info.timestamp.isoformat(),
                'error_category': error_info.category.value,
                'error_severity': error_info.severity.value,
                'error_message': error_info.message,
                'cycle_id': state.get('cycle_id', ''),
                'portfolio_value': state.get('portfolio_performance', {}).get('total_value', 0)
            }
            
            # Would implement actual alerting (email, Discord, etc.)
            self.logger.critical(f"EMERGENCY ALERT: {alert_data}")
            
        except Exception as e:
            self.logger.error(f"Error sending alerts: {e}")

    def is_emergency_active(self) -> bool:
        """Check if emergency stop is currently active."""
        return self.emergency_stop_active

    def clear_emergency_stop(self):
        """Clear emergency stop (manual intervention)."""
        self.emergency_stop_active = False
        self.logger.info("Emergency stop cleared manually")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        
        recent_errors = [
            err for err in self.error_history 
            if (datetime.now() - err.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors_1h': len(recent_errors),
            'error_counts_by_category': dict(self.error_counts),
            'emergency_triggers': self.emergency_triggers,
            'last_emergency': self.last_emergency_time.isoformat() if self.last_emergency_time else None,
            'emergency_active': self.emergency_stop_active,
            'circuit_breaker_states': {
                name: breaker.state 
                for name, breaker in self.circuit_breakers.items()
            }
        }