"""
CDP Error Handling - Custom error classes for CDP SDK operations

This module provides comprehensive error handling for CDP SDK operations including:
- Base CDP errors and specific error types
- Error classification and mapping
- Error recovery strategies
- Retry logic and backoff mechanisms

All CDP SDK operations should use these error classes for consistent error handling.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum


class CDPErrorType(Enum):
    """Error type classification for CDP operations."""
    INSUFFICIENT_BALANCE = "insufficient_balance"
    NONCE_ERROR = "nonce_error"
    GAS_LIMIT_ERROR = "gas_limit_error"
    CONTRACT_REVERT = "contract_revert"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_ERROR = "authentication_error"
    WALLET_ERROR = "wallet_error"
    TRANSACTION_ERROR = "transaction_error"
    UNKNOWN = "unknown"


class CDPError(Exception):
    """
    Base CDP SDK error class.
    
    All CDP SDK related errors should inherit from this base class
    to provide consistent error handling and recovery mechanisms.
    """
    
    def __init__(
        self, 
        message: str, 
        error_type: CDPErrorType = CDPErrorType.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False
    ):
        """
        Initialize CDP error.
        
        Args:
            message: Human-readable error message
            error_type: Classification of error type
            details: Additional error details and context
            retryable: Whether this error can be retried
        """
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.retryable = retryable
        self.timestamp = None  # Set by error handler
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'type': self.error_type.value,
            'details': self.details,
            'retryable': self.retryable,
            'timestamp': self.timestamp
        }


class WalletInitializationError(CDPError):
    """Error during wallet initialization or configuration."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type=CDPErrorType.WALLET_ERROR,
            details=details,
            retryable=False
        )


class InsufficientBalanceError(CDPError):
    """Error when wallet has insufficient balance for operation."""
    
    def __init__(
        self, 
        message: str, 
        required: Optional[int] = None,
        available: Optional[int] = None,
        token: Optional[str] = None
    ):
        details = {
            'required_amount': required,
            'available_amount': available,
            'token_address': token
        }
        super().__init__(
            message=message,
            error_type=CDPErrorType.INSUFFICIENT_BALANCE,
            details=details,
            retryable=False
        )


class TransactionError(CDPError):
    """Error during transaction execution."""
    
    def __init__(
        self, 
        message: str, 
        tx_hash: Optional[str] = None,
        gas_used: Optional[int] = None,
        revert_reason: Optional[str] = None
    ):
        details = {
            'transaction_hash': tx_hash,
            'gas_used': gas_used,
            'revert_reason': revert_reason
        }
        super().__init__(
            message=message,
            error_type=CDPErrorType.TRANSACTION_ERROR,
            details=details,
            retryable=True  # Some transaction errors can be retried
        )


class ContractError(CDPError):
    """Error during smart contract interaction."""
    
    def __init__(
        self, 
        message: str, 
        contract_address: Optional[str] = None,
        method: Optional[str] = None,
        revert_reason: Optional[str] = None
    ):
        details = {
            'contract_address': contract_address,
            'method': method,
            'revert_reason': revert_reason
        }
        super().__init__(
            message=message,
            error_type=CDPErrorType.CONTRACT_REVERT,
            details=details,
            retryable=False
        )


class NetworkError(CDPError):
    """Error related to network connectivity or RPC issues."""
    
    def __init__(self, message: str, network: Optional[str] = None):
        details = {'network': network}
        super().__init__(
            message=message,
            error_type=CDPErrorType.NETWORK_ERROR,
            details=details,
            retryable=True
        )


class RateLimitError(CDPError):
    """Error when rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        retry_after: Optional[int] = None,
        endpoint: Optional[str] = None
    ):
        details = {
            'retry_after': retry_after,
            'endpoint': endpoint
        }
        super().__init__(
            message=message,
            error_type=CDPErrorType.RATE_LIMIT,
            details=details,
            retryable=True
        )


class GasError(CDPError):
    """Error related to gas estimation or limits."""
    
    def __init__(
        self, 
        message: str, 
        estimated_gas: Optional[int] = None,
        gas_limit: Optional[int] = None
    ):
        details = {
            'estimated_gas': estimated_gas,
            'gas_limit': gas_limit
        }
        super().__init__(
            message=message,
            error_type=CDPErrorType.GAS_LIMIT_ERROR,
            details=details,
            retryable=True
        )


class NonceError(CDPError):
    """Error related to transaction nonce management."""
    
    def __init__(
        self, 
        message: str, 
        expected_nonce: Optional[int] = None,
        actual_nonce: Optional[int] = None
    ):
        details = {
            'expected_nonce': expected_nonce,
            'actual_nonce': actual_nonce
        }
        super().__init__(
            message=message,
            error_type=CDPErrorType.NONCE_ERROR,
            details=details,
            retryable=True
        )


class CDPErrorHandler:
    """
    Handle CDP SDK specific errors with automatic classification and recovery.
    
    Provides error analysis, classification, retry logic, and recovery strategies
    for all CDP SDK operations.
    """
    
    ERROR_MAPPINGS = {
        'insufficient funds': CDPErrorType.INSUFFICIENT_BALANCE,
        'insufficient balance': CDPErrorType.INSUFFICIENT_BALANCE,
        'nonce too low': CDPErrorType.NONCE_ERROR,
        'nonce too high': CDPErrorType.NONCE_ERROR,
        'replacement transaction underpriced': CDPErrorType.NONCE_ERROR,
        'gas required exceeds': CDPErrorType.GAS_LIMIT_ERROR,
        'out of gas': CDPErrorType.GAS_LIMIT_ERROR,
        'gas limit reached': CDPErrorType.GAS_LIMIT_ERROR,
        'execution reverted': CDPErrorType.CONTRACT_REVERT,
        'transaction reverted': CDPErrorType.CONTRACT_REVERT,
        'network error': CDPErrorType.NETWORK_ERROR,
        'connection error': CDPErrorType.NETWORK_ERROR,
        'timeout': CDPErrorType.NETWORK_ERROR,
        'rate limit': CDPErrorType.RATE_LIMIT,
        'too many requests': CDPErrorType.RATE_LIMIT,
        'unauthorized': CDPErrorType.AUTHENTICATION_ERROR,
        'invalid api key': CDPErrorType.AUTHENTICATION_ERROR,
        'forbidden': CDPErrorType.AUTHENTICATION_ERROR
    }
    
    def __init__(self):
        """Initialize error handler."""
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}  # Track error frequencies
    
    def classify_error(self, error_msg: str) -> CDPErrorType:
        """
        Classify error type from error message.
        
        Args:
            error_msg: Error message to classify
            
        Returns:
            CDPErrorType: Classified error type
        """
        error_msg_lower = error_msg.lower()
        
        for pattern, error_type in self.ERROR_MAPPINGS.items():
            if pattern in error_msg_lower:
                return error_type
        
        return CDPErrorType.UNKNOWN
    
    async def handle_cdp_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle errors from CDP SDK operations.
        
        Args:
            error: Exception that occurred
            context: Additional context about the operation
            
        Returns:
            Dictionary containing error handling strategy
        """
        context = context or {}
        error_msg = str(error).lower()
        error_type = self.classify_error(error_msg)
        
        # Track error frequency
        self._track_error(error_type)
        
        # Get appropriate handler
        handlers = {
            CDPErrorType.INSUFFICIENT_BALANCE: self._handle_insufficient_balance,
            CDPErrorType.NONCE_ERROR: self._handle_nonce_error,
            CDPErrorType.GAS_LIMIT_ERROR: self._handle_gas_limit,
            CDPErrorType.CONTRACT_REVERT: self._handle_contract_revert,
            CDPErrorType.NETWORK_ERROR: self._handle_network_error,
            CDPErrorType.RATE_LIMIT: self._handle_rate_limit,
            CDPErrorType.AUTHENTICATION_ERROR: self._handle_auth_error,
            CDPErrorType.WALLET_ERROR: self._handle_wallet_error,
            CDPErrorType.TRANSACTION_ERROR: self._handle_transaction_error
        }
        
        handler = handlers.get(error_type, self._handle_unknown_error)
        return await handler(error, context)
    
    def _track_error(self, error_type: CDPErrorType) -> None:
        """Track error frequency for analysis."""
        key = error_type.value
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    async def _handle_insufficient_balance(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle insufficient balance errors."""
        # Extract token information if available
        token = context.get('token_address', 'native token')
        
        return {
            'retry': False,
            'action': 'ALERT_USER',
            'message': f'Insufficient {token} balance',
            'suggestion': 'Fund wallet or reduce position size',
            'details': {
                'error_type': 'insufficient_balance',
                'token': token,
                'operation': context.get('operation', 'unknown')
            }
        }
    
    async def _handle_nonce_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle nonce errors with automatic retry."""
        return {
            'retry': True,
            'delay': 5,  # Wait 5 seconds
            'max_retries': 3,
            'action': 'RESET_NONCE',
            'message': 'Nonce error detected, retrying with corrected nonce',
            'details': {
                'error_type': 'nonce_error',
                'retry_strategy': 'exponential_backoff'
            }
        }
    
    async def _handle_gas_limit(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gas limit errors."""
        return {
            'retry': True,
            'delay': 2,
            'max_retries': 2,
            'action': 'INCREASE_GAS',
            'message': 'Gas limit exceeded, retrying with higher gas limit',
            'details': {
                'error_type': 'gas_limit_error',
                'gas_multiplier': 1.5  # Increase gas by 50%
            }
        }
    
    async def _handle_contract_revert(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle smart contract revert errors."""
        # Extract revert reason if available
        revert_reason = self._extract_revert_reason(str(error))
        
        return {
            'retry': False,
            'action': 'ANALYZE_REVERT',
            'message': f'Contract execution reverted: {revert_reason}',
            'suggestion': 'Check contract parameters and conditions',
            'details': {
                'error_type': 'contract_revert',
                'revert_reason': revert_reason,
                'contract': context.get('contract_address'),
                'method': context.get('method')
            }
        }
    
    async def _handle_network_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network connectivity errors."""
        return {
            'retry': True,
            'delay': 10,  # Wait longer for network issues
            'max_retries': 5,
            'action': 'RETRY_WITH_BACKOFF',
            'message': 'Network error detected, retrying with backoff',
            'details': {
                'error_type': 'network_error',
                'backoff_strategy': 'exponential',
                'network': context.get('network')
            }
        }
    
    async def _handle_rate_limit(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rate limiting errors."""
        # Extract retry-after header if available
        retry_after = self._extract_retry_after(str(error))
        
        return {
            'retry': True,
            'delay': retry_after or 60,  # Default to 60 seconds
            'max_retries': 3,
            'action': 'WAIT_AND_RETRY',
            'message': f'Rate limit exceeded, waiting {retry_after or 60} seconds',
            'details': {
                'error_type': 'rate_limit',
                'retry_after': retry_after,
                'endpoint': context.get('endpoint')
            }
        }
    
    async def _handle_auth_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle authentication errors."""
        return {
            'retry': False,
            'action': 'CHECK_CREDENTIALS',
            'message': 'Authentication failed - check API credentials',
            'suggestion': 'Verify CDP_API_KEY_ID and CDP_API_KEY_SECRET',
            'details': {
                'error_type': 'authentication_error',
                'requires_manual_intervention': True
            }
        }
    
    async def _handle_wallet_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle wallet-related errors."""
        return {
            'retry': False,
            'action': 'CHECK_WALLET',
            'message': 'Wallet operation failed',
            'suggestion': 'Check wallet configuration and permissions',
            'details': {
                'error_type': 'wallet_error',
                'wallet_address': context.get('wallet_address')
            }
        }
    
    async def _handle_transaction_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transaction-related errors."""
        return {
            'retry': True,
            'delay': 5,
            'max_retries': 2,
            'action': 'RETRY_TRANSACTION',
            'message': 'Transaction failed, retrying',
            'details': {
                'error_type': 'transaction_error',
                'tx_hash': context.get('tx_hash')
            }
        }
    
    async def _handle_unknown_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown/unclassified errors."""
        self.logger.warning(f"Unknown error type: {str(error)}")
        
        return {
            'retry': False,
            'action': 'LOG_AND_ALERT',
            'message': f'Unknown error occurred: {str(error)}',
            'suggestion': 'Check logs and contact support if error persists',
            'details': {
                'error_type': 'unknown',
                'raw_error': str(error),
                'context': context
            }
        }
    
    def _extract_revert_reason(self, error_msg: str) -> str:
        """Extract revert reason from error message."""
        # Try to extract revert reason from common patterns
        if 'revert' in error_msg.lower():
            # Look for reason in parentheses or quotes
            import re
            patterns = [
                r'revert (.+)',
                r'"([^"]*)"',
                r"'([^']*)'"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, error_msg, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return "Unknown revert reason"
    
    def _extract_retry_after(self, error_msg: str) -> Optional[int]:
        """Extract retry-after delay from error message."""
        import re
        
        # Look for retry-after values in common formats
        patterns = [
            r'retry.*after.*(\d+)',
            r'wait.*(\d+).*seconds',
            r'try.*again.*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_msg.lower())
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring and analysis.
        
        Returns:
            Dictionary containing error frequency and patterns
        """
        total_errors = sum(self.error_counts.values())
        
        if total_errors == 0:
            return {'total_errors': 0, 'error_distribution': {}}
        
        error_distribution = {
            error_type: {
                'count': count,
                'percentage': (count / total_errors) * 100
            }
            for error_type, count in self.error_counts.items()
        }
        
        return {
            'total_errors': total_errors,
            'error_distribution': error_distribution,
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0]
        }
    
    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.error_counts.clear()


# Convenience function for error handling
async def handle_cdp_error(
    error: Exception, 
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for handling CDP errors.
    
    Args:
        error: Exception to handle
        context: Additional context
        
    Returns:
        Error handling strategy
    """
    handler = CDPErrorHandler()
    return await handler.handle_cdp_error(error, context)


# Error mapping for quick access
ERROR_CLASSES = {
    CDPErrorType.INSUFFICIENT_BALANCE: InsufficientBalanceError,
    CDPErrorType.TRANSACTION_ERROR: TransactionError,
    CDPErrorType.CONTRACT_REVERT: ContractError,
    CDPErrorType.NETWORK_ERROR: NetworkError,
    CDPErrorType.RATE_LIMIT: RateLimitError,
    CDPErrorType.GAS_LIMIT_ERROR: GasError,
    CDPErrorType.NONCE_ERROR: NonceError,
    CDPErrorType.WALLET_ERROR: WalletInitializationError
}


def create_cdp_error(
    error_type: CDPErrorType, 
    message: str, 
    **kwargs
) -> CDPError:
    """
    Create appropriate CDP error instance based on error type.
    
    Args:
        error_type: Type of error to create
        message: Error message
        **kwargs: Additional error-specific arguments
        
    Returns:
        Appropriate CDPError subclass instance
    """
    error_class = ERROR_CLASSES.get(error_type, CDPError)
    return error_class(message, **kwargs)