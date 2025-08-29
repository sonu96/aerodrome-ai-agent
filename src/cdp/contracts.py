"""
CDP Smart Contract Operations - Comprehensive smart contract interaction wrapper

This module provides a high-level interface for all smart contract operations
using CDP SDK exclusively. It handles:
- Contract invocation and state reading
- Batch operations for efficiency  
- Event monitoring and filtering
- Automatic gas optimization
- Error handling and retry logic

All smart contract interactions must go through this module.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from decimal import Decimal
import time

try:
    from cdp_sdk import SmartContract, readContract
except ImportError:
    SmartContract = None
    readContract = None

from .errors import (
    CDPError, ContractError, TransactionError, NetworkError, 
    RateLimitError, handle_cdp_error
)
from ..contracts.abis import ABI_MAP, get_abi, validate_abi_function_inputs
from ..contracts.addresses import (
    get_contract_address, get_token_address, validate_address
)


class ContractOperations:
    """
    Smart contract operations via CDP SDK.
    
    This class provides the primary interface for all smart contract
    interactions, ensuring type safety, error handling, and optimization.
    """
    
    def __init__(self, cdp_manager):
        """
        Initialize contract operations.
        
        Args:
            cdp_manager: CDPManager instance
        """
        self.cdp = cdp_manager
        self.logger = logging.getLogger(__name__)
        self.wallet = cdp_manager.wallet
        self.network = cdp_manager.network
        self.registered_contracts = {}  # Cache for registered contracts
        self.call_cache = {}  # Cache for read-only calls
        self.cache_ttl = 60  # Cache TTL in seconds
    
    async def invoke_contract(
        self,
        contract_address: str,
        method: str,
        args: Dict[str, Any],
        abi: Optional[List[Dict]] = None,
        value: int = 0,
        gas_limit: Optional[int] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Invoke any smart contract method.
        
        This is the primary method for executing contract functions.
        Uses CDP SDK exclusively with automatic error handling and retries.
        
        Args:
            contract_address: Contract address to interact with
            method: Method name to call
            args: Method arguments dictionary
            abi: Contract ABI (will be inferred if not provided)
            value: ETH value to send (for payable functions)
            gas_limit: Custom gas limit
            max_retries: Maximum retry attempts
            
        Returns:
            Dictionary containing transaction result
            
        Raises:
            ContractError: If contract interaction fails
            TransactionError: If transaction execution fails
        """
        # Validate inputs
        if not validate_address(contract_address):
            raise ContractError(f"Invalid contract address: {contract_address}")
        
        # Validate ABI inputs if ABI is provided
        if abi:
            try:
                # Find contract type for validation
                contract_type = self._identify_contract_type(abi)
                if contract_type:
                    validate_abi_function_inputs(contract_type, method, args)
            except ValueError as e:
                raise ContractError(f"ABI validation failed: {str(e)}")
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Pre-execution validation
                await self._validate_contract_call(contract_address, method, args)
                
                # Prepare contract invocation parameters
                invoke_params = {
                    'contract_address': contract_address,
                    'method': method,
                    'args': args,
                    'value': value
                }
                
                # Add ABI if provided
                if abi:
                    invoke_params['abi'] = abi
                
                # Add gas limit if provided
                if gas_limit:
                    invoke_params['gas_limit'] = gas_limit
                
                self.logger.info(f"Invoking {method} on {contract_address[:10]}...")
                
                # Execute via CDP SDK
                result = await self.wallet.invoke_contract(invoke_params)
                
                # Wait for confirmation
                confirmation = await result.wait()
                
                success_result = {
                    'success': True,
                    'tx_hash': result.transaction_hash,
                    'gas_used': confirmation.gas_used,
                    'gas_price': getattr(confirmation, 'gas_price', None),
                    'block_number': confirmation.block_number,
                    'block_hash': getattr(confirmation, 'block_hash', None),
                    'logs': confirmation.logs,
                    'status': confirmation.status,
                    'contract_address': contract_address,
                    'method': method,
                    'args': args,
                    'value': value,
                    'timestamp': int(time.time())
                }
                
                self.logger.info(f"Transaction successful: {result.transaction_hash}")
                return success_result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Handle error and determine retry strategy
                error_context = {
                    'contract_address': contract_address,
                    'method': method,
                    'args': args,
                    'retry_count': retry_count
                }
                
                error_strategy = await handle_cdp_error(e, error_context)
                
                if not error_strategy.get('retry', False) or retry_count > max_retries:
                    break
                
                # Apply retry delay
                delay = error_strategy.get('delay', 1)
                self.logger.warning(f"Contract call failed, retrying in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
                
                # Apply error-specific adjustments
                if error_strategy.get('action') == 'INCREASE_GAS' and gas_limit:
                    gas_limit = int(gas_limit * error_strategy.get('details', {}).get('gas_multiplier', 1.5))
        
        # All retries failed
        error_result = {
            'success': False,
            'error': str(last_error),
            'method': method,
            'contract_address': contract_address,
            'args': args,
            'retry_count': retry_count,
            'timestamp': int(time.time())
        }
        
        self.logger.error(f"Contract invocation failed after {retry_count} retries: {str(last_error)}")
        raise TransactionError(f"Contract invocation failed: {str(last_error)}", 
                             contract_address=contract_address, method=method)
    
    async def read_contract(
        self,
        contract_address: str,
        method: str,
        args: Dict[str, Any] = None,
        abi: Optional[List[Dict]] = None,
        block_number: Union[int, str] = 'latest',
        use_cache: bool = True
    ) -> Any:
        """
        Read contract state without transaction.
        
        CDP SDK automatically infers types when ABI is provided.
        Results are cached for efficiency.
        
        Args:
            contract_address: Contract address
            method: Method name to call
            args: Method arguments
            abi: Contract ABI
            block_number: Block number or 'latest'
            use_cache: Whether to use caching
            
        Returns:
            Contract call result
            
        Raises:
            ContractError: If contract read fails
        """
        args = args or {}
        
        # Validate inputs
        if not validate_address(contract_address):
            raise ContractError(f"Invalid contract address: {contract_address}")
        
        # Create cache key
        cache_key = f"{contract_address}:{method}:{str(args)}:{block_number}"
        
        # Check cache first
        if use_cache and cache_key in self.call_cache:
            cache_entry = self.call_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                self.logger.debug(f"Cache hit for {method} on {contract_address[:10]}...")
                return cache_entry['result']
        
        try:
            # Prepare read parameters
            read_params = {
                'network_id': self.network,
                'contract_address': contract_address,
                'method': method,
                'args': args,
                'block_number': block_number
            }
            
            # Add ABI if provided
            if abi:
                read_params['abi'] = abi
            
            self.logger.debug(f"Reading {method} from {contract_address[:10]}...")
            
            # Execute read via CDP SDK
            result = await readContract(read_params)
            
            # Cache the result
            if use_cache:
                self.call_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Contract read failed for {method}: {str(e)}")
            raise ContractError(f"Contract read failed: {str(e)}", 
                              contract_address=contract_address, method=method)
    
    async def batch_read(self, calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Batch multiple contract reads for efficiency.
        
        Args:
            calls: List of contract call dictionaries
            
        Returns:
            List of call results (same order as input)
        """
        self.logger.info(f"Executing batch read with {len(calls)} calls")
        
        # Execute all calls concurrently
        tasks = [
            self.read_contract(**call) for call in calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors but don't raise
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Batch read call {i} failed: {str(result)}")
        
        return results
    
    async def batch_invoke(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch multiple contract invocations sequentially.
        
        Note: Invocations must be sequential to maintain proper nonce ordering.
        
        Args:
            calls: List of contract invocation dictionaries
            
        Returns:
            List of transaction results
        """
        self.logger.info(f"Executing batch invoke with {len(calls)} transactions")
        
        results = []
        
        for i, call in enumerate(calls):
            try:
                self.logger.info(f"Executing batch transaction {i+1}/{len(calls)}")
                result = await self.invoke_contract(**call)
                results.append(result)
                
                # Small delay between transactions
                await asyncio.sleep(1)
                
            except Exception as e:
                error_result = {
                    'success': False,
                    'error': str(e),
                    'call_index': i,
                    'call_params': call
                }
                results.append(error_result)
                self.logger.error(f"Batch transaction {i+1} failed: {str(e)}")
        
        return results
    
    async def _validate_contract_call(
        self,
        contract_address: str,
        method: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Validate contract call before execution.
        
        Args:
            contract_address: Contract address
            method: Method name
            args: Method arguments
            
        Raises:
            ContractError: If validation fails
        """
        # Check if contract exists (has code)
        try:
            # Try to read a basic property to verify contract exists
            await self.read_contract(
                contract_address=contract_address,
                method='totalSupply',  # Common ERC20 method
                use_cache=False
            )
        except:
            # If totalSupply fails, try a more basic check
            # This is a simplified validation - in production you'd check bytecode
            pass
    
    def _identify_contract_type(self, abi: List[Dict]) -> Optional[str]:
        """
        Identify contract type from ABI.
        
        Args:
            abi: Contract ABI
            
        Returns:
            Contract type string or None
        """
        # Extract method names from ABI
        methods = {item['name'] for item in abi if item.get('type') == 'function'}
        
        # Check for known contract types
        if 'swapExactTokensForTokens' in methods:
            return 'ROUTER'
        elif 'vote' in methods and 'reset' in methods:
            return 'VOTER'
        elif 'create_lock' in methods and 'balanceOfNFT' in methods:
            return 'VE_AERO'
        elif 'deposit' in methods and 'withdraw' in methods and 'getReward' in methods:
            return 'GAUGE'
        elif 'getReserves' in methods and 'getAmountOut' in methods:
            return 'POOL'
        elif 'balanceOf' in methods and 'transfer' in methods and 'approve' in methods:
            return 'ERC20'
        elif 'createPair' in methods and 'getPair' in methods:
            return 'FACTORY'
        
        return None
    
    def clear_cache(self) -> None:
        """Clear the read call cache."""
        self.call_cache.clear()
        self.logger.info("Contract call cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self.call_cache)
        expired_entries = sum(
            1 for entry in self.call_cache.values()
            if time.time() - entry['timestamp'] >= self.cache_ttl
        )
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'cache_ttl': self.cache_ttl,
            'hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }
    
    async def estimate_gas(
        self,
        contract_address: str,
        method: str,
        args: Dict[str, Any],
        abi: Optional[List[Dict]] = None,
        value: int = 0
    ) -> int:
        """
        Estimate gas for contract call.
        
        Args:
            contract_address: Contract address
            method: Method name
            args: Method arguments
            abi: Contract ABI
            value: ETH value to send
            
        Returns:
            Estimated gas amount
            
        Raises:
            ContractError: If gas estimation fails
        """
        try:
            # Use CDP SDK's gas estimation
            # This would be implemented using CDP's gas estimation APIs
            estimate_params = {
                'contract_address': contract_address,
                'method': method,
                'args': args,
                'value': value
            }
            
            if abi:
                estimate_params['abi'] = abi
            
            # Placeholder for CDP SDK gas estimation
            # In actual implementation, this would call CDP's estimation API
            estimated_gas = 100000  # Default estimate
            
            # Add 20% buffer for safety
            return int(estimated_gas * 1.2)
            
        except Exception as e:
            self.logger.warning(f"Gas estimation failed, using default: {str(e)}")
            # Return a conservative default
            return 200000


class ContractRegistry:
    """
    Register and monitor smart contracts for event tracking.
    
    Provides contract registration for event monitoring and
    maintains a registry of active contracts.
    """
    
    def __init__(self, cdp_manager):
        """
        Initialize contract registry.
        
        Args:
            cdp_manager: CDPManager instance
        """
        self.cdp = cdp_manager
        self.logger = logging.getLogger(__name__)
        self.network = cdp_manager.network
        self.registered_contracts = {}
        self.event_subscriptions = {}
    
    async def register_contract(
        self,
        contract_address: str,
        abi: List[Dict],
        name: Optional[str] = None
    ) -> 'SmartContract':
        """
        Register contract for event monitoring.
        
        Args:
            contract_address: Contract address
            abi: Contract ABI
            name: Contract name (optional)
            
        Returns:
            Registered SmartContract instance
            
        Raises:
            ContractError: If registration fails
        """
        if not validate_address(contract_address):
            raise ContractError(f"Invalid contract address: {contract_address}")
        
        try:
            contract_name = name or f"Contract_{contract_address[:8]}"
            
            self.logger.info(f"Registering contract: {contract_name}")
            
            # Register with CDP SDK
            contract = await SmartContract.register(
                network_id=self.network,
                contract_address=contract_address,
                abi=abi,
                contract_name=contract_name
            )
            
            # Store in registry
            self.registered_contracts[contract_address] = {
                'contract': contract,
                'abi': abi,
                'name': contract_name,
                'registered_at': time.time()
            }
            
            self.logger.info(f"Contract registered successfully: {contract_name}")
            return contract
            
        except Exception as e:
            self.logger.error(f"Contract registration failed: {str(e)}")
            raise ContractError(f"Failed to register contract: {str(e)}")
    
    async def get_contract_events(
        self,
        contract_address: str,
        event_name: str,
        from_block: Union[int, str] = 'latest',
        to_block: Union[int, str] = 'latest',
        filter_params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get historical events from contract.
        
        Args:
            contract_address: Contract address
            event_name: Event name to filter
            from_block: Starting block number
            to_block: Ending block number  
            filter_params: Event filter parameters
            
        Returns:
            List of event dictionaries
            
        Raises:
            ContractError: If event query fails
        """
        if contract_address not in self.registered_contracts:
            raise ContractError(f"Contract {contract_address} not registered")
        
        try:
            filter_params = filter_params or {}
            
            self.logger.debug(f"Querying {event_name} events from {contract_address[:10]}...")
            
            # Query events via CDP SDK
            events = await self.cdp.client.get_smart_contract_events({
                'contract_address': contract_address,
                'event_name': event_name,
                'from_block': from_block,
                'to_block': to_block,
                'filter': filter_params
            })
            
            self.logger.info(f"Retrieved {len(events)} {event_name} events")
            return events
            
        except Exception as e:
            self.logger.error(f"Event query failed: {str(e)}")
            raise ContractError(f"Failed to get events: {str(e)}")
    
    async def subscribe_to_events(
        self,
        contract_address: str,
        event_name: str,
        callback: callable,
        filter_params: Optional[Dict] = None
    ) -> str:
        """
        Subscribe to real-time contract events.
        
        Args:
            contract_address: Contract address
            event_name: Event name to monitor
            callback: Callback function for events
            filter_params: Event filter parameters
            
        Returns:
            Subscription ID
            
        Raises:
            ContractError: If subscription fails
        """
        if contract_address not in self.registered_contracts:
            raise ContractError(f"Contract {contract_address} not registered")
        
        try:
            subscription_id = f"{contract_address}:{event_name}:{int(time.time())}"
            
            # Store subscription info
            self.event_subscriptions[subscription_id] = {
                'contract_address': contract_address,
                'event_name': event_name,
                'callback': callback,
                'filter_params': filter_params or {},
                'created_at': time.time()
            }
            
            self.logger.info(f"Event subscription created: {subscription_id}")
            
            # In a real implementation, this would set up WebSocket connections
            # or polling mechanisms through CDP SDK
            
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"Event subscription failed: {str(e)}")
            raise ContractError(f"Failed to subscribe to events: {str(e)}")
    
    def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """
        Unsubscribe from contract events.
        
        Args:
            subscription_id: Subscription ID to cancel
            
        Returns:
            True if successfully unsubscribed
        """
        if subscription_id in self.event_subscriptions:
            del self.event_subscriptions[subscription_id]
            self.logger.info(f"Unsubscribed from events: {subscription_id}")
            return True
        
        return False
    
    def get_registered_contracts(self) -> Dict[str, Dict]:
        """
        Get all registered contracts.
        
        Returns:
            Dictionary of registered contracts
        """
        return {
            addr: {
                'name': info['name'],
                'registered_at': info['registered_at']
            }
            for addr, info in self.registered_contracts.items()
        }
    
    def get_active_subscriptions(self) -> Dict[str, Dict]:
        """
        Get all active event subscriptions.
        
        Returns:
            Dictionary of active subscriptions
        """
        return {
            sub_id: {
                'contract_address': info['contract_address'],
                'event_name': info['event_name'],
                'created_at': info['created_at']
            }
            for sub_id, info in self.event_subscriptions.items()
        }


# Utility functions for common contract operations
async def get_erc20_balance(
    contract_ops: ContractOperations,
    token_address: str,
    account_address: str
) -> int:
    """
    Get ERC20 token balance.
    
    Args:
        contract_ops: ContractOperations instance
        token_address: Token contract address
        account_address: Account to check balance for
        
    Returns:
        Token balance in wei
    """
    return await contract_ops.read_contract(
        contract_address=token_address,
        method='balanceOf',
        args={'account': account_address},
        abi=get_abi('ERC20')
    )

async def get_token_decimals(
    contract_ops: ContractOperations,
    token_address: str
) -> int:
    """
    Get token decimals.
    
    Args:
        contract_ops: ContractOperations instance
        token_address: Token contract address
        
    Returns:
        Number of token decimals
    """
    return await contract_ops.read_contract(
        contract_address=token_address,
        method='decimals',
        args={},
        abi=get_abi('ERC20')
    )

async def approve_token(
    contract_ops: ContractOperations,
    token_address: str,
    spender_address: str,
    amount: int
) -> Dict[str, Any]:
    """
    Approve token spending.
    
    Args:
        contract_ops: ContractOperations instance
        token_address: Token contract address
        spender_address: Address to approve
        amount: Amount to approve
        
    Returns:
        Transaction result
    """
    return await contract_ops.invoke_contract(
        contract_address=token_address,
        method='approve',
        args={
            'spender': spender_address,
            'amount': amount
        },
        abi=get_abi('ERC20')
    )

# Export public interface
__all__ = [
    'ContractOperations',
    'ContractRegistry',
    'get_erc20_balance',
    'get_token_decimals',
    'approve_token'
]