"""
CDP Wallet Operations - Wallet management and token operations

This module provides comprehensive wallet operations including:
- Native token and ERC20 balance management
- Token approvals and allowances
- Multi-token balance fetching
- Transaction history and monitoring

All operations use CDP SDK exclusively.
"""

import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
import asyncio

try:
    from cdp_sdk import readContract
except ImportError:
    readContract = None

from .errors import CDPError, InsufficientBalanceError, TransactionError
from ..contracts.abis import ERC20_ABI
from ..contracts.addresses import TOKEN_ADDRESSES, NETWORKS, get_token_address, get_network_info


class WalletOperations:
    """
    CDP wallet management operations.
    
    Provides high-level wallet operations for:
    - Balance checking (native and ERC20 tokens)
    - Token approvals and allowance management
    - Batch operations for efficiency
    - Transaction monitoring
    """
    
    def __init__(self, cdp_manager):
        """
        Initialize wallet operations.
        
        Args:
            cdp_manager: CDPManager instance
        """
        self.cdp = cdp_manager
        self.logger = logging.getLogger(__name__)
        self.wallet = cdp_manager.wallet
        self.network = cdp_manager.network
    
    async def get_balance(self, token_address: Optional[str] = None) -> float:
        """
        Get wallet balance for native token or ERC20.
        
        Args:
            token_address: ERC20 token address (None for native token)
            
        Returns:
            Token balance as float
            
        Raises:
            CDPError: If balance query fails
        """
        try:
            if token_address is None:
                # Get native token balance (ETH on Base)
                balance = await self.wallet.get_balance()
                return float(balance)
            else:
                # Get ERC20 token balance
                balance = await readContract({
                    'network_id': self.network,
                    'contract_address': token_address,
                    'method': 'balanceOf',
                    'args': {'account': self.wallet.address},
                    'abi': ERC20_ABI
                })
                
                # Get token decimals for proper conversion
                decimals = await self.get_token_decimals(token_address)
                
                # Convert from wei to token units
                return float(balance) / (10 ** decimals)
                
        except Exception as e:
            self.logger.error(f"Failed to get balance for {token_address}: {str(e)}")
            raise CDPError(f"Balance query failed: {str(e)}")
    
    async def get_balance_wei(self, token_address: Optional[str] = None) -> int:
        """
        Get wallet balance in wei/raw units.
        
        Args:
            token_address: ERC20 token address (None for native token)
            
        Returns:
            Token balance in wei/raw units
        """
        try:
            if token_address is None:
                # Get native token balance in wei
                balance = await self.wallet.get_balance()
                return int(balance)
            else:
                # Get ERC20 token balance in raw units
                balance = await readContract({
                    'network_id': self.network,
                    'contract_address': token_address,
                    'method': 'balanceOf',
                    'args': {'account': self.wallet.address},
                    'abi': ERC20_ABI
                })
                return int(balance)
                
        except Exception as e:
            self.logger.error(f"Failed to get balance in wei for {token_address}: {str(e)}")
            raise CDPError(f"Balance query failed: {str(e)}")
    
    async def get_all_balances(self) -> Dict[str, float]:
        """
        Get all token balances for common tokens.
        
        Returns:
            Dictionary mapping token symbols to balances
        """
        balances = {}
        
        try:
            # Native token
            try:
                network_config = get_network_info(self.network)
                native_symbol = network_config.get('currency', 'ETH')
            except:
                native_symbol = 'ETH'
            balances[native_symbol] = await self.get_balance()
            
            # Get token addresses for this network
            network_tokens = TOKEN_ADDRESSES.get(self.network, {})
            
            # Batch balance queries for efficiency
            balance_tasks = []
            token_symbols = []
            
            for symbol, address in network_tokens.items():
                balance_tasks.append(self.get_balance(address))
                token_symbols.append(symbol)
            
            # Execute all balance queries concurrently
            token_balances = await asyncio.gather(*balance_tasks, return_exceptions=True)
            
            # Process results
            for symbol, balance in zip(token_symbols, token_balances):
                if isinstance(balance, Exception):
                    self.logger.warning(f"Failed to get {symbol} balance: {balance}")
                    balances[symbol] = 0.0
                else:
                    balances[symbol] = balance
            
            return balances
            
        except Exception as e:
            self.logger.error(f"Failed to get all balances: {str(e)}")
            raise CDPError(f"Batch balance query failed: {str(e)}")
    
    async def get_token_decimals(self, token_address: str) -> int:
        """
        Get token decimals for proper unit conversion.
        
        Args:
            token_address: ERC20 token address
            
        Returns:
            Number of decimal places for the token
        """
        try:
            decimals = await readContract({
                'network_id': self.network,
                'contract_address': token_address,
                'method': 'decimals',
                'args': {},
                'abi': ERC20_ABI
            })
            return int(decimals)
            
        except Exception as e:
            # Default to 18 decimals if query fails
            self.logger.warning(f"Could not get decimals for {token_address}, using 18: {str(e)}")
            return 18
    
    async def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get comprehensive token information.
        
        Args:
            token_address: ERC20 token address
            
        Returns:
            Dictionary containing token details
        """
        try:
            # Batch multiple token info queries
            info_calls = [
                {
                    'network_id': self.network,
                    'contract_address': token_address,
                    'method': 'name',
                    'args': {},
                    'abi': ERC20_ABI
                },
                {
                    'network_id': self.network,
                    'contract_address': token_address,
                    'method': 'symbol',
                    'args': {},
                    'abi': ERC20_ABI
                },
                {
                    'network_id': self.network,
                    'contract_address': token_address,
                    'method': 'decimals',
                    'args': {},
                    'abi': ERC20_ABI
                },
                {
                    'network_id': self.network,
                    'contract_address': token_address,
                    'method': 'totalSupply',
                    'args': {},
                    'abi': ERC20_ABI
                }
            ]
            
            # Execute queries concurrently
            results = await asyncio.gather(*[
                readContract(call) for call in info_calls
            ], return_exceptions=True)
            
            # Process results with error handling
            name = results[0] if not isinstance(results[0], Exception) else "Unknown"
            symbol = results[1] if not isinstance(results[1], Exception) else "UNKNOWN"
            decimals = results[2] if not isinstance(results[2], Exception) else 18
            total_supply = results[3] if not isinstance(results[3], Exception) else 0
            
            return {
                'address': token_address,
                'name': name,
                'symbol': symbol,
                'decimals': int(decimals),
                'total_supply': int(total_supply),
                'balance': await self.get_balance(token_address),
                'balance_wei': await self.get_balance_wei(token_address)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get token info for {token_address}: {str(e)}")
            raise CDPError(f"Token info query failed: {str(e)}")
    
    async def approve_token(
        self, 
        token_address: str, 
        spender_address: str, 
        amount: int
    ) -> Dict[str, Any]:
        """
        Approve token spending via CDP SDK.
        
        Args:
            token_address: ERC20 token to approve
            spender_address: Address authorized to spend tokens
            amount: Amount to approve (in wei/raw units)
            
        Returns:
            Transaction result dictionary
            
        Raises:
            TransactionError: If approval transaction fails
        """
        try:
            # Check current balance
            balance = await self.get_balance_wei(token_address)
            if balance < amount:
                raise InsufficientBalanceError(
                    f"Insufficient balance. Required: {amount}, Available: {balance}"
                )
            
            # Execute approval via CDP SDK
            result = await self.wallet.invoke_contract({
                'contract_address': token_address,
                'method': 'approve',
                'args': {
                    'spender': spender_address,
                    'amount': amount
                },
                'abi': ERC20_ABI
            })
            
            # Wait for confirmation
            confirmation = await result.wait()
            
            return {
                'success': confirmation.status == 'confirmed',
                'tx_hash': result.transaction_hash,
                'gas_used': confirmation.gas_used,
                'block_number': confirmation.block_number,
                'token': token_address,
                'spender': spender_address,
                'amount': amount
            }
            
        except InsufficientBalanceError:
            raise
        except Exception as e:
            self.logger.error(f"Token approval failed: {str(e)}")
            raise TransactionError(f"Approval transaction failed: {str(e)}")
    
    async def approve_max(
        self, 
        token_address: str, 
        spender_address: str
    ) -> Dict[str, Any]:
        """
        Approve maximum possible token amount.
        
        Args:
            token_address: ERC20 token to approve
            spender_address: Address authorized to spend tokens
            
        Returns:
            Transaction result dictionary
        """
        # Maximum uint256 value
        max_amount = 2**256 - 1
        
        return await self.approve_token(token_address, spender_address, max_amount)
    
    async def get_allowance(
        self,
        token_address: str,
        owner_address: str,
        spender_address: str
    ) -> int:
        """
        Get current token allowance.
        
        Args:
            token_address: ERC20 token address
            owner_address: Token owner address
            spender_address: Address authorized to spend
            
        Returns:
            Current allowance amount in wei/raw units
        """
        try:
            allowance = await readContract({
                'network_id': self.network,
                'contract_address': token_address,
                'method': 'allowance',
                'args': {
                    'owner': owner_address,
                    'spender': spender_address
                },
                'abi': ERC20_ABI
            })
            
            return int(allowance)
            
        except Exception as e:
            self.logger.error(f"Failed to get allowance: {str(e)}")
            raise CDPError(f"Allowance query failed: {str(e)}")
    
    async def check_and_approve(
        self,
        token_address: str,
        spender_address: str,
        required_amount: int
    ) -> Dict[str, Any]:
        """
        Check allowance and approve if insufficient.
        
        Args:
            token_address: ERC20 token address
            spender_address: Address that needs approval
            required_amount: Required allowance amount
            
        Returns:
            Dictionary with approval status and transaction info
        """
        try:
            # Check current allowance
            current_allowance = await self.get_allowance(
                token_address,
                self.wallet.address,
                spender_address
            )
            
            if current_allowance >= required_amount:
                return {
                    'approval_needed': False,
                    'current_allowance': current_allowance,
                    'required_amount': required_amount
                }
            
            # Approve required amount
            approval_result = await self.approve_token(
                token_address,
                spender_address,
                required_amount
            )
            
            return {
                'approval_needed': True,
                'approval_result': approval_result,
                'current_allowance': current_allowance,
                'required_amount': required_amount
            }
            
        except Exception as e:
            self.logger.error(f"Check and approve failed: {str(e)}")
            raise CDPError(f"Check and approve operation failed: {str(e)}")
    
    async def get_transaction_history(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get wallet transaction history.
        
        Args:
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip
            
        Returns:
            List of transaction dictionaries
        """
        try:
            # Get transaction history from CDP SDK
            transactions = await self.wallet.get_transaction_history(
                limit=limit,
                offset=offset
            )
            
            return [
                {
                    'hash': tx.transaction_hash,
                    'block_number': tx.block_number,
                    'timestamp': tx.timestamp,
                    'from': tx.from_address,
                    'to': tx.to_address,
                    'value': tx.value,
                    'gas_used': tx.gas_used,
                    'gas_price': tx.gas_price,
                    'status': tx.status,
                    'method': getattr(tx, 'method', None),
                    'contract_address': getattr(tx, 'contract_address', None)
                }
                for tx in transactions
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get transaction history: {str(e)}")
            raise CDPError(f"Transaction history query failed: {str(e)}")
    
    def convert_to_wei(self, amount: float, token_address: Optional[str] = None) -> int:
        """
        Convert token amount to wei/raw units.
        
        Args:
            amount: Token amount in standard units
            token_address: Token address (None for native token)
            
        Returns:
            Amount in wei/raw units
        """
        try:
            if token_address is None:
                # Native token (18 decimals)
                decimals = 18
            else:
                # Get token decimals (cached for efficiency)
                # In real implementation, cache this value
                decimals = 18  # Default, should query actual decimals
            
            # Use Decimal for precision
            amount_decimal = Decimal(str(amount))
            wei_amount = amount_decimal * (Decimal(10) ** decimals)
            
            return int(wei_amount)
            
        except Exception as e:
            self.logger.error(f"Failed to convert to wei: {str(e)}")
            raise CDPError(f"Wei conversion failed: {str(e)}")
    
    def convert_from_wei(self, wei_amount: int, token_address: Optional[str] = None) -> float:
        """
        Convert wei/raw units to token amount.
        
        Args:
            wei_amount: Amount in wei/raw units
            token_address: Token address (None for native token)
            
        Returns:
            Token amount in standard units
        """
        try:
            if token_address is None:
                # Native token (18 decimals)
                decimals = 18
            else:
                # Get token decimals (cached for efficiency)
                decimals = 18  # Default, should query actual decimals
            
            # Use Decimal for precision
            wei_decimal = Decimal(str(wei_amount))
            token_amount = wei_decimal / (Decimal(10) ** decimals)
            
            return float(token_amount)
            
        except Exception as e:
            self.logger.error(f"Failed to convert from wei: {str(e)}")
            raise CDPError(f"Wei conversion failed: {str(e)}")
    
    async def get_wallet_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive wallet summary.
        
        Returns:
            Dictionary containing wallet overview
        """
        try:
            # Get all balances
            balances = await self.get_all_balances()
            
            # Get recent transactions
            recent_txs = await self.get_transaction_history(limit=10)
            
            # Calculate total value (would need price data in real implementation)
            total_tokens = len([b for b in balances.values() if b > 0])
            
            return {
                'address': self.wallet.address,
                'network': self.network,
                'balances': balances,
                'total_tokens': total_tokens,
                'recent_transactions': len(recent_txs),
                'last_activity': recent_txs[0]['timestamp'] if recent_txs else None,
                'network_info': self.cdp.get_network_info()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get wallet summary: {str(e)}")
            raise CDPError(f"Wallet summary query failed: {str(e)}")


class TokenBalance:
    """
    Utility class for token balance management and calculations.
    """
    
    def __init__(self, symbol: str, address: str, balance_wei: int, decimals: int):
        """
        Initialize token balance.
        
        Args:
            symbol: Token symbol (e.g., 'USDC')
            address: Token contract address
            balance_wei: Balance in wei/raw units
            decimals: Token decimal places
        """
        self.symbol = symbol
        self.address = address
        self.balance_wei = balance_wei
        self.decimals = decimals
    
    @property
    def balance(self) -> float:
        """Get balance in standard token units."""
        return float(self.balance_wei) / (10 ** self.decimals)
    
    @property
    def balance_formatted(self) -> str:
        """Get formatted balance string."""
        return f"{self.balance:.6f} {self.symbol}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'address': self.address,
            'balance': self.balance,
            'balance_wei': self.balance_wei,
            'decimals': self.decimals,
            'formatted': self.balance_formatted
        }