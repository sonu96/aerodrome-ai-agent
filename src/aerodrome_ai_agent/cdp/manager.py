"""
CDP Manager - Main interface for Coinbase Developer Platform SDK

This class provides the primary interface for all blockchain operations through
CDP SDK, including wallet management, contract interactions, and transaction handling.
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta

try:
    from cdp import Cdp, Wallet, WalletData
    from cdp.errors import APIError, InvalidAddressError
except ImportError:
    # For development/testing when CDP SDK isn't available
    logging.warning("CDP SDK not available, using mock implementations")
    Cdp = None
    Wallet = None
    WalletData = None
    APIError = Exception
    InvalidAddressError = Exception

from .config import CDPConfig
from .wallet import WalletManager
from .contracts import ContractManager


logger = logging.getLogger(__name__)


class CDPManager:
    """
    Main CDP SDK manager for all blockchain operations
    
    Handles wallet management, contract interactions, transaction execution,
    and blockchain data retrieval through CDP SDK.
    """
    
    def __init__(self, config: CDPConfig):
        self.config = config
        self.config.validate()
        
        # Initialize CDP if available
        if Cdp:
            Cdp.configure(
                api_key_name=config.api_key_name,
                api_key_private_key=config.api_key_private_key,
                network_id=config.network_id
            )
        
        # Initialize managers
        self.wallet_manager = WalletManager(config)
        self.contract_manager = ContractManager(config)
        
        # Runtime state
        self._wallet: Optional[Wallet] = None
        self._rate_limiter = asyncio.Semaphore(config.requests_per_second)
        self._last_request_time = datetime.now()
        
        logger.info("CDP Manager initialized successfully")
    
    async def initialize_wallet(self) -> Dict[str, Any]:
        """Initialize or load wallet"""
        try:
            if self.config.wallet_id:
                # Load existing wallet
                self._wallet = await self.wallet_manager.load_wallet(
                    self.config.wallet_id,
                    self.config.wallet_seed
                )
                logger.info(f"Loaded existing wallet: {self._wallet.id}")
            else:
                # Create new wallet
                self._wallet = await self.wallet_manager.create_wallet()
                logger.info(f"Created new wallet: {self._wallet.id}")
            
            # Get wallet info
            wallet_info = await self.get_wallet_info()
            return wallet_info
            
        except Exception as e:
            logger.error(f"Failed to initialize wallet: {e}")
            raise
    
    async def get_wallet_info(self) -> Dict[str, Any]:
        """Get current wallet information"""
        if not self._wallet:
            raise RuntimeError("Wallet not initialized")
        
        try:
            balance = await self._wallet.balance("eth")
            addresses = await self._wallet.addresses
            
            return {
                "wallet_id": self._wallet.id,
                "network": self._wallet.network_id,
                "default_address": str(addresses[0]) if addresses else None,
                "balance_eth": float(balance.amount),
                "addresses": [str(addr) for addr in addresses]
            }
            
        except Exception as e:
            logger.error(f"Failed to get wallet info: {e}")
            raise
    
    async def get_balances(self, tokens: List[str] = None) -> Dict[str, float]:
        """Get token balances for the wallet"""
        if not self._wallet:
            raise RuntimeError("Wallet not initialized")
        
        try:
            balances = {}
            
            # Get ETH balance
            eth_balance = await self._wallet.balance("eth")
            balances["eth"] = float(eth_balance.amount)
            
            # Get specified token balances
            if tokens:
                for token in tokens:
                    try:
                        token_balance = await self._wallet.balance(token)
                        balances[token] = float(token_balance.amount)
                    except Exception as e:
                        logger.warning(f"Failed to get balance for {token}: {e}")
                        balances[token] = 0.0
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            raise
    
    async def read_contract(
        self,
        contract_address: str,
        method: str,
        abi: List[Dict],
        args: List[Any] = None
    ) -> Any:
        """Read from a smart contract"""
        return await self.contract_manager.read_contract(
            contract_address, method, abi, args or []
        )
    
    async def invoke_contract(
        self,
        contract_address: str,
        method: str,
        abi: List[Dict],
        args: Dict[str, Any] = None,
        value: float = 0.0
    ) -> Dict[str, Any]:
        """Invoke a smart contract method"""
        if not self._wallet:
            raise RuntimeError("Wallet not initialized")
        
        return await self.contract_manager.invoke_contract(
            self._wallet, contract_address, method, abi, args or {}, value
        )
    
    async def simulate_transaction(
        self,
        action: Dict[str, Any],
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Simulate a transaction before execution"""
        try:
            # This would use CDP's transaction simulation capabilities
            # For now, we'll return a basic simulation result
            
            simulation = {
                "success": True,
                "profitable": True,
                "gas_estimate": 150000,
                "gas_price": await self.get_gas_price(),
                "expected_output": action.get("expected_output", 0),
                "slippage": params.get("slippage", 0.01) if params else 0.01,
                "timestamp": datetime.now()
            }
            
            # Add action-specific simulation logic
            if action.get("type") == "SWAP":
                simulation.update(await self._simulate_swap(action, params))
            elif action.get("type") == "ADD_LIQUIDITY":
                simulation.update(await self._simulate_add_liquidity(action, params))
            elif action.get("type") == "REMOVE_LIQUIDITY":
                simulation.update(await self._simulate_remove_liquidity(action, params))
            
            return simulation
            
        except Exception as e:
            logger.error(f"Transaction simulation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def _simulate_swap(self, action: Dict, params: Dict = None) -> Dict:
        """Simulate swap transaction"""
        return {
            "swap_rate": action.get("rate", 1.0),
            "price_impact": action.get("price_impact", 0.001),
            "minimum_output": action.get("minimum_output", 0)
        }
    
    async def _simulate_add_liquidity(self, action: Dict, params: Dict = None) -> Dict:
        """Simulate add liquidity transaction"""
        return {
            "lp_tokens": action.get("expected_lp_tokens", 0),
            "token_a_used": action.get("token_a_amount", 0),
            "token_b_used": action.get("token_b_amount", 0)
        }
    
    async def _simulate_remove_liquidity(self, action: Dict, params: Dict = None) -> Dict:
        """Simulate remove liquidity transaction"""
        return {
            "token_a_received": action.get("expected_token_a", 0),
            "token_b_received": action.get("expected_token_b", 0),
            "lp_tokens_burned": action.get("lp_tokens", 0)
        }
    
    async def get_gas_price(self) -> float:
        """Get current gas price"""
        try:
            # This would use CDP's gas price estimation
            # For now, return a mock value
            return 20.0  # gwei
            
        except Exception as e:
            logger.warning(f"Failed to get gas price: {e}")
            return 25.0  # fallback value
    
    async def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status and details"""
        try:
            # This would query transaction status via CDP
            # For now, return mock status
            return {
                "hash": tx_hash,
                "status": "confirmed",
                "block_number": 12345678,
                "gas_used": 150000,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {
                "hash": tx_hash,
                "status": "unknown",
                "error": str(e)
            }
    
    async def wait_for_confirmation(
        self, 
        tx_hash: str, 
        confirmations: int = None
    ) -> Dict[str, Any]:
        """Wait for transaction confirmation"""
        confirmations = confirmations or self.config.min_confirmations
        timeout = self.config.transaction_timeout
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            status = await self.get_transaction_status(tx_hash)
            
            if status.get("status") == "confirmed":
                logger.info(f"Transaction {tx_hash} confirmed")
                return status
            elif status.get("status") == "failed":
                logger.error(f"Transaction {tx_hash} failed")
                return status
            
            await asyncio.sleep(2)  # Check every 2 seconds
        
        logger.warning(f"Transaction {tx_hash} confirmation timeout")
        return {
            "hash": tx_hash,
            "status": "timeout",
            "error": "Confirmation timeout"
        }
    
    async def get_pool_data(self, pool_address: str) -> Dict[str, Any]:
        """Get comprehensive pool data"""
        return await self.contract_manager.get_pool_data(pool_address)
    
    async def get_top_pools(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top pools by TVL"""
        return await self.contract_manager.get_top_pools(limit)
    
    async def close(self):
        """Clean up resources"""
        logger.info("CDP Manager closed")
    
    def __del__(self):
        """Destructor"""
        try:
            asyncio.create_task(self.close())
        except:
            pass