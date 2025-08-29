"""
Aerodrome Protocol Client using QuickNode API

This client provides comprehensive access to Aerodrome DEX operations including:
- Pool operations (search, details, analytics)
- Token operations (prices, search, batch info)  
- Swap operations (quotes, transactions)
- Real-time WebSocket updates
- Transaction tracking

Uses QuickNode Aerodrome API addon 1051 for Base mainnet integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json

import aiohttp
import websockets
from web3 import Web3
from web3.middleware import geth_poa_middleware
from hexbytes import HexBytes
import structlog

# Aerodrome contract addresses on Base mainnet
AERODROME_CONTRACTS = {
    "router": "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
    "voter": "0x16613524e02ad97eDfeF371bC883F2F5d6C480A5", 
    "aero_token": "0x940181a94A35A4569E4529A3CDfB74e38FD98631",
    "factory": "0x420DD381b31aEf6683db6B902084cB0FFECe40Da",
    "gauge_factory": "0x12527c6f82E0c6EbF6b9Ae7F80E3e5c02B6D34b5"
}

# Base mainnet configuration
BASE_MAINNET_CONFIG = {
    "chain_id": 8453,
    "rpc_url": "https://mainnet.base.org",
    "explorer_url": "https://basescan.org"
}

logger = structlog.get_logger(__name__)


class SwapType(Enum):
    """Swap operation types"""
    EXACT_INPUT = "exactInput"
    EXACT_OUTPUT = "exactOutput"


class PoolType(Enum):
    """Pool types in Aerodrome"""
    STABLE = "stable"
    VOLATILE = "volatile"
    CL = "concentrated_liquidity"


@dataclass
class TokenInfo:
    """Token information structure"""
    address: str
    symbol: str
    name: str
    decimals: int
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None


@dataclass
class PoolInfo:
    """Pool information structure"""
    address: str
    token0: TokenInfo
    token1: TokenInfo
    pool_type: PoolType
    fee: float
    tvl_usd: float
    volume_24h: float
    volume_7d: float
    fees_24h: float
    apr: float
    reserves: Dict[str, float]
    gauge_address: Optional[str] = None
    bribe_address: Optional[str] = None
    is_active: bool = True


@dataclass  
class SwapQuote:
    """Swap quote information"""
    amount_in: str
    amount_out: str
    price_impact: float
    route: List[str]
    gas_estimate: int
    min_amount_out: str
    deadline: int


@dataclass
class Transaction:
    """Transaction information"""
    hash: str
    status: str
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    gas_price: Optional[int] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire rate limit permission"""
        now = datetime.now()
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(seconds=self.time_window)]
        
        if len(self.calls) >= self.max_calls:
            wait_time = self.time_window - (now - self.calls[0]).seconds
            await asyncio.sleep(wait_time)
            await self.acquire()
        
        self.calls.append(now)


class AerodromeClient:
    """
    Comprehensive Aerodrome protocol client using QuickNode API
    
    Provides access to all Aerodrome DEX functionality including pools,
    tokens, swaps, and real-time data via WebSocket connections.
    """
    
    def __init__(
        self,
        quicknode_url: str,
        quicknode_api_key: Optional[str] = None,
        rate_limit_calls: int = 100,
        rate_limit_window: int = 60,
        enable_websocket: bool = True
    ):
        """
        Initialize Aerodrome client
        
        Args:
            quicknode_url: QuickNode RPC endpoint URL
            quicknode_api_key: Optional QuickNode API key
            rate_limit_calls: Max API calls per time window
            rate_limit_window: Time window in seconds for rate limiting
            enable_websocket: Whether to enable WebSocket connections
        """
        self.quicknode_url = quicknode_url
        self.api_key = quicknode_api_key
        self.enable_websocket = enable_websocket
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(quicknode_url))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connections
        self.ws_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.ws_callbacks: Dict[str, List[Callable]] = {}
        
        # Cache for frequently accessed data
        self._token_cache: Dict[str, TokenInfo] = {}
        self._pool_cache: Dict[str, PoolInfo] = {}
        
        logger.info("Aerodrome client initialized", 
                   quicknode_url=quicknode_url,
                   enable_websocket=enable_websocket)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Initialize connections"""
        # Create HTTP session with proper headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AerodromeClient/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers
        )
        
        # Verify connection
        try:
            block = await self._make_rpc_call("eth_blockNumber")
            logger.info("Connected to Base mainnet", block_number=int(block, 16))
        except Exception as e:
            logger.error("Failed to connect to Base mainnet", error=str(e))
            raise
    
    async def disconnect(self):
        """Close all connections"""
        if self.session:
            await self.session.close()
        
        # Close WebSocket connections
        for ws_name, ws in self.ws_connections.items():
            try:
                await ws.close()
                logger.debug("Closed WebSocket connection", name=ws_name)
            except Exception as e:
                logger.warning("Error closing WebSocket", name=ws_name, error=str(e))
        
        self.ws_connections.clear()
        logger.info("Aerodrome client disconnected")
    
    async def _make_rpc_call(self, method: str, params: List[Any] = None) -> Any:
        """Make RPC call to QuickNode endpoint"""
        await self.rate_limiter.acquire()
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": 1
        }
        
        try:
            async with self.session.post(self.quicknode_url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "error" in data:
                    raise Exception(f"RPC Error: {data['error']}")
                
                return data.get("result")
        
        except Exception as e:
            logger.error("RPC call failed", method=method, error=str(e))
            raise
    
    async def _make_api_call(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API call to QuickNode Aerodrome addon"""
        await self.rate_limiter.acquire()
        
        # QuickNode Aerodrome API endpoint format
        url = f"{self.quicknode_url.replace('/v1/', '/addon/1051/')}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        
        except Exception as e:
            logger.error("API call failed", endpoint=endpoint, error=str(e))
            raise
    
    # ===== TOKEN OPERATIONS =====
    
    async def get_token_info(self, token_address: str, use_cache: bool = True) -> TokenInfo:
        """
        Get comprehensive token information
        
        Args:
            token_address: Token contract address
            use_cache: Whether to use cached data
            
        Returns:
            TokenInfo object with token details
        """
        if use_cache and token_address in self._token_cache:
            return self._token_cache[token_address]
        
        try:
            # Get basic token info from contract
            token_data = await self._make_api_call(f"tokens/{token_address}")
            
            # Get price data
            price_data = await self._make_api_call(f"tokens/{token_address}/price")
            
            token_info = TokenInfo(
                address=token_address,
                symbol=token_data.get("symbol", ""),
                name=token_data.get("name", ""),
                decimals=token_data.get("decimals", 18),
                price_usd=price_data.get("price_usd"),
                market_cap=price_data.get("market_cap"),
                volume_24h=price_data.get("volume_24h"),
                price_change_24h=price_data.get("price_change_24h")
            )
            
            if use_cache:
                self._token_cache[token_address] = token_info
            
            logger.debug("Retrieved token info", symbol=token_info.symbol, address=token_address)
            return token_info
            
        except Exception as e:
            logger.error("Failed to get token info", address=token_address, error=str(e))
            raise
    
    async def get_token_prices(self, token_addresses: List[str]) -> Dict[str, float]:
        """
        Get prices for multiple tokens in batch
        
        Args:
            token_addresses: List of token addresses
            
        Returns:
            Dictionary mapping addresses to USD prices
        """
        try:
            params = {"addresses": ",".join(token_addresses)}
            price_data = await self._make_api_call("tokens/prices", params)
            
            prices = {}
            for address, data in price_data.items():
                prices[address] = data.get("price_usd", 0.0)
            
            logger.debug("Retrieved batch token prices", count=len(prices))
            return prices
            
        except Exception as e:
            logger.error("Failed to get token prices", addresses=token_addresses, error=str(e))
            raise
    
    async def search_tokens(self, query: str, limit: int = 20) -> List[TokenInfo]:
        """
        Search for tokens by name or symbol
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching TokenInfo objects
        """
        try:
            params = {"q": query, "limit": limit}
            search_data = await self._make_api_call("tokens/search", params)
            
            tokens = []
            for token_data in search_data.get("tokens", []):
                token_info = TokenInfo(
                    address=token_data["address"],
                    symbol=token_data["symbol"],
                    name=token_data["name"],
                    decimals=token_data["decimals"],
                    price_usd=token_data.get("price_usd")
                )
                tokens.append(token_info)
            
            logger.debug("Token search completed", query=query, results=len(tokens))
            return tokens
            
        except Exception as e:
            logger.error("Token search failed", query=query, error=str(e))
            raise
    
    # ===== POOL OPERATIONS =====
    
    async def get_pool_info(self, pool_address: str, use_cache: bool = True) -> PoolInfo:
        """
        Get comprehensive pool information
        
        Args:
            pool_address: Pool contract address
            use_cache: Whether to use cached data
            
        Returns:
            PoolInfo object with pool details
        """
        if use_cache and pool_address in self._pool_cache:
            return self._pool_cache[pool_address]
        
        try:
            # Get pool data from API
            pool_data = await self._make_api_call(f"pools/{pool_address}")
            
            # Get token information
            token0_info = await self.get_token_info(pool_data["token0"], use_cache)
            token1_info = await self.get_token_info(pool_data["token1"], use_cache)
            
            # Determine pool type
            pool_type = PoolType.STABLE if pool_data.get("stable") else PoolType.VOLATILE
            if pool_data.get("tick_spacing"):  # CL pools have tick spacing
                pool_type = PoolType.CL
            
            pool_info = PoolInfo(
                address=pool_address,
                token0=token0_info,
                token1=token1_info,
                pool_type=pool_type,
                fee=pool_data.get("fee", 0) / 10000,  # Convert from basis points
                tvl_usd=pool_data.get("tvl_usd", 0.0),
                volume_24h=pool_data.get("volume_24h", 0.0),
                volume_7d=pool_data.get("volume_7d", 0.0),
                fees_24h=pool_data.get("fees_24h", 0.0),
                apr=pool_data.get("apr", 0.0),
                reserves={
                    "token0": pool_data.get("reserve0", 0.0),
                    "token1": pool_data.get("reserve1", 0.0)
                },
                gauge_address=pool_data.get("gauge_address"),
                bribe_address=pool_data.get("bribe_address"),
                is_active=pool_data.get("is_active", True)
            )
            
            if use_cache:
                self._pool_cache[pool_address] = pool_info
            
            logger.debug("Retrieved pool info", 
                        address=pool_address,
                        type=pool_type.value,
                        tvl=pool_info.tvl_usd)
            return pool_info
            
        except Exception as e:
            logger.error("Failed to get pool info", address=pool_address, error=str(e))
            raise
    
    async def search_pools(
        self,
        token0: Optional[str] = None,
        token1: Optional[str] = None,
        pool_type: Optional[PoolType] = None,
        min_tvl: Optional[float] = None,
        sort_by: str = "tvl",
        limit: int = 50
    ) -> List[PoolInfo]:
        """
        Search for pools with filters
        
        Args:
            token0: First token address filter
            token1: Second token address filter  
            pool_type: Pool type filter
            min_tvl: Minimum TVL filter
            sort_by: Sort field (tvl, volume_24h, apr)
            limit: Maximum number of results
            
        Returns:
            List of matching PoolInfo objects
        """
        try:
            params = {
                "sort_by": sort_by,
                "limit": limit
            }
            
            if token0:
                params["token0"] = token0
            if token1:
                params["token1"] = token1
            if pool_type:
                params["type"] = pool_type.value
            if min_tvl:
                params["min_tvl"] = min_tvl
            
            search_data = await self._make_api_call("pools/search", params)
            
            pools = []
            for pool_data in search_data.get("pools", []):
                pool_info = await self.get_pool_info(pool_data["address"], use_cache=True)
                pools.append(pool_info)
            
            logger.debug("Pool search completed", 
                        filters=params,
                        results=len(pools))
            return pools
            
        except Exception as e:
            logger.error("Pool search failed", error=str(e))
            raise
    
    async def get_pool_analytics(
        self,
        pool_address: str,
        timeframe: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get detailed pool analytics
        
        Args:
            pool_address: Pool contract address
            timeframe: Analytics timeframe (1h, 24h, 7d, 30d)
            
        Returns:
            Dictionary with analytics data
        """
        try:
            params = {"timeframe": timeframe}
            analytics = await self._make_api_call(f"pools/{pool_address}/analytics", params)
            
            logger.debug("Retrieved pool analytics",
                        address=pool_address,
                        timeframe=timeframe)
            return analytics
            
        except Exception as e:
            logger.error("Failed to get pool analytics", 
                        address=pool_address, 
                        error=str(e))
            raise
    
    # ===== SWAP OPERATIONS =====
    
    async def get_swap_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: str,
        swap_type: SwapType = SwapType.EXACT_INPUT,
        slippage_bps: int = 50  # 0.5% default slippage
    ) -> SwapQuote:
        """
        Get swap quote for token exchange
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount (in token units)
            swap_type: Type of swap (exact input/output)
            slippage_bps: Slippage tolerance in basis points
            
        Returns:
            SwapQuote object with quote details
        """
        try:
            params = {
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": amount_in,
                "type": swap_type.value,
                "slippage_bps": slippage_bps
            }
            
            quote_data = await self._make_api_call("swaps/quote", params)
            
            quote = SwapQuote(
                amount_in=quote_data["amount_in"],
                amount_out=quote_data["amount_out"],
                price_impact=quote_data.get("price_impact", 0.0),
                route=quote_data.get("route", []),
                gas_estimate=quote_data.get("gas_estimate", 200000),
                min_amount_out=quote_data["min_amount_out"],
                deadline=quote_data.get("deadline", int(datetime.now().timestamp()) + 1200)
            )
            
            logger.debug("Generated swap quote",
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=amount_in,
                        amount_out=quote.amount_out)
            return quote
            
        except Exception as e:
            logger.error("Failed to get swap quote", 
                        token_in=token_in,
                        token_out=token_out,
                        error=str(e))
            raise
    
    async def build_swap_transaction(
        self,
        quote: SwapQuote,
        recipient: str,
        token_in: str,
        token_out: str
    ) -> Dict[str, Any]:
        """
        Build swap transaction data
        
        Args:
            quote: SwapQuote from get_swap_quote
            recipient: Recipient address
            token_in: Input token address
            token_out: Output token address
            
        Returns:
            Transaction data dictionary
        """
        try:
            params = {
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": quote.amount_in,
                "min_amount_out": quote.min_amount_out,
                "recipient": recipient,
                "deadline": quote.deadline,
                "route": quote.route
            }
            
            tx_data = await self._make_api_call("swaps/build", params)
            
            logger.debug("Built swap transaction", 
                        recipient=recipient,
                        gas_limit=tx_data.get("gas_limit"))
            return tx_data
            
        except Exception as e:
            logger.error("Failed to build swap transaction", error=str(e))
            raise
    
    # ===== TRANSACTION TRACKING =====
    
    async def track_transaction(self, tx_hash: str) -> Transaction:
        """
        Track transaction status and details
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction object with current status
        """
        try:
            # Get transaction receipt
            tx_receipt = await self._make_rpc_call("eth_getTransactionReceipt", [tx_hash])
            
            if not tx_receipt:
                return Transaction(hash=tx_hash, status="pending")
            
            # Get transaction details
            tx_data = await self._make_rpc_call("eth_getTransactionByHash", [tx_hash])
            
            # Get block timestamp
            block_data = await self._make_rpc_call("eth_getBlockByNumber", [tx_receipt["blockNumber"], False])
            
            transaction = Transaction(
                hash=tx_hash,
                status="success" if tx_receipt["status"] == "0x1" else "failed",
                block_number=int(tx_receipt["blockNumber"], 16),
                gas_used=int(tx_receipt["gasUsed"], 16),
                gas_price=int(tx_data["gasPrice"], 16),
                timestamp=datetime.fromtimestamp(int(block_data["timestamp"], 16))
            )
            
            logger.debug("Retrieved transaction info",
                        hash=tx_hash,
                        status=transaction.status,
                        block=transaction.block_number)
            return transaction
            
        except Exception as e:
            logger.error("Failed to track transaction", hash=tx_hash, error=str(e))
            return Transaction(hash=tx_hash, status="error", error=str(e))
    
    async def wait_for_transaction(
        self,
        tx_hash: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> Transaction:
        """
        Wait for transaction confirmation
        
        Args:
            tx_hash: Transaction hash
            timeout: Maximum wait time in seconds
            poll_interval: Poll interval in seconds
            
        Returns:
            Transaction object with final status
        """
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            transaction = await self.track_transaction(tx_hash)
            
            if transaction.status in ["success", "failed", "error"]:
                return transaction
            
            await asyncio.sleep(poll_interval)
        
        logger.warning("Transaction wait timeout", hash=tx_hash, timeout=timeout)
        return Transaction(hash=tx_hash, status="timeout")
    
    # ===== WEBSOCKET OPERATIONS =====
    
    async def subscribe_to_pool_updates(
        self,
        pool_address: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to real-time pool updates via WebSocket
        
        Args:
            pool_address: Pool address to monitor
            callback: Function to call with update data
        """
        if not self.enable_websocket:
            logger.warning("WebSocket disabled, cannot subscribe to pool updates")
            return
        
        try:
            # WebSocket endpoint for pool updates
            ws_url = self.quicknode_url.replace("https://", "wss://").replace("http://", "ws://")
            ws_url = f"{ws_url}/addon/1051/pools/{pool_address}/subscribe"
            
            async with websockets.connect(ws_url) as websocket:
                self.ws_connections[f"pool_{pool_address}"] = websocket
                
                # Register callback
                if f"pool_{pool_address}" not in self.ws_callbacks:
                    self.ws_callbacks[f"pool_{pool_address}"] = []
                self.ws_callbacks[f"pool_{pool_address}"].append(callback)
                
                logger.info("Subscribed to pool updates", pool=pool_address)
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Call all registered callbacks
                        for cb in self.ws_callbacks[f"pool_{pool_address}"]:
                            try:
                                if asyncio.iscoroutinefunction(cb):
                                    await cb(data)
                                else:
                                    cb(data)
                            except Exception as e:
                                logger.error("Callback error", error=str(e))
                    
                    except json.JSONDecodeError as e:
                        logger.warning("Invalid WebSocket message", error=str(e))
                
        except Exception as e:
            logger.error("WebSocket subscription failed", 
                        pool=pool_address, 
                        error=str(e))
            # Remove from connections if failed
            self.ws_connections.pop(f"pool_{pool_address}", None)
    
    async def subscribe_to_price_updates(
        self,
        token_addresses: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to real-time price updates via WebSocket
        
        Args:
            token_addresses: List of token addresses to monitor
            callback: Function to call with price update data
        """
        if not self.enable_websocket:
            logger.warning("WebSocket disabled, cannot subscribe to price updates")
            return
        
        try:
            ws_url = self.quicknode_url.replace("https://", "wss://").replace("http://", "ws://")
            ws_url = f"{ws_url}/addon/1051/prices/subscribe"
            
            async with websockets.connect(ws_url) as websocket:
                # Subscribe to tokens
                subscribe_msg = {
                    "type": "subscribe",
                    "tokens": token_addresses
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                self.ws_connections["prices"] = websocket
                if "prices" not in self.ws_callbacks:
                    self.ws_callbacks["prices"] = []
                self.ws_callbacks["prices"].append(callback)
                
                logger.info("Subscribed to price updates", tokens=len(token_addresses))
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Call all registered callbacks
                        for cb in self.ws_callbacks["prices"]:
                            try:
                                if asyncio.iscoroutinefunction(cb):
                                    await cb(data)
                                else:
                                    cb(data)
                            except Exception as e:
                                logger.error("Price callback error", error=str(e))
                    
                    except json.JSONDecodeError as e:
                        logger.warning("Invalid price WebSocket message", error=str(e))
                
        except Exception as e:
            logger.error("Price WebSocket subscription failed", error=str(e))
            self.ws_connections.pop("prices", None)
    
    # ===== UTILITY METHODS =====
    
    def clear_cache(self):
        """Clear all cached data"""
        self._token_cache.clear()
        self._pool_cache.clear()
        logger.debug("Cleared all caches")
    
    async def get_network_info(self) -> Dict[str, Any]:
        """Get current network information"""
        try:
            block_number = await self._make_rpc_call("eth_blockNumber")
            gas_price = await self._make_rpc_call("eth_gasPrice")
            
            return {
                "chain_id": BASE_MAINNET_CONFIG["chain_id"],
                "block_number": int(block_number, 16),
                "gas_price": int(gas_price, 16),
                "rpc_url": self.quicknode_url,
                "contracts": AERODROME_CONTRACTS
            }
        except Exception as e:
            logger.error("Failed to get network info", error=str(e))
            raise


# ===== EXAMPLE USAGE =====

async def example_usage():
    """Example usage of AerodromeClient"""
    
    # Initialize client
    quicknode_url = "https://your-quicknode-endpoint.quiknode.pro/your-api-key/"
    
    async with AerodromeClient(quicknode_url) as client:
        
        # Get token information
        aero_token = await client.get_token_info(AERODROME_CONTRACTS["aero_token"])
        print(f"AERO Token: {aero_token.symbol} - ${aero_token.price_usd}")
        
        # Search for pools
        pools = await client.search_pools(min_tvl=100000, limit=10)
        print(f"Found {len(pools)} pools with >$100k TVL")
        
        # Get swap quote
        if len(pools) > 0:
            pool = pools[0]
            quote = await client.get_swap_quote(
                token_in=pool.token0.address,
                token_out=pool.token1.address,
                amount_in="1000000"  # 1 token (assuming 6 decimals)
            )
            print(f"Swap quote: {quote.amount_in} -> {quote.amount_out}")
        
        # Subscribe to pool updates (example callback)
        def pool_update_handler(data):
            print(f"Pool update: {data}")
        
        if len(pools) > 0:
            await client.subscribe_to_pool_updates(
                pools[0].address,
                pool_update_handler
            )


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())