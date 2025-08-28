# CDP SDK Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [SDK Setup](#sdk-setup)
3. [Wallet Management](#wallet-management)
4. [Smart Contract Operations](#smart-contract-operations)
5. [Aerodrome Contract Integration](#aerodrome-contract-integration)
6. [Transaction Management](#transaction-management)
7. [Event Monitoring](#event-monitoring)
8. [Error Handling](#error-handling)
9. [Best Practices](#best-practices)

## Overview

This guide details the complete integration of Coinbase Developer Platform (CDP) SDK for all blockchain operations in the Aerodrome DeFi Agent. The CDP SDK is the **EXCLUSIVE** interface for blockchain interactions - no direct web3.js or ethers.js usage.

### Key Principles
- **CDP SDK Only**: All blockchain operations through CDP SDK
- **Type Safety**: Leverage CDP's automatic ABI type inference
- **Security First**: MPC wallets for production environments
- **Error Resilience**: Built-in retry logic and error handling

## SDK Setup

### Installation and Configuration

```python
# Python Setup
from cdp_sdk import CDPClient, Wallet, SmartContract, readContract
from cdp_sdk.wallet import WalletConfig
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

class CDPManager:
    """Centralized CDP SDK manager for all blockchain operations"""
    
    def __init__(self, network: str = 'base-mainnet'):
        load_dotenv()
        
        # Initialize CDP client
        self.client = CDPClient(
            api_key_id=os.getenv('CDP_API_KEY_ID'),
            api_key_secret=os.getenv('CDP_API_KEY_SECRET')
        )
        
        # Initialize wallet
        self.wallet = self._initialize_wallet(network)
        
        # Network configuration
        self.network = network
        self.chain_id = self._get_chain_id(network)
        
    def _initialize_wallet(self, network: str) -> Wallet:
        """Initialize CDP wallet with MPC security"""
        
        wallet_config = WalletConfig(
            network_id=network,
            wallet_secret=os.getenv('CDP_WALLET_SECRET'),
            # MPC configuration for production
            use_mpc=os.getenv('ENVIRONMENT') == 'production',
            server_signer_url=os.getenv('SERVER_SIGNER_URL') if os.getenv('ENVIRONMENT') == 'production' else None
        )
        
        # Create or load existing wallet
        wallet_data = os.getenv('CDP_WALLET_DATA')
        if wallet_data:
            # Load existing wallet
            wallet = Wallet.import_wallet(wallet_data, wallet_config)
        else:
            # Create new wallet
            wallet = Wallet.create(wallet_config)
            # Save wallet data securely
            self._save_wallet_data(wallet.export())
        
        return wallet
```

```typescript
// TypeScript Setup
import { 
    CDPClient, 
    Wallet, 
    SmartContract, 
    readContract,
    WalletProvider,
    AgentKit 
} from '@coinbase/cdp-sdk';

class CDPManager {
    private client: CDPClient;
    private wallet: Wallet;
    private agentKit: AgentKit;
    
    constructor(network: string = 'base-mainnet') {
        this.initialize(network);
    }
    
    private async initialize(network: string) {
        // Initialize CDP client
        this.client = new CDPClient({
            apiKeyId: process.env.CDP_API_KEY_ID!,
            apiKeySecret: process.env.CDP_API_KEY_SECRET!
        });
        
        // Initialize wallet with MPC
        const walletProvider = await WalletProvider.configureWithWallet({
            apiKeyId: process.env.CDP_API_KEY_ID!,
            apiKeySecret: process.env.CDP_API_KEY_SECRET!,
            walletSecret: process.env.CDP_WALLET_SECRET!,
            networkId: network,
            useMPC: process.env.ENVIRONMENT === 'production'
        });
        
        // Initialize AgentKit
        this.agentKit = await AgentKit.from({
            walletProvider,
            actionProviders: []
        });
        
        this.wallet = walletProvider.getWallet();
    }
}
```

## Wallet Management

### Wallet Operations

```python
class WalletOperations:
    """CDP wallet management operations"""
    
    async def get_balance(self, token_address: Optional[str] = None) -> float:
        """Get wallet balance for native token or ERC20"""
        
        if token_address is None:
            # Get native token balance (ETH on Base)
            balance = await self.wallet.get_balance()
        else:
            # Get ERC20 token balance
            balance = await readContract({
                'network_id': self.network,
                'contract_address': token_address,
                'method': 'balanceOf',
                'args': {'account': self.wallet.address},
                'abi': ERC20_ABI
            })
        
        return float(balance)
    
    async def get_all_balances(self) -> Dict[str, float]:
        """Get all token balances"""
        
        balances = {}
        
        # Native token
        balances['ETH'] = await self.get_balance()
        
        # Common tokens on Base
        tokens = {
            'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
            'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631',
            'WETH': '0x4200000000000000000000000000000000000006',
            'DAI': '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb'
        }
        
        for symbol, address in tokens.items():
            try:
                balances[symbol] = await self.get_balance(address)
            except Exception as e:
                self.logger.warning(f"Failed to get {symbol} balance: {e}")
                balances[symbol] = 0.0
        
        return balances
    
    async def approve_token(
        self, 
        token_address: str, 
        spender_address: str, 
        amount: int
    ) -> Dict[str, Any]:
        """Approve token spending via CDP SDK"""
        
        result = await self.wallet.invoke_contract({
            'contract_address': token_address,
            'method': 'approve',
            'args': {
                'spender': spender_address,
                'amount': amount
            },
            'abi': ERC20_ABI
        })
        
        return {
            'tx_hash': result.transaction_hash,
            'success': result.status == 'confirmed'
        }
```

### Multi-Signature Wallet Support

```python
class MPCWallet:
    """MPC (2-of-2) wallet operations for production"""
    
    def __init__(self):
        self.server_signer_url = os.getenv('SERVER_SIGNER_URL')
        self.wallet = self._init_mpc_wallet()
    
    def _init_mpc_wallet(self) -> Wallet:
        """Initialize MPC wallet with server signer"""
        
        config = {
            'network_id': 'base-mainnet',
            'use_mpc': True,
            'server_signer': {
                'url': self.server_signer_url,
                'api_key': os.getenv('SERVER_SIGNER_API_KEY')
            }
        }
        
        wallet = Wallet.create_mpc_wallet(config)
        return wallet
    
    async def sign_transaction(self, tx_params: Dict) -> Dict:
        """Sign transaction with MPC wallet"""
        
        # Server signer participates automatically
        signed_tx = await self.wallet.sign_transaction(tx_params)
        
        return {
            'signed_tx': signed_tx,
            'requires_cosigner': False  # CDP handles this
        }
```

## Smart Contract Operations

### Contract Invocation

```python
class ContractOperations:
    """Smart contract operations via CDP SDK"""
    
    async def invoke_contract(
        self,
        contract_address: str,
        method: str,
        args: Dict[str, Any],
        abi: List[Dict],
        value: int = 0
    ) -> Dict[str, Any]:
        """
        Invoke any smart contract method
        
        This is the ONLY way to interact with contracts
        """
        
        try:
            # Pre-execution validation
            await self._validate_contract_call(contract_address, method, args)
            
            # Execute via CDP SDK
            result = await self.wallet.invoke_contract({
                'contract_address': contract_address,
                'method': method,
                'args': args,
                'abi': abi,
                'value': value  # For payable functions
            })
            
            # Wait for confirmation
            confirmation = await result.wait()
            
            return {
                'success': True,
                'tx_hash': result.transaction_hash,
                'gas_used': confirmation.gas_used,
                'block_number': confirmation.block_number,
                'logs': confirmation.logs
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': method,
                'contract': contract_address
            }
    
    async def read_contract(
        self,
        contract_address: str,
        method: str,
        args: Dict[str, Any] = {},
        abi: List[Dict] = None
    ) -> Any:
        """
        Read contract state without transaction
        
        CDP SDK automatically infers types when ABI is provided
        """
        
        result = await readContract({
            'network_id': self.network,
            'contract_address': contract_address,
            'method': method,
            'args': args,
            'abi': abi  # Optional - CDP can infer for standard contracts
        })
        
        return result
    
    async def batch_read(self, calls: List[Dict]) -> List[Any]:
        """Batch multiple contract reads for efficiency"""
        
        tasks = [
            self.read_contract(**call) for call in calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
```

### Contract Registration and Events

```python
class ContractRegistry:
    """Register and monitor smart contracts"""
    
    async def register_contract(
        self,
        contract_address: str,
        abi: List[Dict],
        name: str
    ) -> SmartContract:
        """Register contract for event monitoring"""
        
        contract = await SmartContract.register(
            network_id=self.network,
            contract_address=contract_address,
            abi=abi,
            contract_name=name
        )
        
        self.registered_contracts[contract_address] = contract
        
        return contract
    
    async def get_contract_events(
        self,
        contract_address: str,
        event_name: str,
        from_block: int = 'latest',
        filter_params: Dict = {}
    ) -> List[Dict]:
        """Get historical events from contract"""
        
        events = await self.client.get_smart_contract_events({
            'contract_address': contract_address,
            'event_name': event_name,
            'from_block': from_block,
            'to_block': 'latest',
            'filter': filter_params
        })
        
        return events
```

## Aerodrome Contract Integration

### Aerodrome Protocol Constants

```python
# Aerodrome Contract Addresses on Base
AERODROME_CONTRACTS = {
    'ROUTER': '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43',
    'VOTER': '0x16613524e02ad97eDfeF371bC883F2F5d6C480A5',
    'FACTORY': '0x420DD381b31aEf6683db6B902084cB0FFECe40Da',
    'MINTER': '0xeB018363F0a9Af8f91F06FEe6613a751b2A33FE5',
    'VE_AERO': '0xeBf418Fe2512e7E6bd9b87a8F0f294aCDC67e6B4',
    'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631',
    'REWARDS_DISTRIBUTOR': '0x227f65131A261548b057215bB1D5Ab2997964C7d'
}
```

### Router Operations

```python
class AerodromeRouter:
    """Aerodrome Router operations via CDP SDK"""
    
    def __init__(self, cdp_manager: CDPManager):
        self.cdp = cdp_manager
        self.router_address = AERODROME_CONTRACTS['ROUTER']
    
    async def swap_exact_tokens_for_tokens(
        self,
        amount_in: int,
        amount_out_min: int,
        routes: List[Dict],
        deadline: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute token swap through Aerodrome Router"""
        
        if deadline is None:
            deadline = int(time.time()) + 900  # 15 minutes
        
        # Build swap parameters
        swap_args = {
            'amountIn': amount_in,
            'amountOutMin': amount_out_min,
            'routes': routes,
            'to': self.cdp.wallet.address,
            'deadline': deadline
        }
        
        # Execute swap via CDP SDK
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.router_address,
            'method': 'swapExactTokensForTokens',
            'args': swap_args,
            'abi': ROUTER_ABI
        })
        
        return result
    
    async def add_liquidity(
        self,
        token_a: str,
        token_b: str,
        amount_a: int,
        amount_b: int,
        stable: bool = False
    ) -> Dict[str, Any]:
        """Add liquidity to Aerodrome pool"""
        
        # Calculate minimum amounts (2% slippage)
        amount_a_min = int(amount_a * 0.98)
        amount_b_min = int(amount_b * 0.98)
        
        args = {
            'tokenA': token_a,
            'tokenB': token_b,
            'stable': stable,
            'amountADesired': amount_a,
            'amountBDesired': amount_b,
            'amountAMin': amount_a_min,
            'amountBMin': amount_b_min,
            'to': self.cdp.wallet.address,
            'deadline': int(time.time()) + 900
        }
        
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.router_address,
            'method': 'addLiquidity',
            'args': args,
            'abi': ROUTER_ABI
        })
        
        return result
    
    async def remove_liquidity(
        self,
        token_a: str,
        token_b: str,
        liquidity: int,
        stable: bool = False
    ) -> Dict[str, Any]:
        """Remove liquidity from Aerodrome pool"""
        
        args = {
            'tokenA': token_a,
            'tokenB': token_b,
            'stable': stable,
            'liquidity': liquidity,
            'amountAMin': 0,  # Accept any amount
            'amountBMin': 0,  # Accept any amount
            'to': self.cdp.wallet.address,
            'deadline': int(time.time()) + 900
        }
        
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.router_address,
            'method': 'removeLiquidity',
            'args': args,
            'abi': ROUTER_ABI
        })
        
        return result
    
    async def get_amount_out(
        self,
        amount_in: int,
        token_in: str,
        token_out: str
    ) -> int:
        """Get expected output amount for swap"""
        
        # Find pool
        pool = await self._get_pool(token_in, token_out)
        
        if not pool:
            return 0
        
        # Read from pool contract
        amount_out = await readContract({
            'network_id': 'base-mainnet',
            'contract_address': pool['address'],
            'method': 'getAmountOut',
            'args': {
                'amountIn': amount_in,
                'tokenIn': token_in
            },
            'abi': POOL_ABI
        })
        
        return amount_out
```

### Voting Operations

```python
class AerodromeVoter:
    """Aerodrome Voter contract operations"""
    
    def __init__(self, cdp_manager: CDPManager):
        self.cdp = cdp_manager
        self.voter_address = AERODROME_CONTRACTS['VOTER']
        self.ve_address = AERODROME_CONTRACTS['VE_AERO']
    
    async def vote(
        self,
        token_id: int,
        pool_votes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Vote for gauge allocations"""
        
        pools = [vote['pool'] for vote in pool_votes]
        weights = [vote['weight'] for vote in pool_votes]
        
        # Ensure weights sum to 10000 (100%)
        total_weight = sum(weights)
        if total_weight != 10000:
            # Normalize weights
            weights = [int(w * 10000 / total_weight) for w in weights]
        
        args = {
            'tokenId': token_id,
            'poolVote': pools,
            'weights': weights
        }
        
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.voter_address,
            'method': 'vote',
            'args': args,
            'abi': VOTER_ABI
        })
        
        return result
    
    async def reset(self, token_id: int) -> Dict[str, Any]:
        """Reset votes for a veNFT"""
        
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.voter_address,
            'method': 'reset',
            'args': {'tokenId': token_id},
            'abi': VOTER_ABI
        })
        
        return result
    
    async def claim_bribes(
        self,
        bribes: List[str],
        tokens: List[List[str]],
        token_id: int
    ) -> Dict[str, Any]:
        """Claim bribes from voting"""
        
        args = {
            'bribes': bribes,
            'tokens': tokens,
            'tokenId': token_id
        }
        
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.voter_address,
            'method': 'claimBribes',
            'args': args,
            'abi': VOTER_ABI
        })
        
        return result
```

### veAERO Operations

```python
class VotingEscrow:
    """veAERO (Voting Escrow) operations"""
    
    def __init__(self, cdp_manager: CDPManager):
        self.cdp = cdp_manager
        self.ve_address = AERODROME_CONTRACTS['VE_AERO']
    
    async def create_lock(
        self,
        amount: int,
        lock_duration: int
    ) -> Dict[str, Any]:
        """Create a new veAERO lock"""
        
        # Lock duration in seconds
        unlock_time = int(time.time()) + lock_duration
        
        args = {
            'value': amount,
            'lockDuration': unlock_time
        }
        
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.ve_address,
            'method': 'create_lock',
            'args': args,
            'abi': VE_AERO_ABI
        })
        
        return result
    
    async def increase_amount(
        self,
        token_id: int,
        amount: int
    ) -> Dict[str, Any]:
        """Increase amount in existing lock"""
        
        args = {
            'tokenId': token_id,
            'value': amount
        }
        
        result = await self.cdp.wallet.invoke_contract({
            'contract_address': self.ve_address,
            'method': 'increase_amount',
            'args': args,
            'abi': VE_AERO_ABI
        })
        
        return result
    
    async def get_voting_power(self, token_id: int) -> int:
        """Get current voting power of veNFT"""
        
        power = await readContract({
            'network_id': 'base-mainnet',
            'contract_address': self.ve_address,
            'method': 'balanceOfNFT',
            'args': {'tokenId': token_id},
            'abi': VE_AERO_ABI
        })
        
        return power
```

## Transaction Management

### Gas Optimization

```python
class GasOptimizer:
    """Optimize gas for transactions"""
    
    async def get_optimal_gas_params(self) -> Dict[str, int]:
        """Get optimized EIP-1559 gas parameters"""
        
        # Get current base fee
        base_fee = await self._get_base_fee()
        
        # Get priority fee stats
        priority_fee = await self._get_priority_fee()
        
        # Calculate optimal parameters
        max_priority_fee = int(priority_fee * 1.1)  # 10% buffer
        max_fee = int(base_fee * 2 + max_priority_fee)  # 2x base fee ceiling
        
        return {
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': max_priority_fee
        }
    
    async def should_execute_now(self) -> bool:
        """Determine if gas prices are favorable"""
        
        current_base_fee = await self._get_base_fee()
        avg_base_fee = await self._get_average_base_fee(blocks=100)
        
        # Execute if current gas is below average
        return current_base_fee < avg_base_fee * 1.1
```

### Transaction Builder

```python
class TransactionBuilder:
    """Build optimized transactions"""
    
    def build_swap_transaction(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        max_slippage: float = 0.02
    ) -> Dict[str, Any]:
        """Build swap transaction with safety checks"""
        
        # Calculate minimum output with slippage
        expected_out = self.calculate_expected_output(token_in, token_out, amount_in)
        min_out = int(expected_out * (1 - max_slippage))
        
        # Build route
        route = [{
            'from': token_in,
            'to': token_out,
            'stable': self.is_stable_pair(token_in, token_out)
        }]
        
        # Add deadline (15 minutes)
        deadline = int(time.time()) + 900
        
        return {
            'method': 'swapExactTokensForTokens',
            'args': {
                'amountIn': amount_in,
                'amountOutMin': min_out,
                'routes': route,
                'to': self.wallet_address,
                'deadline': deadline
            },
            'safety': {
                'max_slippage': max_slippage,
                'deadline': deadline,
                'expected_out': expected_out,
                'min_out': min_out
            }
        }
```

## Event Monitoring

### Event Listener

```python
class EventMonitor:
    """Monitor blockchain events via CDP SDK"""
    
    def __init__(self, cdp_manager: CDPManager):
        self.cdp = cdp_manager
        self.subscriptions = {}
    
    async def monitor_pool_events(self, pool_address: str):
        """Monitor all events from a pool"""
        
        # Register pool contract
        await SmartContract.register(
            network_id='base-mainnet',
            contract_address=pool_address,
            abi=POOL_ABI,
            contract_name=f"Pool_{pool_address[:8]}"
        )
        
        # Monitor Swap events
        swap_events = await self.cdp.client.get_smart_contract_events({
            'contract_address': pool_address,
            'event_name': 'Swap',
            'from_block': 'latest'
        })
        
        # Process events
        for event in swap_events:
            await self.process_swap_event(event)
    
    async def process_swap_event(self, event: Dict):
        """Process swap event"""
        
        # Extract event data
        swap_data = {
            'sender': event['args']['sender'],
            'to': event['args']['to'],
            'amount0In': event['args']['amount0In'],
            'amount1In': event['args']['amount1In'],
            'amount0Out': event['args']['amount0Out'],
            'amount1Out': event['args']['amount1Out'],
            'block': event['blockNumber'],
            'tx_hash': event['transactionHash']
        }
        
        # Store in memory for pattern analysis
        await self.memory.add_swap_event(swap_data)
        
        # Check for arbitrage opportunity
        if self.is_arbitrage_opportunity(swap_data):
            await self.execute_arbitrage(swap_data)
```

## Error Handling

### CDP Error Handler

```python
class CDPErrorHandler:
    """Handle CDP SDK specific errors"""
    
    ERROR_MAPPINGS = {
        'insufficient funds': 'INSUFFICIENT_BALANCE',
        'nonce too low': 'NONCE_ERROR',
        'gas required exceeds': 'GAS_LIMIT_ERROR',
        'execution reverted': 'CONTRACT_REVERT',
        'network error': 'NETWORK_ERROR',
        'rate limit': 'RATE_LIMIT'
    }
    
    async def handle_cdp_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors from CDP SDK operations"""
        
        error_msg = str(error).lower()
        error_type = self.classify_error(error_msg)
        
        handlers = {
            'INSUFFICIENT_BALANCE': self.handle_insufficient_balance,
            'NONCE_ERROR': self.handle_nonce_error,
            'GAS_LIMIT_ERROR': self.handle_gas_limit,
            'CONTRACT_REVERT': self.handle_revert,
            'NETWORK_ERROR': self.handle_network_error,
            'RATE_LIMIT': self.handle_rate_limit
        }
        
        handler = handlers.get(error_type, self.handle_unknown_error)
        return await handler(error)
    
    def classify_error(self, error_msg: str) -> str:
        """Classify CDP error type"""
        
        for pattern, error_type in self.ERROR_MAPPINGS.items():
            if pattern in error_msg:
                return error_type
        
        return 'UNKNOWN'
    
    async def handle_insufficient_balance(self, error: Exception) -> Dict:
        """Handle insufficient balance errors"""
        
        # Check which token is insufficient
        token = self.extract_token_from_error(error)
        
        return {
            'retry': False,
            'action': 'ALERT_USER',
            'message': f'Insufficient {token} balance',
            'suggestion': 'Fund wallet or reduce position size'
        }
    
    async def handle_nonce_error(self, error: Exception) -> Dict:
        """Handle nonce errors"""
        
        # CDP SDK should handle this automatically
        # If we get here, there's a deeper issue
        
        return {
            'retry': True,
            'delay': 5,  # Wait 5 seconds
            'action': 'RESET_NONCE'
        }
```

## Best Practices

### 1. Always Use CDP SDK

```python
# ✅ CORRECT - Using CDP SDK
result = await self.wallet.invoke_contract({
    'contract_address': ROUTER_ADDRESS,
    'method': 'swap',
    'args': swap_params,
    'abi': ROUTER_ABI
})

# ❌ WRONG - Direct web3/ethers usage
# web3.eth.contract(ROUTER_ADDRESS).methods.swap(...).send()
```

### 2. Leverage Type Inference

```python
# CDP SDK automatically infers types with ABI
balance = await readContract({
    'network_id': 'base-mainnet',
    'contract_address': TOKEN_ADDRESS,
    'method': 'balanceOf',
    'args': {'account': wallet_address},
    'abi': ERC20_ABI
})
# balance is automatically typed as int
```

### 3. Handle Errors Gracefully

```python
async def safe_contract_call(self, **kwargs):
    """Wrapper for safe contract calls"""
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            result = await self.wallet.invoke_contract(**kwargs)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Handle specific errors
            if 'rate limit' in str(e).lower():
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

### 4. Batch Operations

```python
async def batch_operations(self, operations: List[Dict]):
    """Execute multiple operations efficiently"""
    
    # Group read operations
    reads = [op for op in operations if op['type'] == 'read']
    writes = [op for op in operations if op['type'] == 'write']
    
    # Execute reads in parallel
    if reads:
        read_results = await asyncio.gather(*[
            readContract(**op['params']) for op in reads
        ])
    
    # Execute writes sequentially (for nonce management)
    write_results = []
    for write_op in writes:
        result = await self.wallet.invoke_contract(**write_op['params'])
        write_results.append(result)
    
    return {
        'reads': read_results,
        'writes': write_results
    }
```

### 5. Monitor Gas Prices

```python
async def execute_when_optimal(self, transaction: Dict):
    """Execute transaction when gas is optimal"""
    
    while True:
        gas_optimizer = GasOptimizer()
        
        if await gas_optimizer.should_execute_now():
            # Add optimized gas params
            gas_params = await gas_optimizer.get_optimal_gas_params()
            transaction.update(gas_params)
            
            # Execute
            result = await self.wallet.invoke_contract(transaction)
            return result
        
        # Wait and check again
        await asyncio.sleep(30)
```

### 6. Use MPC Wallets in Production

```python
# Production configuration
if os.getenv('ENVIRONMENT') == 'production':
    wallet_config = {
        'use_mpc': True,  # Enable MPC
        'server_signer_url': os.getenv('SERVER_SIGNER_URL'),
        'backup_key_provider': 'aws_kms',  # Use AWS KMS for backup
        'require_2fa': True  # Additional security
    }
```