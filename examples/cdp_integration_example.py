"""
Complete CDP SDK Integration Example

This example demonstrates the complete CDP SDK integration layer usage
for the Aerodrome AI agent, showing:

1. CDP Manager initialization and wallet setup
2. Wallet operations and token management
3. Smart contract interactions
4. Aerodrome protocol operations (swaps, voting, veAERO)
5. Gas optimization strategies
6. Error handling patterns
7. Complete transaction workflows

Run this example to verify all components work correctly together.
"""

import asyncio
import logging
import os
from decimal import Decimal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CDP integration layer
try:
    from src.cdp import (
        CDPManager, WalletOperations, ContractOperations, 
        AerodromeRouter, AerodromeVoter, VotingEscrow,
        GasOptimizer, MEVProtection, SwapRoute, VoteAllocation,
        CDPError, handle_cdp_error
    )
    from src.contracts import get_token_address, get_contract_address, is_stable_pair, get_abi
except ImportError as e:
    logger.error(f"Failed to import CDP integration layer: {e}")
    logger.error("Make sure you've installed the CDP SDK: pip install cdp-sdk")
    raise


class AerodromeAIAgent:
    """
    Complete Aerodrome AI Agent with CDP SDK integration.
    
    Demonstrates all major operations using the CDP integration layer.
    """
    
    def __init__(self, network: str = 'base-mainnet'):
        """Initialize the AI agent with CDP integration."""
        self.network = network
        self.cdp_manager = None
        self.wallet_ops = None
        self.contract_ops = None
        self.aerodrome_router = None
        self.aerodrome_voter = None
        self.voting_escrow = None
        self.gas_optimizer = None
        self.mev_protection = None
        
    async def initialize(self):
        """Initialize all CDP components."""
        try:
            logger.info("Initializing Aerodrome AI Agent with CDP SDK...")
            
            # 1. Initialize CDP Manager
            self.cdp_manager = CDPManager(network=self.network)
            logger.info(f"CDP Manager initialized for network: {self.network}")
            logger.info(f"Wallet address: {self.cdp_manager.wallet_address}")
            logger.info(f"Using MPC: {self.cdp_manager.uses_mpc}")
            
            # 2. Initialize wallet operations
            self.wallet_ops = WalletOperations(self.cdp_manager)
            
            # 3. Initialize contract operations
            self.contract_ops = ContractOperations(self.cdp_manager)
            
            # 4. Initialize Aerodrome protocol operations
            self.aerodrome_router = AerodromeRouter(self.cdp_manager, self.contract_ops)
            self.aerodrome_voter = AerodromeVoter(self.cdp_manager, self.contract_ops)
            self.voting_escrow = VotingEscrow(self.cdp_manager, self.contract_ops)
            
            # 5. Initialize gas optimization
            self.gas_optimizer = GasOptimizer(self.cdp_manager)
            self.mev_protection = MEVProtection(self.gas_optimizer)
            
            # 6. Perform health check
            health_status = await self.cdp_manager.health_check()
            logger.info(f"CDP Health Status: {health_status}")
            
            if not health_status['overall']:
                logger.warning("CDP health check failed - some operations may not work")
                for error in health_status['errors']:
                    logger.warning(f"  - {error}")
            
            logger.info("✅ Aerodrome AI Agent fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            raise
    
    async def demonstrate_wallet_operations(self):
        """Demonstrate wallet operations."""
        logger.info("\n🔶 Demonstrating Wallet Operations...")
        
        try:
            # Get all token balances
            balances = await self.wallet_ops.get_all_balances()
            logger.info("Current wallet balances:")
            for token, balance in balances.items():
                if balance > 0:
                    logger.info(f"  {token}: {balance:.6f}")
            
            # Get detailed token information
            try:
                usdc_address = get_token_address('USDC', self.network)
                usdc_info = await self.wallet_ops.get_token_info(usdc_address)
                logger.info(f"USDC token info: {usdc_info}")
            except Exception as e:
                logger.warning(f"Could not get USDC info: {e}")
            
            # Get wallet summary
            wallet_summary = await self.wallet_ops.get_wallet_summary()
            logger.info(f"Wallet summary: {wallet_summary}")
            
        except Exception as e:
            logger.error(f"Wallet operations failed: {str(e)}")
    
    async def demonstrate_gas_optimization(self):
        """Demonstrate gas optimization features."""
        logger.info("\n⛽ Demonstrating Gas Optimization...")
        
        try:
            # Get optimal gas parameters
            gas_params = await self.gas_optimizer.get_optimal_gas_params(priority='standard')
            logger.info(f"Optimal gas params: {gas_params.to_gwei()}")
            
            # Check execution timing
            timing_analysis = await self.gas_optimizer.should_execute_now(
                target_cost_gwei=0.1,  # Target 0.1 gwei
                max_wait_minutes=10
            )
            logger.info(f"Execution timing: {timing_analysis}")
            
            # Estimate transaction cost
            cost_estimate = await self.gas_optimizer.estimate_transaction_cost(
                gas_limit=200000,
                priority='standard'
            )
            logger.info(f"Transaction cost estimate: ${cost_estimate['likely_cost']['usd']:.4f}")
            
            # MEV protection
            protected_params = self.mev_protection.add_mev_protection_params(
                gas_params, 
                protection_level='medium'
            )
            logger.info(f"MEV protected gas: {protected_params.to_gwei()}")
            
        except Exception as e:
            logger.error(f"Gas optimization failed: {str(e)}")
    
    async def demonstrate_token_swap(self):
        """Demonstrate token swap using Aerodrome Router."""
        logger.info("\n🔄 Demonstrating Token Swap...")
        
        try:
            # Example: Swap 0.01 WETH for USDC
            weth_address = get_token_address('WETH', self.network)
            usdc_address = get_token_address('USDC', self.network)
            
            # Convert 0.01 ETH to wei
            swap_amount = int(0.01 * 1e18)
            
            # Check if we have enough WETH balance (this is just for demo)
            weth_balance = await self.wallet_ops.get_balance_wei(weth_address)
            logger.info(f"Current WETH balance: {weth_balance / 1e18:.6f} WETH")
            
            if weth_balance < swap_amount:
                logger.warning("Insufficient WETH balance for demo swap")
                return
            
            # Get expected output amount
            routes = [SwapRoute(
                from_token=weth_address,
                to_token=usdc_address,
                stable=is_stable_pair(weth_address, usdc_address, self.network)
            )]
            
            expected_amounts = await self.aerodrome_router.get_amounts_out(swap_amount, routes)
            logger.info(f"Expected output: {expected_amounts[-1] / 1e6:.6f} USDC")
            
            # Execute swap (commented out to avoid actual transaction)
            # swap_result = await self.aerodrome_router.swap_exact_tokens_for_tokens(
            #     token_in=weth_address,
            #     token_out=usdc_address,
            #     amount_in=swap_amount,
            #     max_slippage=0.02
            # )
            # logger.info(f"Swap result: {swap_result}")
            
            logger.info("✅ Swap simulation completed (actual swap commented out)")
            
        except Exception as e:
            error_context = await handle_cdp_error(e, {'operation': 'swap'})
            logger.error(f"Swap failed: {error_context}")
    
    async def demonstrate_liquidity_operations(self):
        """Demonstrate liquidity provision."""
        logger.info("\n💧 Demonstrating Liquidity Operations...")
        
        try:
            # Example: Add liquidity to USDC-DAI stable pool
            usdc_address = get_token_address('USDC', self.network)
            dai_address = get_token_address('DAI', self.network)
            
            # Example amounts (in token units, not wei)
            usdc_amount = int(100 * 1e6)  # 100 USDC (6 decimals)
            dai_amount = int(100 * 1e18)   # 100 DAI (18 decimals)
            
            # Check balances
            usdc_balance = await self.wallet_ops.get_balance_wei(usdc_address)
            dai_balance = await self.wallet_ops.get_balance_wei(dai_address)
            
            logger.info(f"USDC balance: {usdc_balance / 1e6:.2f}")
            logger.info(f"DAI balance: {dai_balance / 1e18:.2f}")
            
            if usdc_balance < usdc_amount or dai_balance < dai_amount:
                logger.warning("Insufficient balance for liquidity provision demo")
                return
            
            # Simulate add liquidity (actual execution commented out)
            # liquidity_result = await self.aerodrome_router.add_liquidity(
            #     token_a=usdc_address,
            #     token_b=dai_address,
            #     amount_a=usdc_amount,
            #     amount_b=dai_amount,
            #     stable=True  # USDC-DAI should be stable pool
            # )
            # logger.info(f"Liquidity result: {liquidity_result}")
            
            logger.info("✅ Liquidity operation simulation completed")
            
        except Exception as e:
            logger.error(f"Liquidity operations failed: {str(e)}")
    
    async def demonstrate_voting_operations(self):
        """Demonstrate veAERO and voting operations."""
        logger.info("\n🗳️  Demonstrating Voting Operations...")
        
        try:
            # Check if wallet has any veAERO tokens
            owned_tokens = await self.voting_escrow.get_owned_tokens()
            logger.info(f"Owned veAERO tokens: {owned_tokens}")
            
            if not owned_tokens:
                logger.info("No veAERO tokens found - would need AERO to create lock")
                
                # Demonstrate lock creation parameters (without execution)
                aero_address = get_token_address('AERO', self.network)
                aero_balance = await self.wallet_ops.get_balance_wei(aero_address)
                
                if aero_balance > 0:
                    logger.info(f"AERO balance: {aero_balance / 1e18:.6f} AERO")
                    logger.info("Could create veAERO lock with available AERO")
                    
                    # Example lock creation (commented out)
                    # lock_result = await self.voting_escrow.create_lock(
                    #     amount=int(100 * 1e18),  # 100 AERO
                    #     lock_duration_weeks=52   # 1 year
                    # )
                else:
                    logger.info("No AERO balance - would need AERO tokens to participate in voting")
                
                return
            
            # If we have veAERO tokens, demonstrate voting operations
            for token_id in owned_tokens:
                # Get voting power
                voting_power = await self.voting_escrow.get_voting_power(token_id)
                logger.info(f"Token {token_id} voting power: {voting_power}")
                
                # Get lock info
                lock_info = await self.voting_escrow.get_lock_info(token_id)
                logger.info(f"Token {token_id} lock info: {lock_info}")
                
                # Example vote allocation (commented out to avoid actual voting)
                # vote_allocations = [
                #     VoteAllocation(
                #         pool_address="0x...",  # Pool address
                #         weight=5000  # 50% of voting power
                #     ),
                #     VoteAllocation(
                #         pool_address="0x...",  # Another pool
                #         weight=5000  # 50% of voting power
                #     )
                # ]
                # 
                # vote_result = await self.aerodrome_voter.vote(token_id, vote_allocations)
                # logger.info(f"Vote result: {vote_result}")
                
                break  # Just demonstrate with first token
            
            logger.info("✅ Voting operations demonstration completed")
            
        except Exception as e:
            logger.error(f"Voting operations failed: {str(e)}")
    
    async def demonstrate_contract_operations(self):
        """Demonstrate low-level contract operations."""
        logger.info("\n📋 Demonstrating Contract Operations...")
        
        try:
            # Example: Read pool reserves
            try:
                usdc_address = get_token_address('USDC', self.network)
                weth_address = get_token_address('WETH', self.network)
                
                # Get factory contract to find pool
                factory_address = get_contract_address('FACTORY', self.network)
                
                pool_address = await self.contract_ops.read_contract(
                    contract_address=factory_address,
                    method='getPair',
                    args={
                        'tokenA': usdc_address,
                        'tokenB': weth_address,
                        'stable': False  # Volatile pool
                    },
                    abi=get_abi('FACTORY')
                )
                
                if pool_address and pool_address != '0x0000000000000000000000000000000000000000':
                    # Read pool reserves
                    reserves = await self.contract_ops.read_contract(
                        contract_address=pool_address,
                        method='getReserves',
                        abi=get_abi('POOL')
                    )
                    logger.info(f"USDC-WETH pool reserves: {reserves}")
                else:
                    logger.info("USDC-WETH pool not found")
                
            except Exception as e:
                logger.warning(f"Pool data reading failed: {e}")
            
            # Demonstrate batch read operations
            batch_calls = [
                {
                    'contract_address': get_token_address('USDC', self.network),
                    'method': 'totalSupply',
                    'abi': get_abi('ERC20')
                },
                {
                    'contract_address': get_token_address('AERO', self.network),
                    'method': 'totalSupply',
                    'abi': get_abi('ERC20')
                }
            ]
            
            batch_results = await self.contract_ops.batch_read(batch_calls)
            logger.info(f"Batch read results: {batch_results}")
            
            # Get cache statistics
            cache_stats = self.contract_ops.get_cache_stats()
            logger.info(f"Contract call cache stats: {cache_stats}")
            
            logger.info("✅ Contract operations demonstration completed")
            
        except Exception as e:
            logger.error(f"Contract operations failed: {str(e)}")
    
    async def run_complete_demo(self):
        """Run complete demonstration of all CDP integration features."""
        try:
            await self.initialize()
            
            # Run all demonstrations
            await self.demonstrate_wallet_operations()
            await self.demonstrate_gas_optimization()
            await self.demonstrate_token_swap()
            await self.demonstrate_liquidity_operations()
            await self.demonstrate_voting_operations()
            await self.demonstrate_contract_operations()
            
            # Final statistics
            logger.info("\n📊 Final Statistics:")
            
            # Gas analytics
            gas_analytics = self.gas_optimizer.get_gas_analytics()
            logger.info(f"Gas analytics: {gas_analytics}")
            
            # Network info
            network_info = self.cdp_manager.get_network_info()
            logger.info(f"Network info: {network_info}")
            
            logger.info("\n🎉 Complete CDP integration demonstration finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
            raise


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Aerodrome AI Agent CDP Integration Demo")
    
    # Check environment variables
    required_env_vars = ['CDP_API_KEY_ID', 'CDP_API_KEY_SECRET']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Some operations may fail without proper CDP credentials")
    
    # Initialize and run agent
    agent = AerodromeAIAgent(network='base-mainnet')
    
    try:
        await agent.run_complete_demo()
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the complete demo
    exit_code = asyncio.run(main())
    exit(exit_code)