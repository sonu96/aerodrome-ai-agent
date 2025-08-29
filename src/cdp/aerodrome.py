"""
Aerodrome Protocol Operations - Specialized operations for Aerodrome DeFi protocol

This module provides high-level interfaces for all Aerodrome protocol operations:
- Router operations (swaps, liquidity management)
- Voter operations (voting, bribe claiming)
- veAERO operations (locking, voting power)
- Gauge operations (staking, rewards)
- Pool analytics and monitoring

All operations use CDP SDK exclusively through the ContractOperations layer.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from decimal import Decimal
from dataclasses import dataclass

from .contracts import ContractOperations
from .errors import CDPError, ContractError, InsufficientBalanceError, TransactionError
from ..contracts.abis import get_abi
from ..contracts.addresses import (
    AERODROME_CONTRACTS, get_contract_address, get_token_address,
    is_stable_pair, get_pool_address, get_gauge_address, get_bribe_address
)


@dataclass
class SwapRoute:
    """Swap route data structure."""
    from_token: str
    to_token: str
    stable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for contract calls."""
        return {
            'from': self.from_token,
            'to': self.to_token,
            'stable': self.stable
        }


@dataclass  
class LiquidityPosition:
    """Liquidity position data structure."""
    pool_address: str
    token_a: str
    token_b: str
    stable: bool
    liquidity_tokens: int
    reserve_a: int
    reserve_b: int
    share_percentage: float


@dataclass
class VoteAllocation:
    """Vote allocation data structure."""
    pool_address: str
    weight: int  # Weight in basis points (10000 = 100%)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for contract calls."""
        return {
            'pool': self.pool_address,
            'weight': self.weight
        }


class AerodromeRouter:
    """
    Aerodrome Router operations via CDP SDK.
    
    Handles all DEX operations including swaps, liquidity management,
    and routing optimization.
    """
    
    def __init__(self, cdp_manager, contract_ops: ContractOperations):
        """
        Initialize Aerodrome Router operations.
        
        Args:
            cdp_manager: CDPManager instance
            contract_ops: ContractOperations instance
        """
        self.cdp = cdp_manager
        self.contract_ops = contract_ops
        self.logger = logging.getLogger(__name__)
        self.router_address = get_contract_address('ROUTER', cdp_manager.network)
        self.factory_address = get_contract_address('FACTORY', cdp_manager.network)
        self.router_abi = get_abi('ROUTER')
        
    async def swap_exact_tokens_for_tokens(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out_min: Optional[int] = None,
        max_slippage: float = 0.02,
        deadline: Optional[int] = None,
        routes: Optional[List[SwapRoute]] = None
    ) -> Dict[str, Any]:
        """
        Execute token swap through Aerodrome Router.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount in wei
            amount_out_min: Minimum output amount (calculated if not provided)
            max_slippage: Maximum allowed slippage (default 2%)
            deadline: Transaction deadline (15 minutes if not provided)
            routes: Custom routing path (auto-calculated if not provided)
            
        Returns:
            Transaction result dictionary
            
        Raises:
            InsufficientBalanceError: If insufficient balance
            TransactionError: If swap fails
        """
        if deadline is None:
            deadline = int(time.time()) + 900  # 15 minutes
        
        try:
            # Build or validate routes
            if routes is None:
                routes = await self._build_optimal_route(token_in, token_out)
            
            # Calculate minimum output amount if not provided
            if amount_out_min is None:
                expected_out = await self.get_amounts_out(amount_in, routes)
                amount_out_min = int(expected_out[-1] * (1 - max_slippage))
            
            # Validate balance
            wallet_balance = await self.contract_ops.read_contract(
                contract_address=token_in,
                method='balanceOf',
                args={'account': self.cdp.wallet_address},
                abi=get_abi('ERC20')
            )
            
            if wallet_balance < amount_in:
                raise InsufficientBalanceError(
                    f"Insufficient balance. Required: {amount_in}, Available: {wallet_balance}",
                    required=amount_in,
                    available=wallet_balance,
                    token=token_in
                )
            
            # Check and approve token if needed
            allowance = await self.contract_ops.read_contract(
                contract_address=token_in,
                method='allowance',
                args={
                    'owner': self.cdp.wallet_address,
                    'spender': self.router_address
                },
                abi=get_abi('ERC20')
            )
            
            if allowance < amount_in:
                self.logger.info(f"Approving {token_in} for router")
                approval_result = await self.contract_ops.invoke_contract(
                    contract_address=token_in,
                    method='approve',
                    args={
                        'spender': self.router_address,
                        'amount': amount_in
                    },
                    abi=get_abi('ERC20')
                )
                
                if not approval_result['success']:
                    raise TransactionError("Token approval failed")
            
            # Execute swap
            swap_args = {
                'amountIn': amount_in,
                'amountOutMin': amount_out_min,
                'routes': [route.to_dict() for route in routes],
                'to': self.cdp.wallet_address,
                'deadline': deadline
            }
            
            self.logger.info(f"Executing swap: {amount_in} tokens via {len(routes)} route(s)")
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.router_address,
                method='swapExactTokensForTokens',
                args=swap_args,
                abi=self.router_abi
            )
            
            # Enhance result with swap details
            result.update({
                'swap_type': 'exact_tokens_for_tokens',
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': amount_in,
                'amount_out_min': amount_out_min,
                'routes': [route.to_dict() for route in routes],
                'max_slippage': max_slippage
            })
            
            return result
            
        except (InsufficientBalanceError, TransactionError):
            raise
        except Exception as e:
            self.logger.error(f"Swap failed: {str(e)}")
            raise TransactionError(f"Swap execution failed: {str(e)}")
    
    async def swap_exact_eth_for_tokens(
        self,
        token_out: str,
        amount_in: int,
        amount_out_min: Optional[int] = None,
        max_slippage: float = 0.02,
        deadline: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Swap ETH for tokens.
        
        Args:
            token_out: Output token address
            amount_in: ETH amount in wei
            amount_out_min: Minimum output amount
            max_slippage: Maximum slippage
            deadline: Transaction deadline
            
        Returns:
            Transaction result
        """
        if deadline is None:
            deadline = int(time.time()) + 900
        
        try:
            # Build route from WETH to token_out
            weth_address = get_token_address('WETH', self.cdp.network)
            routes = await self._build_optimal_route(weth_address, token_out)
            
            # Calculate minimum output if not provided
            if amount_out_min is None:
                expected_out = await self.get_amounts_out(amount_in, routes)
                amount_out_min = int(expected_out[-1] * (1 - max_slippage))
            
            # Check ETH balance
            eth_balance = await self.cdp.wallet.get_balance()
            if eth_balance < amount_in:
                raise InsufficientBalanceError(
                    f"Insufficient ETH balance. Required: {amount_in}, Available: {eth_balance}",
                    required=amount_in,
                    available=eth_balance,
                    token='ETH'
                )
            
            swap_args = {
                'amountOutMin': amount_out_min,
                'routes': [route.to_dict() for route in routes],
                'to': self.cdp.wallet_address,
                'deadline': deadline
            }
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.router_address,
                method='swapExactETHForTokens',
                args=swap_args,
                value=amount_in,  # Send ETH value
                abi=self.router_abi
            )
            
            result.update({
                'swap_type': 'exact_eth_for_tokens',
                'token_out': token_out,
                'amount_in': amount_in,
                'amount_out_min': amount_out_min
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"ETH swap failed: {str(e)}")
            raise TransactionError(f"ETH swap failed: {str(e)}")
    
    async def add_liquidity(
        self,
        token_a: str,
        token_b: str,
        amount_a: int,
        amount_b: int,
        stable: Optional[bool] = None,
        min_liquidity: Optional[int] = None,
        slippage_tolerance: float = 0.02
    ) -> Dict[str, Any]:
        """
        Add liquidity to Aerodrome pool.
        
        Args:
            token_a: First token address
            token_b: Second token address  
            amount_a: Amount of token A
            amount_b: Amount of token B
            stable: Whether to use stable pool (auto-detect if None)
            min_liquidity: Minimum liquidity tokens to receive
            slippage_tolerance: Slippage tolerance
            
        Returns:
            Transaction result with liquidity details
        """
        try:
            # Auto-detect stable pairing if not specified
            if stable is None:
                stable = is_stable_pair(token_a, token_b, self.cdp.network)
            
            # Calculate minimum amounts with slippage
            amount_a_min = int(amount_a * (1 - slippage_tolerance))
            amount_b_min = int(amount_b * (1 - slippage_tolerance))
            
            # Validate balances
            balance_a = await self.contract_ops.read_contract(
                contract_address=token_a,
                method='balanceOf',
                args={'account': self.cdp.wallet_address},
                abi=get_abi('ERC20')
            )
            
            balance_b = await self.contract_ops.read_contract(
                contract_address=token_b,
                method='balanceOf',
                args={'account': self.cdp.wallet_address},
                abi=get_abi('ERC20')
            )
            
            if balance_a < amount_a:
                raise InsufficientBalanceError(f"Insufficient token A balance", 
                                             required=amount_a, available=balance_a, token=token_a)
            
            if balance_b < amount_b:
                raise InsufficientBalanceError(f"Insufficient token B balance",
                                             required=amount_b, available=balance_b, token=token_b)
            
            # Approve tokens
            await self._ensure_approval(token_a, amount_a)
            await self._ensure_approval(token_b, amount_b)
            
            # Add liquidity
            liquidity_args = {
                'tokenA': token_a,
                'tokenB': token_b,
                'stable': stable,
                'amountADesired': amount_a,
                'amountBDesired': amount_b,
                'amountAMin': amount_a_min,
                'amountBMin': amount_b_min,
                'to': self.cdp.wallet_address,
                'deadline': int(time.time()) + 900
            }
            
            self.logger.info(f"Adding liquidity to {token_a[:8]}-{token_b[:8]} {'stable' if stable else 'volatile'} pool")
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.router_address,
                method='addLiquidity',
                args=liquidity_args,
                abi=self.router_abi
            )
            
            # Enhance result with liquidity details
            result.update({
                'operation': 'add_liquidity',
                'token_a': token_a,
                'token_b': token_b,
                'stable': stable,
                'amount_a_desired': amount_a,
                'amount_b_desired': amount_b,
                'slippage_tolerance': slippage_tolerance
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Add liquidity failed: {str(e)}")
            raise TransactionError(f"Add liquidity failed: {str(e)}")
    
    async def remove_liquidity(
        self,
        token_a: str,
        token_b: str,
        liquidity: int,
        stable: Optional[bool] = None,
        slippage_tolerance: float = 0.02
    ) -> Dict[str, Any]:
        """
        Remove liquidity from Aerodrome pool.
        
        Args:
            token_a: First token address
            token_b: Second token address
            liquidity: Liquidity tokens to remove
            stable: Whether pool is stable (auto-detect if None)
            slippage_tolerance: Slippage tolerance
            
        Returns:
            Transaction result
        """
        try:
            if stable is None:
                stable = is_stable_pair(token_a, token_b, self.cdp.network)
            
            # Get pool address for balance check
            pool_address = await self._get_pool_address(token_a, token_b, stable)
            
            # Check LP token balance
            lp_balance = await self.contract_ops.read_contract(
                contract_address=pool_address,
                method='balanceOf',
                args={'account': self.cdp.wallet_address},
                abi=get_abi('ERC20')
            )
            
            if lp_balance < liquidity:
                raise InsufficientBalanceError(f"Insufficient LP tokens",
                                             required=liquidity, available=lp_balance, 
                                             token=pool_address)
            
            # Approve LP tokens for router
            await self._ensure_approval(pool_address, liquidity)
            
            remove_args = {
                'tokenA': token_a,
                'tokenB': token_b,
                'stable': stable,
                'liquidity': liquidity,
                'amountAMin': 0,  # Accept any amount (could be improved)
                'amountBMin': 0,  # Accept any amount
                'to': self.cdp.wallet_address,
                'deadline': int(time.time()) + 900
            }
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.router_address,
                method='removeLiquidity',
                args=remove_args,
                abi=self.router_abi
            )
            
            result.update({
                'operation': 'remove_liquidity',
                'token_a': token_a,
                'token_b': token_b,
                'stable': stable,
                'liquidity_removed': liquidity
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Remove liquidity failed: {str(e)}")
            raise TransactionError(f"Remove liquidity failed: {str(e)}")
    
    async def get_amounts_out(
        self,
        amount_in: int,
        routes: List[SwapRoute]
    ) -> List[int]:
        """
        Get expected output amounts for swap routes.
        
        Args:
            amount_in: Input amount
            routes: Swap routes
            
        Returns:
            List of amounts for each step in route
        """
        try:
            route_dicts = [route.to_dict() for route in routes]
            
            amounts = await self.contract_ops.read_contract(
                contract_address=self.router_address,
                method='getAmountsOut',
                args={
                    'amountIn': amount_in,
                    'routes': route_dicts
                },
                abi=self.router_abi
            )
            
            return amounts
            
        except Exception as e:
            self.logger.error(f"Get amounts out failed: {str(e)}")
            return [amount_in]  # Fallback
    
    async def _build_optimal_route(
        self,
        token_in: str,
        token_out: str
    ) -> List[SwapRoute]:
        """
        Build optimal swap route between two tokens.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            
        Returns:
            List of swap routes
        """
        # For now, create direct route
        stable = is_stable_pair(token_in, token_out, self.cdp.network)
        
        return [SwapRoute(
            from_token=token_in,
            to_token=token_out,
            stable=stable
        )]
    
    async def _get_pool_address(
        self,
        token_a: str,
        token_b: str,
        stable: bool
    ) -> str:
        """
        Get pool address for token pair.
        
        Args:
            token_a: First token address
            token_b: Second token address
            stable: Whether pool is stable
            
        Returns:
            Pool address
        """
        try:
            return await self.contract_ops.read_contract(
                contract_address=self.factory_address,
                method='getPair',
                args={
                    'tokenA': token_a,
                    'tokenB': token_b,
                    'stable': stable
                },
                abi=get_abi('FACTORY')
            )
        except Exception as e:
            self.logger.error(f"Failed to get pool address: {str(e)}")
            raise ContractError(f"Pool not found for {token_a}-{token_b}")
    
    async def _ensure_approval(self, token_address: str, amount: int) -> None:
        """
        Ensure token approval for router.
        
        Args:
            token_address: Token contract address
            amount: Amount to approve
        """
        allowance = await self.contract_ops.read_contract(
            contract_address=token_address,
            method='allowance',
            args={
                'owner': self.cdp.wallet_address,
                'spender': self.router_address
            },
            abi=get_abi('ERC20')
        )
        
        if allowance < amount:
            self.logger.info(f"Approving {token_address[:8]} for router")
            approval_result = await self.contract_ops.invoke_contract(
                contract_address=token_address,
                method='approve',
                args={
                    'spender': self.router_address,
                    'amount': amount * 2  # Approve 2x to avoid frequent re-approvals
                },
                abi=get_abi('ERC20')
            )
            
            if not approval_result['success']:
                raise TransactionError("Token approval failed")


class AerodromeVoter:
    """
    Aerodrome Voter contract operations.
    
    Handles voting on gauge allocations, claiming bribes and rewards.
    """
    
    def __init__(self, cdp_manager, contract_ops: ContractOperations):
        """
        Initialize Voter operations.
        
        Args:
            cdp_manager: CDPManager instance
            contract_ops: ContractOperations instance
        """
        self.cdp = cdp_manager
        self.contract_ops = contract_ops
        self.logger = logging.getLogger(__name__)
        self.voter_address = get_contract_address('VOTER', cdp_manager.network)
        self.ve_address = get_contract_address('VE_AERO', cdp_manager.network)
        self.voter_abi = get_abi('VOTER')
    
    async def vote(
        self,
        token_id: int,
        pool_votes: List[VoteAllocation]
    ) -> Dict[str, Any]:
        """
        Vote for gauge allocations with veAERO NFT.
        
        Args:
            token_id: veAERO NFT token ID
            pool_votes: List of pool vote allocations
            
        Returns:
            Transaction result
        """
        try:
            # Validate token ownership
            await self._validate_token_ownership(token_id)
            
            # Prepare vote data
            pools = [vote.pool_address for vote in pool_votes]
            weights = [vote.weight for vote in pool_votes]
            
            # Ensure weights sum to 10000 (100%)
            total_weight = sum(weights)
            if total_weight != 10000:
                # Normalize weights proportionally
                weights = [int(w * 10000 / total_weight) for w in weights]
                self.logger.warning(f"Vote weights normalized. Total was {total_weight}, now 10000")
            
            vote_args = {
                'tokenId': token_id,
                'poolVote': pools,
                'weights': weights
            }
            
            self.logger.info(f"Voting with token ID {token_id} on {len(pools)} pools")
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.voter_address,
                method='vote',
                args=vote_args,
                abi=self.voter_abi
            )
            
            result.update({
                'operation': 'vote',
                'token_id': token_id,
                'pools_voted': len(pools),
                'total_weight': sum(weights)
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Voting failed: {str(e)}")
            raise TransactionError(f"Voting failed: {str(e)}")
    
    async def reset_vote(self, token_id: int) -> Dict[str, Any]:
        """
        Reset all votes for a veAERO NFT.
        
        Args:
            token_id: veAERO NFT token ID
            
        Returns:
            Transaction result
        """
        try:
            await self._validate_token_ownership(token_id)
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.voter_address,
                method='reset',
                args={'tokenId': token_id},
                abi=self.voter_abi
            )
            
            result.update({
                'operation': 'reset_vote',
                'token_id': token_id
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Vote reset failed: {str(e)}")
            raise TransactionError(f"Vote reset failed: {str(e)}")
    
    async def claim_bribes(
        self,
        bribes: List[str],
        tokens: List[List[str]],
        token_id: int
    ) -> Dict[str, Any]:
        """
        Claim bribes from voting.
        
        Args:
            bribes: List of bribe contract addresses
            tokens: List of token lists for each bribe
            token_id: veAERO NFT token ID
            
        Returns:
            Transaction result
        """
        try:
            await self._validate_token_ownership(token_id)
            
            bribe_args = {
                'bribes': bribes,
                'tokens': tokens,
                'tokenId': token_id
            }
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.voter_address,
                method='claimBribes',
                args=bribe_args,
                abi=self.voter_abi
            )
            
            result.update({
                'operation': 'claim_bribes',
                'token_id': token_id,
                'bribes_claimed': len(bribes)
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bribe claiming failed: {str(e)}")
            raise TransactionError(f"Bribe claiming failed: {str(e)}")
    
    async def claim_rewards(
        self,
        gauges: List[str],
        token_id: int
    ) -> Dict[str, Any]:
        """
        Claim rewards from gauges.
        
        Args:
            gauges: List of gauge addresses
            token_id: veAERO NFT token ID
            
        Returns:
            Transaction result
        """
        try:
            await self._validate_token_ownership(token_id)
            
            reward_args = {
                'gauges': gauges,
                'tokenId': token_id
            }
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.voter_address,
                method='claimRewards',
                args=reward_args,
                abi=self.voter_abi
            )
            
            result.update({
                'operation': 'claim_rewards',
                'token_id': token_id,
                'gauges_claimed': len(gauges)
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reward claiming failed: {str(e)}")
            raise TransactionError(f"Reward claiming failed: {str(e)}")
    
    async def get_votes(self, token_id: int, pool_address: str) -> int:
        """
        Get vote weight for specific pool.
        
        Args:
            token_id: veAERO NFT token ID
            pool_address: Pool address
            
        Returns:
            Vote weight
        """
        return await self.contract_ops.read_contract(
            contract_address=self.voter_address,
            method='votes',
            args={'tokenId': token_id, 'pool': pool_address},
            abi=self.voter_abi
        )
    
    async def get_used_weights(self, token_id: int) -> int:
        """
        Get total used voting weight for token.
        
        Args:
            token_id: veAERO NFT token ID
            
        Returns:
            Total used weight
        """
        return await self.contract_ops.read_contract(
            contract_address=self.voter_address,
            method='usedWeights',
            args={'tokenId': token_id},
            abi=self.voter_abi
        )
    
    async def _validate_token_ownership(self, token_id: int) -> None:
        """
        Validate that wallet owns the veAERO token.
        
        Args:
            token_id: Token ID to validate
            
        Raises:
            ContractError: If token is not owned by wallet
        """
        try:
            owner = await self.contract_ops.read_contract(
                contract_address=self.ve_address,
                method='ownerOf',
                args={'tokenId': token_id},
                abi=get_abi('VE_AERO')
            )
            
            if owner.lower() != self.cdp.wallet_address.lower():
                raise ContractError(f"Token {token_id} not owned by wallet")
                
        except Exception as e:
            raise ContractError(f"Token ownership validation failed: {str(e)}")


class VotingEscrow:
    """
    veAERO (Voting Escrow) operations.
    
    Handles AERO token locking, voting power management, and NFT operations.
    """
    
    def __init__(self, cdp_manager, contract_ops: ContractOperations):
        """
        Initialize VotingEscrow operations.
        
        Args:
            cdp_manager: CDPManager instance
            contract_ops: ContractOperations instance
        """
        self.cdp = cdp_manager
        self.contract_ops = contract_ops
        self.logger = logging.getLogger(__name__)
        self.ve_address = get_contract_address('VE_AERO', cdp_manager.network)
        self.aero_address = get_contract_address('AERO', cdp_manager.network)
        self.ve_abi = get_abi('VE_AERO')
    
    async def create_lock(
        self,
        amount: int,
        lock_duration_weeks: int
    ) -> Dict[str, Any]:
        """
        Create a new veAERO lock.
        
        Args:
            amount: AERO amount to lock (in wei)
            lock_duration_weeks: Lock duration in weeks
            
        Returns:
            Transaction result with token ID
        """
        try:
            # Validate lock duration (typically 1-208 weeks for Aerodrome)
            if lock_duration_weeks < 1 or lock_duration_weeks > 208:
                raise ValueError(f"Invalid lock duration: {lock_duration_weeks} weeks")
            
            # Check AERO balance
            aero_balance = await self.contract_ops.read_contract(
                contract_address=self.aero_address,
                method='balanceOf',
                args={'account': self.cdp.wallet_address},
                abi=get_abi('ERC20')
            )
            
            if aero_balance < amount:
                raise InsufficientBalanceError(f"Insufficient AERO balance",
                                             required=amount, available=aero_balance, 
                                             token=self.aero_address)
            
            # Approve AERO for veAERO contract
            await self._ensure_aero_approval(amount)
            
            # Calculate unlock time
            lock_duration_seconds = lock_duration_weeks * 7 * 24 * 60 * 60
            unlock_time = int(time.time()) + lock_duration_seconds
            
            lock_args = {
                'value': amount,
                'lockDuration': unlock_time
            }
            
            self.logger.info(f"Creating veAERO lock: {amount} AERO for {lock_duration_weeks} weeks")
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.ve_address,
                method='create_lock',
                args=lock_args,
                abi=self.ve_abi
            )
            
            result.update({
                'operation': 'create_lock',
                'amount_locked': amount,
                'lock_duration_weeks': lock_duration_weeks,
                'unlock_time': unlock_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Create lock failed: {str(e)}")
            raise TransactionError(f"Create lock failed: {str(e)}")
    
    async def increase_amount(
        self,
        token_id: int,
        amount: int
    ) -> Dict[str, Any]:
        """
        Increase amount in existing veAERO lock.
        
        Args:
            token_id: veAERO NFT token ID
            amount: Additional AERO amount
            
        Returns:
            Transaction result
        """
        try:
            # Validate ownership and lock status
            await self._validate_lock(token_id)
            
            # Check AERO balance
            aero_balance = await self.contract_ops.read_contract(
                contract_address=self.aero_address,
                method='balanceOf',
                args={'account': self.cdp.wallet_address},
                abi=get_abi('ERC20')
            )
            
            if aero_balance < amount:
                raise InsufficientBalanceError(f"Insufficient AERO balance",
                                             required=amount, available=aero_balance)
            
            # Approve additional AERO
            await self._ensure_aero_approval(amount)
            
            increase_args = {
                'tokenId': token_id,
                'value': amount
            }
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.ve_address,
                method='increase_amount',
                args=increase_args,
                abi=self.ve_abi
            )
            
            result.update({
                'operation': 'increase_amount',
                'token_id': token_id,
                'amount_added': amount
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Increase amount failed: {str(e)}")
            raise TransactionError(f"Increase amount failed: {str(e)}")
    
    async def increase_unlock_time(
        self,
        token_id: int,
        additional_weeks: int
    ) -> Dict[str, Any]:
        """
        Extend lock duration for existing veAERO.
        
        Args:
            token_id: veAERO NFT token ID
            additional_weeks: Additional weeks to extend
            
        Returns:
            Transaction result
        """
        try:
            await self._validate_lock(token_id)
            
            # Get current lock info
            lock_info = await self.get_lock_info(token_id)
            new_unlock_time = lock_info['end'] + (additional_weeks * 7 * 24 * 60 * 60)
            
            extend_args = {
                'tokenId': token_id,
                'lockDuration': new_unlock_time
            }
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.ve_address,
                method='increase_unlock_time',
                args=extend_args,
                abi=self.ve_abi
            )
            
            result.update({
                'operation': 'increase_unlock_time',
                'token_id': token_id,
                'additional_weeks': additional_weeks,
                'new_unlock_time': new_unlock_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Extend lock failed: {str(e)}")
            raise TransactionError(f"Extend lock failed: {str(e)}")
    
    async def withdraw(self, token_id: int) -> Dict[str, Any]:
        """
        Withdraw from expired veAERO lock.
        
        Args:
            token_id: veAERO NFT token ID
            
        Returns:
            Transaction result
        """
        try:
            # Check if lock is expired
            lock_info = await self.get_lock_info(token_id)
            if lock_info['end'] > int(time.time()):
                raise ContractError(f"Lock not yet expired. Expires at {lock_info['end']}")
            
            result = await self.contract_ops.invoke_contract(
                contract_address=self.ve_address,
                method='withdraw',
                args={'tokenId': token_id},
                abi=self.ve_abi
            )
            
            result.update({
                'operation': 'withdraw',
                'token_id': token_id,
                'withdrawn_amount': lock_info['amount']
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Withdraw failed: {str(e)}")
            raise TransactionError(f"Withdraw failed: {str(e)}")
    
    async def get_voting_power(self, token_id: int) -> int:
        """
        Get current voting power of veAERO NFT.
        
        Args:
            token_id: veAERO NFT token ID
            
        Returns:
            Current voting power
        """
        return await self.contract_ops.read_contract(
            contract_address=self.ve_address,
            method='balanceOfNFT',
            args={'tokenId': token_id},
            abi=self.ve_abi
        )
    
    async def get_lock_info(self, token_id: int) -> Dict[str, Any]:
        """
        Get detailed lock information.
        
        Args:
            token_id: veAERO NFT token ID
            
        Returns:
            Lock information dictionary
        """
        lock_data = await self.contract_ops.read_contract(
            contract_address=self.ve_address,
            method='locked',
            args={'tokenId': token_id},
            abi=self.ve_abi
        )
        
        return {
            'amount': int(lock_data['amount']),
            'end': int(lock_data['end']),
            'expired': lock_data['end'] <= int(time.time()),
            'weeks_remaining': max(0, (lock_data['end'] - int(time.time())) // (7 * 24 * 60 * 60))
        }
    
    async def get_owned_tokens(self) -> List[int]:
        """
        Get all veAERO tokens owned by wallet.
        
        Returns:
            List of owned token IDs
        """
        try:
            balance = await self.contract_ops.read_contract(
                contract_address=self.ve_address,
                method='balanceOf',
                args={'owner': self.cdp.wallet_address},
                abi=self.ve_abi
            )
            
            token_ids = []
            for i in range(balance):
                token_id = await self.contract_ops.read_contract(
                    contract_address=self.ve_address,
                    method='tokenOfOwnerByIndex',
                    args={'owner': self.cdp.wallet_address, 'index': i},
                    abi=self.ve_abi
                )
                token_ids.append(token_id)
            
            return token_ids
            
        except Exception as e:
            self.logger.error(f"Failed to get owned tokens: {str(e)}")
            return []
    
    async def _validate_lock(self, token_id: int) -> None:
        """
        Validate lock exists and is owned by wallet.
        
        Args:
            token_id: Token ID to validate
        """
        try:
            owner = await self.contract_ops.read_contract(
                contract_address=self.ve_address,
                method='ownerOf',
                args={'tokenId': token_id},
                abi=self.ve_abi
            )
            
            if owner.lower() != self.cdp.wallet_address.lower():
                raise ContractError(f"veAERO token {token_id} not owned by wallet")
                
        except Exception as e:
            raise ContractError(f"Lock validation failed: {str(e)}")
    
    async def _ensure_aero_approval(self, amount: int) -> None:
        """
        Ensure AERO approval for veAERO contract.
        
        Args:
            amount: Amount to approve
        """
        allowance = await self.contract_ops.read_contract(
            contract_address=self.aero_address,
            method='allowance',
            args={
                'owner': self.cdp.wallet_address,
                'spender': self.ve_address
            },
            abi=get_abi('ERC20')
        )
        
        if allowance < amount:
            self.logger.info("Approving AERO for veAERO contract")
            approval_result = await self.contract_ops.invoke_contract(
                contract_address=self.aero_address,
                method='approve',
                args={
                    'spender': self.ve_address,
                    'amount': amount * 2  # Approve extra to avoid frequent re-approvals
                },
                abi=get_abi('ERC20')
            )
            
            if not approval_result['success']:
                raise TransactionError("AERO approval failed")


# Export public interface
__all__ = [
    'AerodromeRouter',
    'AerodromeVoter', 
    'VotingEscrow',
    'SwapRoute',
    'LiquidityPosition',
    'VoteAllocation'
]