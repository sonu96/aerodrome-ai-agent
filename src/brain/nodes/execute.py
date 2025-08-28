"""
Action execution node for the Aerodrome Brain.

This node handles the actual execution of selected actions via the CDP SDK,
including transaction construction, simulation, submission, and monitoring.
Implements robust error handling and recovery strategies.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from ..state import BrainState, BrainConfig, ActionPlan, ExecutionResult


class ExecutionNode:
    """
    Action execution node that executes selected actions via CDP SDK.
    
    This node handles:
    - Transaction parameter construction
    - Pre-execution validation and simulation
    - Transaction submission via CDP SDK
    - Transaction monitoring and confirmation
    - Post-execution analysis
    - Error handling and recovery
    """

    def __init__(self, cdp_manager, config: BrainConfig):
        """
        Initialize the execution node.
        
        Args:
            cdp_manager: CDP SDK manager for blockchain interactions
            config: Brain configuration parameters
        """
        self.cdp_manager = cdp_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Execution parameters
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.confirmation_timeout = 300  # 5 minutes
        
        # Contract addresses (Base network)
        self.AERODROME_ROUTER = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
        self.AERODROME_FACTORY = "0x420DD381b31aEf6683db6B902084cB0FFECe40Da"
        self.VOTER_CONTRACT = "0x16613524e02ad97eDfeF371bC883F2F5d6C480A5"

    async def execute_action(self, state: BrainState) -> BrainState:
        """
        Execute the selected action.
        
        Args:
            state: Current brain state with selected action and execution plan
            
        Returns:
            Updated state with execution results
        """
        
        self.logger.info("Starting action execution")
        
        selected_action = state.get('selected_action')
        execution_plan = state.get('execution_plan')
        
        if not selected_action or not execution_plan:
            self.logger.warning("No action or execution plan found")
            return {
                **state,
                'execution_result': {
                    'success': False,
                    'error_type': 'NO_ACTION',
                    'error_message': 'No action selected for execution',
                    'execution_time': 0,
                    'recovery_attempted': False
                }
            }
        
        action_type = selected_action.get('action_type')
        
        try:
            # Pre-execution validation
            validation_result = await self._pre_execution_validation(state)
            if not validation_result['valid']:
                return self._create_failed_execution_state(
                    state, 
                    'VALIDATION_FAILED', 
                    validation_result['reason']
                )
            
            # Build transaction parameters
            tx_params = await self._build_transaction_parameters(state)
            
            # Simulate transaction if required
            if self.config.simulation_required:
                simulation_result = await self._simulate_transaction(tx_params, state)
                
                # Update state with simulation result
                state = {**state, 'simulation_result': simulation_result}
                
                if not simulation_result.get('success', False):
                    return self._create_failed_execution_state(
                        state,
                        'SIMULATION_FAILED',
                        simulation_result.get('error', 'Transaction simulation failed')
                    )
            
            # Execute the transaction
            execution_start = time.time()
            execution_result = await self._execute_transaction_with_retry(tx_params, action_type)
            execution_time = time.time() - execution_start
            
            # Monitor transaction
            if execution_result.get('success', False):
                monitoring_result = await self._monitor_transaction(
                    execution_result['transaction_hash'],
                    execution_plan,
                    state
                )
                
                # Merge monitoring results
                execution_result.update(monitoring_result)
            
            # Add execution timing
            execution_result['execution_time'] = execution_time
            
            # Update state with execution results
            updated_state = {
                **state,
                'execution_result': execution_result,
                'transaction_params': tx_params,
                'debug_logs': state.get('debug_logs', []) + [
                    f"Executed {action_type}: {'success' if execution_result.get('success') else 'failed'}"
                ]
            }
            
            if execution_result.get('success'):
                self.logger.info(f"Action execution completed successfully in {execution_time:.2f}s")
            else:
                self.logger.error(f"Action execution failed: {execution_result.get('error_message')}")
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Critical error in action execution: {e}")
            
            return self._create_failed_execution_state(
                state,
                'CRITICAL_ERROR',
                f"Execution failed with error: {str(e)}"
            )

    async def _pre_execution_validation(self, state: BrainState) -> Dict[str, Any]:
        """Perform comprehensive pre-execution validation."""
        
        try:
            # Check wallet connection
            if not self._is_wallet_connected():
                return {'valid': False, 'reason': 'Wallet not connected'}
            
            # Check sufficient balances
            balance_check = await self._check_sufficient_balances(state)
            if not balance_check['sufficient']:
                return {'valid': False, 'reason': f"Insufficient balance: {balance_check['reason']}"}
            
            # Check gas price limits
            gas_price = state.get('gas_price', 0)
            if gas_price > self.config.max_gas_price:
                return {'valid': False, 'reason': f"Gas price too high: {gas_price}"}
            
            # Check network conditions
            congestion = state.get('network_congestion', 0)
            if congestion > 0.9:  # Very high congestion
                return {'valid': False, 'reason': 'Network congestion too high'}
            
            # Check execution plan validity
            execution_plan = state.get('execution_plan')
            if not self._is_execution_plan_valid(execution_plan):
                return {'valid': False, 'reason': 'Invalid execution plan'}
            
            return {'valid': True, 'reason': 'All validations passed'}
            
        except Exception as e:
            self.logger.error(f"Error in pre-execution validation: {e}")
            return {'valid': False, 'reason': f"Validation error: {str(e)}"}

    async def _build_transaction_parameters(self, state: BrainState) -> Dict[str, Any]:
        """Build comprehensive transaction parameters."""
        
        selected_action = state['selected_action']
        execution_plan = state['execution_plan']
        action_type = selected_action['action_type']
        
        # Base transaction parameters
        base_params = {
            'from': await self._get_wallet_address(),
            'deadline': int(time.time()) + (execution_plan['deadline_minutes'] * 60),
            'gas_strategy': execution_plan['gas_strategy'],
            'slippage_tolerance': execution_plan['slippage_tolerance']
        }
        
        # Action-specific parameters
        if action_type == 'SWAP':
            action_params = await self._build_swap_parameters(selected_action, execution_plan, state)
        elif action_type == 'ADD_LIQUIDITY':
            action_params = await self._build_add_liquidity_parameters(selected_action, execution_plan, state)
        elif action_type == 'REMOVE_LIQUIDITY':
            action_params = await self._build_remove_liquidity_parameters(selected_action, execution_plan, state)
        elif action_type == 'VOTE':
            action_params = await self._build_vote_parameters(selected_action, execution_plan, state)
        elif action_type == 'CLAIM':
            action_params = await self._build_claim_parameters(selected_action, execution_plan, state)
        elif action_type == 'STAKE_LP':
            action_params = await self._build_stake_parameters(selected_action, execution_plan, state)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
        
        # Merge parameters
        tx_params = {**base_params, **action_params}
        
        # Add gas optimization
        gas_params = await self._optimize_gas_parameters(state)
        tx_params.update(gas_params)
        
        return tx_params

    async def _simulate_transaction(
        self, 
        tx_params: Dict[str, Any], 
        state: BrainState
    ) -> Dict[str, Any]:
        """Simulate transaction execution to predict outcomes."""
        
        try:
            self.logger.info("Simulating transaction")
            
            # This would use CDP SDK's simulation capabilities
            # For now, return a basic simulation result
            
            simulation_result = {
                'success': True,
                'profitable': True,
                'expected_gas': 150000,
                'expected_slippage': tx_params.get('slippage_tolerance', 0.01) * 0.5,
                'price_impact': 0.001,
                'simulation_time': time.time()
            }
            
            # Validate simulation results
            if simulation_result['expected_slippage'] > tx_params.get('slippage_tolerance', 0.01):
                simulation_result['success'] = False
                simulation_result['error'] = 'Expected slippage exceeds tolerance'
            
            if simulation_result['price_impact'] > 0.03:  # 3% max price impact
                simulation_result['success'] = False
                simulation_result['error'] = 'Price impact too high'
            
            return simulation_result
            
        except Exception as e:
            self.logger.error(f"Error in transaction simulation: {e}")
            return {
                'success': False,
                'error': f"Simulation failed: {str(e)}",
                'simulation_time': time.time()
            }

    async def _execute_transaction_with_retry(
        self, 
        tx_params: Dict[str, Any], 
        action_type: str
    ) -> ExecutionResult:
        """Execute transaction with retry logic."""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Executing {action_type} (attempt {attempt + 1}/{self.max_retries})")
                
                # Execute based on action type
                if action_type == 'SWAP':
                    result = await self._execute_swap(tx_params)
                elif action_type == 'ADD_LIQUIDITY':
                    result = await self._execute_add_liquidity(tx_params)
                elif action_type == 'REMOVE_LIQUIDITY':
                    result = await self._execute_remove_liquidity(tx_params)
                elif action_type == 'VOTE':
                    result = await self._execute_vote(tx_params)
                elif action_type == 'CLAIM':
                    result = await self._execute_claim(tx_params)
                elif action_type == 'STAKE_LP':
                    result = await self._execute_stake(tx_params)
                else:
                    raise ValueError(f"Unsupported action type: {action_type}")
                
                # If successful, return result
                if result.get('success', False):
                    result['recovery_attempted'] = attempt > 0
                    return result
                
                last_error = result.get('error_message', 'Unknown error')
                
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Execution attempt {attempt + 1} failed: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        return {
            'success': False,
            'transaction_hash': None,
            'gas_used': None,
            'gas_price_paid': None,
            'execution_time': 0,
            'tokens_received': {},
            'tokens_spent': {},
            'net_value_change': 0,
            'slippage_experienced': 0,
            'error_type': 'EXECUTION_FAILED',
            'error_message': f"All {self.max_retries} execution attempts failed. Last error: {last_error}",
            'recovery_attempted': True,
            'new_balances': {},
            'new_positions': []
        }

    async def _monitor_transaction(
        self, 
        tx_hash: str, 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Monitor transaction until confirmation."""
        
        try:
            self.logger.info(f"Monitoring transaction {tx_hash}")
            
            start_time = time.time()
            timeout = self.confirmation_timeout
            
            while time.time() - start_time < timeout:
                # Check transaction status
                tx_status = await self._get_transaction_status(tx_hash)
                
                if tx_status['confirmed']:
                    # Transaction confirmed, analyze results
                    results = await self._analyze_execution_results(tx_hash, tx_status, execution_plan)
                    
                    self.logger.info(f"Transaction confirmed: {tx_hash}")
                    return {
                        'confirmed': True,
                        'confirmation_time': time.time() - start_time,
                        **results
                    }
                
                elif tx_status['failed']:
                    # Transaction failed
                    self.logger.error(f"Transaction failed: {tx_hash}")
                    return {
                        'confirmed': False,
                        'failed': True,
                        'error_message': tx_status.get('error', 'Transaction failed'),
                        'gas_used': tx_status.get('gas_used', 0)
                    }
                
                # Wait before next check
                await asyncio.sleep(5)
            
            # Timeout reached
            self.logger.warning(f"Transaction monitoring timeout: {tx_hash}")
            return {
                'confirmed': False,
                'timeout': True,
                'error_message': 'Transaction confirmation timeout',
                'monitoring_time': timeout
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring transaction: {e}")
            return {
                'confirmed': False,
                'monitoring_error': True,
                'error_message': f"Monitoring failed: {str(e)}"
            }

    async def _analyze_execution_results(
        self, 
        tx_hash: str, 
        tx_status: Dict[str, Any], 
        execution_plan: ActionPlan
    ) -> Dict[str, Any]:
        """Analyze the results of the executed transaction."""
        
        try:
            # Get transaction receipt and details
            receipt = tx_status.get('receipt', {})
            logs = tx_status.get('logs', [])
            
            # Calculate actual gas costs
            gas_used = receipt.get('gasUsed', 0)
            gas_price = receipt.get('effectiveGasPrice', 0)
            gas_cost_eth = (gas_used * gas_price) / 1e18
            
            # Analyze token transfers from logs
            transfers = self._parse_transfer_events(logs)
            
            # Calculate actual slippage
            expected_outcome = execution_plan.get('expected_outcome', {})
            actual_slippage = self._calculate_actual_slippage(transfers, expected_outcome)
            
            # Calculate net value change
            net_value_change = self._calculate_net_value_change(transfers, gas_cost_eth)
            
            # Get updated balances
            new_balances = await self._get_updated_balances()
            
            # Get updated positions
            new_positions = await self._get_updated_positions()
            
            return {
                'gas_used': gas_used,
                'gas_price_paid': gas_price,
                'gas_cost_eth': gas_cost_eth,
                'tokens_received': transfers.get('received', {}),
                'tokens_spent': transfers.get('spent', {}),
                'net_value_change': net_value_change,
                'slippage_experienced': actual_slippage,
                'new_balances': new_balances,
                'new_positions': new_positions,
                'success_criteria_met': self._check_success_criteria(
                    execution_plan['success_criteria'],
                    {
                        'slippage': actual_slippage,
                        'gas_cost': gas_cost_eth,
                        'net_change': net_value_change
                    }
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing execution results: {e}")
            return {
                'analysis_error': True,
                'error_message': f"Result analysis failed: {str(e)}"
            }

    def _create_failed_execution_state(
        self, 
        state: BrainState, 
        error_type: str, 
        error_message: str
    ) -> BrainState:
        """Create state for failed execution."""
        
        execution_result = {
            'success': False,
            'transaction_hash': None,
            'gas_used': None,
            'gas_price_paid': None,
            'execution_time': 0,
            'tokens_received': {},
            'tokens_spent': {},
            'net_value_change': 0,
            'slippage_experienced': 0,
            'error_type': error_type,
            'error_message': error_message,
            'recovery_attempted': False,
            'new_balances': {},
            'new_positions': []
        }
        
        return {
            **state,
            'execution_result': execution_result,
            'warnings': state.get('warnings', []) + [f"Execution failed: {error_message}"]
        }

    # Action-specific execution methods

    async def _execute_swap(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute token swap via CDP SDK."""
        
        try:
            # Build swap transaction
            transaction = await self.cdp_manager.wallet.invoke_contract({
                'contract_address': self.AERODROME_ROUTER,
                'method': 'swapExactTokensForTokens',
                'args': {
                    'amountIn': tx_params['amount_in'],
                    'amountOutMin': tx_params['amount_out_min'],
                    'routes': tx_params['routes'],
                    'to': tx_params['from'],
                    'deadline': tx_params['deadline']
                },
                'abi': self._get_router_abi(),
                **self._extract_gas_params(tx_params)
            })
            
            return {
                'success': True,
                'transaction_hash': transaction.transaction_hash,
                'raw_transaction': transaction
            }
            
        except Exception as e:
            self.logger.error(f"Swap execution failed: {e}")
            return {
                'success': False,
                'error_type': 'SWAP_FAILED',
                'error_message': str(e)
            }

    async def _execute_add_liquidity(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute add liquidity via CDP SDK."""
        
        try:
            transaction = await self.cdp_manager.wallet.invoke_contract({
                'contract_address': self.AERODROME_ROUTER,
                'method': 'addLiquidity',
                'args': {
                    'tokenA': tx_params['token_a'],
                    'tokenB': tx_params['token_b'],
                    'stable': tx_params['stable'],
                    'amountADesired': tx_params['amount_a_desired'],
                    'amountBDesired': tx_params['amount_b_desired'],
                    'amountAMin': tx_params['amount_a_min'],
                    'amountBMin': tx_params['amount_b_min'],
                    'to': tx_params['from'],
                    'deadline': tx_params['deadline']
                },
                'abi': self._get_router_abi(),
                **self._extract_gas_params(tx_params)
            })
            
            return {
                'success': True,
                'transaction_hash': transaction.transaction_hash,
                'raw_transaction': transaction
            }
            
        except Exception as e:
            self.logger.error(f"Add liquidity execution failed: {e}")
            return {
                'success': False,
                'error_type': 'ADD_LIQUIDITY_FAILED', 
                'error_message': str(e)
            }

    async def _execute_remove_liquidity(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute remove liquidity via CDP SDK."""
        
        try:
            transaction = await self.cdp_manager.wallet.invoke_contract({
                'contract_address': self.AERODROME_ROUTER,
                'method': 'removeLiquidity',
                'args': {
                    'tokenA': tx_params['token_a'],
                    'tokenB': tx_params['token_b'],
                    'stable': tx_params['stable'],
                    'liquidity': tx_params['liquidity_amount'],
                    'amountAMin': tx_params['amount_a_min'],
                    'amountBMin': tx_params['amount_b_min'],
                    'to': tx_params['from'],
                    'deadline': tx_params['deadline']
                },
                'abi': self._get_router_abi(),
                **self._extract_gas_params(tx_params)
            })
            
            return {
                'success': True,
                'transaction_hash': transaction.transaction_hash,
                'raw_transaction': transaction
            }
            
        except Exception as e:
            self.logger.error(f"Remove liquidity execution failed: {e}")
            return {
                'success': False,
                'error_type': 'REMOVE_LIQUIDITY_FAILED',
                'error_message': str(e)
            }

    async def _execute_vote(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute governance vote via CDP SDK."""
        
        try:
            transaction = await self.cdp_manager.wallet.invoke_contract({
                'contract_address': self.VOTER_CONTRACT,
                'method': 'vote',
                'args': {
                    'tokenId': tx_params['token_id'],
                    'poolVotes': tx_params['pool_votes'],
                    'weights': tx_params['weights']
                },
                'abi': self._get_voter_abi(),
                **self._extract_gas_params(tx_params)
            })
            
            return {
                'success': True,
                'transaction_hash': transaction.transaction_hash,
                'raw_transaction': transaction
            }
            
        except Exception as e:
            self.logger.error(f"Vote execution failed: {e}")
            return {
                'success': False,
                'error_type': 'VOTE_FAILED',
                'error_message': str(e)
            }

    async def _execute_claim(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rewards claim via CDP SDK."""
        
        try:
            # This would depend on the specific rewards contract
            # Placeholder implementation
            transaction = await self.cdp_manager.wallet.invoke_contract({
                'contract_address': tx_params['rewards_contract'],
                'method': 'claimRewards',
                'args': {
                    'tokenId': tx_params.get('token_id'),
                    'pools': tx_params.get('pools', [])
                },
                'abi': tx_params.get('rewards_abi', []),
                **self._extract_gas_params(tx_params)
            })
            
            return {
                'success': True,
                'transaction_hash': transaction.transaction_hash,
                'raw_transaction': transaction
            }
            
        except Exception as e:
            self.logger.error(f"Claim execution failed: {e}")
            return {
                'success': False,
                'error_type': 'CLAIM_FAILED',
                'error_message': str(e)
            }

    async def _execute_stake(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LP token staking via CDP SDK."""
        
        try:
            # This would interact with the gauge contract
            # Placeholder implementation
            transaction = await self.cdp_manager.wallet.invoke_contract({
                'contract_address': tx_params['gauge_contract'],
                'method': 'deposit',
                'args': {
                    'amount': tx_params['lp_amount']
                },
                'abi': tx_params.get('gauge_abi', []),
                **self._extract_gas_params(tx_params)
            })
            
            return {
                'success': True,
                'transaction_hash': transaction.transaction_hash,
                'raw_transaction': transaction
            }
            
        except Exception as e:
            self.logger.error(f"Stake execution failed: {e}")
            return {
                'success': False,
                'error_type': 'STAKE_FAILED',
                'error_message': str(e)
            }

    # Helper methods (many would be implemented with actual CDP SDK calls)

    def _is_wallet_connected(self) -> bool:
        """Check if wallet is connected and ready."""
        return self.cdp_manager and hasattr(self.cdp_manager, 'wallet')

    async def _check_sufficient_balances(self, state: BrainState) -> Dict[str, Any]:
        """Check if wallet has sufficient balances for the transaction."""
        # Placeholder - would implement actual balance checking
        return {'sufficient': True, 'reason': 'Balance check passed'}

    def _is_execution_plan_valid(self, execution_plan: Optional[ActionPlan]) -> bool:
        """Validate execution plan structure."""
        if not execution_plan:
            return False
        
        required_fields = ['action_type', 'pool_address', 'amounts']
        return all(field in execution_plan for field in required_fields)

    async def _get_wallet_address(self) -> str:
        """Get the wallet address."""
        if self.cdp_manager and hasattr(self.cdp_manager, 'wallet'):
            return self.cdp_manager.wallet.default_address.address_id
        return ""

    async def _optimize_gas_parameters(self, state: BrainState) -> Dict[str, Any]:
        """Optimize gas parameters based on current conditions."""
        
        gas_price = state.get('gas_price', 0.001)  # gwei
        base_fee_trend = state.get('base_fee_trend', 'stable')
        
        if base_fee_trend == 'increasing':
            return {
                'maxFeePerGas': int(gas_price * 1.2 * 1e9),  # Convert to wei
                'maxPriorityFeePerGas': int(2e9)  # 2 gwei priority
            }
        else:
            return {
                'maxFeePerGas': int(gas_price * 1.05 * 1e9),
                'maxPriorityFeePerGas': int(1e9)  # 1 gwei priority
            }

    def _extract_gas_params(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract gas parameters from transaction parameters."""
        gas_params = {}
        
        if 'maxFeePerGas' in tx_params:
            gas_params['maxFeePerGas'] = tx_params['maxFeePerGas']
        
        if 'maxPriorityFeePerGas' in tx_params:
            gas_params['maxPriorityFeePerGas'] = tx_params['maxPriorityFeePerGas']
        
        return gas_params

    async def _get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status and receipt."""
        # Placeholder - would implement actual transaction status checking
        await asyncio.sleep(2)  # Simulate network delay
        
        return {
            'confirmed': True,
            'failed': False,
            'receipt': {
                'gasUsed': 150000,
                'effectiveGasPrice': int(0.001 * 1e9),
                'status': 1
            },
            'logs': []
        }

    def _parse_transfer_events(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse transfer events from transaction logs."""
        # Placeholder - would implement actual log parsing
        return {
            'received': {},
            'spent': {}
        }

    def _calculate_actual_slippage(
        self, 
        transfers: Dict[str, Any], 
        expected_outcome: Dict[str, Any]
    ) -> float:
        """Calculate actual slippage experienced."""
        # Placeholder calculation
        return 0.001  # 0.1% slippage

    def _calculate_net_value_change(
        self, 
        transfers: Dict[str, Any], 
        gas_cost_eth: float
    ) -> float:
        """Calculate net value change from the transaction."""
        # Placeholder - would implement actual value calculation
        return 100.0  # $100 gain

    async def _get_updated_balances(self) -> Dict[str, float]:
        """Get updated wallet balances after transaction."""
        # Placeholder
        return {}

    async def _get_updated_positions(self) -> List[Dict[str, Any]]:
        """Get updated positions after transaction."""
        # Placeholder
        return []

    def _check_success_criteria(
        self, 
        criteria: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> bool:
        """Check if execution met success criteria."""
        
        if criteria.get('slippage_within_tolerance', False):
            if results['slippage'] > 0.05:  # 5% max
                return False
        
        if criteria.get('gas_cost_reasonable', False):
            if results['gas_cost'] > 0.01:  # $10 max gas cost (assuming ETH price)
                return False
        
        if criteria.get('expected_outcome_achieved', False):
            if results['net_change'] <= 0:
                return False
        
        return True

    # Parameter building methods (placeholders - would implement actual parameter construction)

    async def _build_swap_parameters(
        self, 
        selected_action: Dict[str, Any], 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Build parameters for swap transaction."""
        amounts = execution_plan['amounts']
        
        return {
            'amount_in': amounts['amount_in'],
            'amount_out_min': int(amounts['expected_amount_out'] * (1 - execution_plan['slippage_tolerance'])),
            'routes': [],  # Would construct route array
            'to': await self._get_wallet_address()
        }

    async def _build_add_liquidity_parameters(
        self, 
        selected_action: Dict[str, Any], 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Build parameters for add liquidity transaction."""
        amounts = execution_plan['amounts']
        tokens = execution_plan['token_addresses']
        
        return {
            'token_a': tokens[0],
            'token_b': tokens[1],
            'stable': selected_action['pool_metadata'].get('is_stable', False),
            'amount_a_desired': amounts['token0_amount'],
            'amount_b_desired': amounts['token1_amount'],
            'amount_a_min': int(amounts['token0_amount'] * (1 - execution_plan['slippage_tolerance'])),
            'amount_b_min': int(amounts['token1_amount'] * (1 - execution_plan['slippage_tolerance'])),
        }

    async def _build_remove_liquidity_parameters(
        self, 
        selected_action: Dict[str, Any], 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Build parameters for remove liquidity transaction."""
        # Placeholder
        return {}

    async def _build_vote_parameters(
        self, 
        selected_action: Dict[str, Any], 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Build parameters for vote transaction."""
        # Placeholder
        return {}

    async def _build_claim_parameters(
        self, 
        selected_action: Dict[str, Any], 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Build parameters for claim transaction."""
        # Placeholder
        return {}

    async def _build_stake_parameters(
        self, 
        selected_action: Dict[str, Any], 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> Dict[str, Any]:
        """Build parameters for stake transaction."""
        # Placeholder
        return {}

    def _get_router_abi(self) -> List[Dict[str, Any]]:
        """Get Aerodrome router ABI."""
        # Would return actual ABI
        return []

    def _get_voter_abi(self) -> List[Dict[str, Any]]:
        """Get voter contract ABI."""
        # Would return actual ABI
        return []

    async def cancel_pending_transactions(self):
        """Cancel any pending transactions (emergency stop)."""
        try:
            # Would implement transaction cancellation logic
            self.logger.info("Cancelling pending transactions")
        except Exception as e:
            self.logger.error(f"Error cancelling transactions: {e}")