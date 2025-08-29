"""
Decision making node for the Aerodrome Brain.

This node makes final decisions about which actions to execute based on:
- Opportunity analysis results
- Risk assessment
- Portfolio constraints
- Memory-based insights
- Market conditions

Implements sophisticated decision logic with multi-criteria evaluation.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from ..state import BrainState, BrainConfig, RiskAssessment, ActionPlan


class DecisionNode:
    """
    Decision making node that selects the best action to execute.
    
    This node performs comprehensive decision analysis including:
    - Multi-criteria decision analysis (MCDA)
    - Risk-adjusted opportunity selection
    - Portfolio constraint checking
    - Gas cost optimization
    - Failure pattern avoidance
    - Confidence-based thresholds
    """

    def __init__(self, config: BrainConfig):
        """
        Initialize the decision node.
        
        Args:
            config: Brain configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Decision criteria weights
        self.decision_weights = {
            'opportunity_score': 0.30,
            'risk_score': 0.25,
            'confidence': 0.20,
            'portfolio_fit': 0.15,
            'timing': 0.10
        }
        
        # Risk tolerance parameters
        self.max_portfolio_risk = 0.30
        self.max_individual_position_risk = 0.20

    async def make_decision(self, state: BrainState) -> BrainState:
        """
        Make final decision about which action to execute.
        
        Args:
            state: Current brain state with opportunities and analysis
            
        Returns:
            Updated state with selected action and decision reasoning
        """
        
        self.logger.info("Starting decision making process")
        
        try:
            opportunities = state.get('opportunities', [])
            
            if not opportunities:
                self.logger.info("No opportunities available for decision")
                return self._create_no_action_decision(state, "No opportunities found")
            
            # Filter opportunities by basic constraints
            viable_opportunities = self._filter_viable_opportunities(opportunities, state)
            
            if not viable_opportunities:
                self.logger.info("No viable opportunities after filtering")
                return self._create_no_action_decision(state, "No opportunities passed filters")
            
            # Perform risk assessment for each viable opportunity
            risk_assessed_opportunities = await self._assess_risks(viable_opportunities, state)
            
            # Filter by risk tolerance
            acceptable_risk_opportunities = self._filter_by_risk_tolerance(
                risk_assessed_opportunities, state
            )
            
            if not acceptable_risk_opportunities:
                self.logger.info("No opportunities within risk tolerance")
                return self._create_no_action_decision(state, "All opportunities exceed risk tolerance")
            
            # Apply decision criteria and select best opportunity
            best_opportunity = self._select_best_opportunity(acceptable_risk_opportunities, state)
            
            if not best_opportunity:
                self.logger.info("No opportunity selected after decision analysis")
                return self._create_no_action_decision(state, "Decision analysis rejected all opportunities")
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(best_opportunity, state)
            
            # Calculate final confidence score
            confidence_score = self._calculate_decision_confidence(
                best_opportunity, execution_plan, state
            )
            
            # Final decision validation
            if confidence_score < self.config.confidence_threshold:
                self.logger.info(f"Decision confidence too low: {confidence_score}")
                return self._create_no_action_decision(
                    state, 
                    f"Decision confidence {confidence_score:.3f} below threshold {self.config.confidence_threshold}"
                )
            
            # Generate decision reasoning
            reasoning = self._generate_decision_reasoning(
                best_opportunity, execution_plan, confidence_score, state
            )
            
            # Update state with decision
            updated_state = {
                **state,
                'selected_action': self._format_selected_action(best_opportunity),
                'confidence_score': confidence_score,
                'execution_plan': execution_plan,
                'decision_reasoning': reasoning,
                'risk_assessment': best_opportunity.get('risk_assessment', {}),
                'debug_logs': state.get('debug_logs', []) + [
                    f"Selected action: {best_opportunity['action_type']} on {best_opportunity['pool_address'][:8]}..."
                ]
            }
            
            self.logger.info(
                f"Decision made: {best_opportunity['action_type']} with confidence {confidence_score:.3f}"
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {e}")
            
            return self._create_no_action_decision(
                state, 
                f"Decision making failed: {str(e)}"
            )

    def _filter_viable_opportunities(
        self, 
        opportunities: List[Dict[str, Any]], 
        state: BrainState
    ) -> List[Dict[str, Any]]:
        """Filter opportunities by basic viability criteria."""
        
        viable = []
        portfolio_value = state.get('portfolio_performance', {}).get('total_value', 0)
        
        for opp in opportunities:
            # Check minimum score threshold
            score = opp.get('score', {}).get('total_score', 0)
            if score < self.config.min_opportunity_score:
                continue
            
            # Check minimum confidence threshold
            confidence = opp.get('score', {}).get('confidence', 0)
            if confidence < 0.4:  # Basic confidence threshold
                continue
            
            # Check position size constraints
            optimal_size = opp.get('optimal_size_usd', 0)
            if optimal_size > portfolio_value * self.config.max_position_size:
                continue
            
            # Check gas price constraints
            gas_price = state.get('gas_price', 0)
            if gas_price > self.config.max_gas_price:
                continue
            
            # Check failure patterns
            if self._matches_failure_pattern(opp, state):
                continue
            
            viable.append(opp)
        
        self.logger.info(f"Filtered to {len(viable)} viable opportunities from {len(opportunities)}")
        return viable

    async def _assess_risks(
        self, 
        opportunities: List[Dict[str, Any]], 
        state: BrainState
    ) -> List[Dict[str, Any]]:
        """Perform comprehensive risk assessment for each opportunity."""
        
        risk_assessed = []
        
        for opp in opportunities:
            try:
                risk_assessment = await self._calculate_comprehensive_risk(opp, state)
                
                # Add risk assessment to opportunity
                opp_with_risk = {
                    **opp,
                    'risk_assessment': risk_assessment
                }
                
                risk_assessed.append(opp_with_risk)
                
            except Exception as e:
                self.logger.error(f"Error assessing risk for opportunity: {e}")
                continue
        
        return risk_assessed

    async def _calculate_comprehensive_risk(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> RiskAssessment:
        """Calculate comprehensive risk assessment for an opportunity."""
        
        # Position risk
        position_risk = self._calculate_position_risk(opportunity, state)
        
        # Market risk  
        market_risk = self._calculate_market_risk(opportunity, state)
        
        # Liquidity risk
        liquidity_risk = self._calculate_liquidity_risk(opportunity, state)
        
        # Smart contract risk
        contract_risk = self._calculate_contract_risk(opportunity, state)
        
        # Timing risk
        timing_risk = self._calculate_timing_risk(opportunity, state)
        
        # Impermanent loss risk (for LP positions)
        il_risk = self._calculate_il_risk(opportunity, state)
        
        # Calculate total risk (weighted average)
        risk_weights = {
            'position': 0.25,
            'market': 0.20,
            'liquidity': 0.15,
            'contract': 0.15,
            'timing': 0.15,
            'il': 0.10
        }
        
        total_risk = (
            position_risk * risk_weights['position'] +
            market_risk * risk_weights['market'] +
            liquidity_risk * risk_weights['liquidity'] +
            contract_risk * risk_weights['contract'] +
            timing_risk * risk_weights['timing'] +
            il_risk * risk_weights['il']
        )
        
        # Determine acceptability
        acceptable = total_risk <= self.config.risk_threshold
        
        # Generate recommendation
        if total_risk <= 0.2:
            recommendation = "proceed"
        elif total_risk <= 0.4:
            recommendation = "caution"
        else:
            recommendation = "abort"
        
        # Generate risk factors and mitigations
        risk_factors = self._identify_risk_factors(
            position_risk, market_risk, liquidity_risk, 
            contract_risk, timing_risk, il_risk
        )
        
        mitigations = self._suggest_risk_mitigations(opportunity, risk_factors)
        
        return {
            'position_risk': position_risk,
            'market_risk': market_risk,
            'liquidity_risk': liquidity_risk,
            'contract_risk': contract_risk,
            'timing_risk': timing_risk,
            'impermanent_loss_risk': il_risk,
            'total_risk': total_risk,
            'acceptable': acceptable,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'mitigations': mitigations
        }

    def _filter_by_risk_tolerance(
        self, 
        opportunities: List[Dict[str, Any]], 
        state: BrainState
    ) -> List[Dict[str, Any]]:
        """Filter opportunities by risk tolerance."""
        
        acceptable = []
        
        for opp in opportunities:
            risk_assessment = opp.get('risk_assessment', {})
            
            if not risk_assessment.get('acceptable', False):
                continue
            
            # Additional portfolio-level risk checks
            if not self._fits_portfolio_risk_profile(opp, state):
                continue
            
            acceptable.append(opp)
        
        self.logger.info(f"{len(acceptable)} opportunities within risk tolerance")
        return acceptable

    def _select_best_opportunity(
        self, 
        opportunities: List[Dict[str, Any]], 
        state: BrainState
    ) -> Optional[Dict[str, Any]]:
        """Select the best opportunity using multi-criteria decision analysis."""
        
        if not opportunities:
            return None
        
        # Calculate decision scores for each opportunity
        scored_opportunities = []
        
        for opp in opportunities:
            decision_score = self._calculate_decision_score(opp, state)
            
            opp_with_score = {
                **opp,
                'decision_score': decision_score
            }
            scored_opportunities.append(opp_with_score)
        
        # Sort by decision score
        scored_opportunities.sort(key=lambda x: x['decision_score'], reverse=True)
        
        # Return best opportunity
        best = scored_opportunities[0]
        self.logger.info(f"Selected opportunity with decision score: {best['decision_score']:.3f}")
        
        return best

    def _calculate_decision_score(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate overall decision score using MCDA."""
        
        # Normalize scores to 0-1 scale
        opportunity_score = opportunity.get('score', {}).get('total_score', 0)
        
        # Risk score (invert since lower risk is better)
        risk_score = 1.0 - opportunity.get('risk_assessment', {}).get('total_risk', 0.5)
        
        # Confidence score
        confidence = opportunity.get('score', {}).get('confidence', 0.5)
        
        # Portfolio fit score
        portfolio_fit = self._calculate_portfolio_fit_score(opportunity, state)
        
        # Timing score
        timing_score = self._calculate_timing_score(opportunity, state)
        
        # Weighted combination
        decision_score = (
            opportunity_score * self.decision_weights['opportunity_score'] +
            risk_score * self.decision_weights['risk_score'] +
            confidence * self.decision_weights['confidence'] +
            portfolio_fit * self.decision_weights['portfolio_fit'] +
            timing_score * self.decision_weights['timing']
        )
        
        return decision_score

    async def _create_execution_plan(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> ActionPlan:
        """Create detailed execution plan for the selected opportunity."""
        
        action_type = opportunity['action_type']
        pool_address = opportunity['pool_address']
        pool_data = opportunity.get('pool_data', {})
        
        # Determine optimal amounts
        amounts = self._calculate_optimal_amounts(opportunity, state)
        
        # Calculate slippage tolerance based on market conditions
        slippage_tolerance = self._calculate_slippage_tolerance(opportunity, state)
        
        # Determine gas strategy
        gas_strategy = self._determine_gas_strategy(state)
        
        # Calculate deadline
        deadline_minutes = self._calculate_deadline(opportunity, state)
        
        # Set risk parameters
        max_acceptable_loss = self._calculate_max_acceptable_loss(opportunity, state)
        stop_loss_threshold = self._calculate_stop_loss_threshold(opportunity)
        
        # Determine monitoring parameters
        monitor_duration = self._calculate_monitor_duration(action_type)
        success_criteria = self._define_success_criteria(opportunity)
        
        execution_plan = {
            'action_type': action_type,
            'pool_address': pool_address,
            'token_addresses': self._extract_token_addresses(pool_data),
            'amounts': amounts,
            'expected_outcome': opportunity.get('expected_returns', {}),
            'slippage_tolerance': slippage_tolerance,
            'deadline_minutes': deadline_minutes,
            'gas_strategy': gas_strategy,
            'max_acceptable_loss': max_acceptable_loss,
            'stop_loss_threshold': stop_loss_threshold,
            'monitor_duration': monitor_duration,
            'success_criteria': success_criteria
        }
        
        return execution_plan

    def _calculate_decision_confidence(
        self, 
        opportunity: Dict[str, Any], 
        execution_plan: ActionPlan, 
        state: BrainState
    ) -> float:
        """Calculate final confidence score for the decision."""
        
        # Base confidence from opportunity analysis
        base_confidence = opportunity.get('score', {}).get('confidence', 0.5)
        
        # Risk adjustment (lower risk = higher confidence)
        risk_assessment = opportunity.get('risk_assessment', {})
        risk_factor = 1.0 - risk_assessment.get('total_risk', 0.5)
        
        # Memory-based confidence modifiers
        memory_insights = state.get('memory_insights', {})
        confidence_modifiers = memory_insights.get('confidence_modifiers', {})
        
        historical_modifier = confidence_modifiers.get('historical', 1.0)
        recency_modifier = confidence_modifiers.get('recency', 1.0)
        consistency_modifier = confidence_modifiers.get('consistency', 1.0)
        
        # Market condition modifier
        market_condition_modifier = self._calculate_market_condition_modifier(state)
        
        # Combine all factors
        final_confidence = (
            base_confidence * 0.4 +
            risk_factor * 0.3 +
            (historical_modifier * recency_modifier * consistency_modifier - 2.0) * 0.2 +
            market_condition_modifier * 0.1
        )
        
        return min(final_confidence, 1.0)

    def _create_no_action_decision(self, state: BrainState, reason: str) -> BrainState:
        """Create a decision state with no action selected."""
        
        return {
            **state,
            'selected_action': None,
            'confidence_score': 0.0,
            'execution_plan': None,
            'decision_reasoning': [f"No action selected: {reason}"],
            'debug_logs': state.get('debug_logs', []) + [f"Decision: No action - {reason}"]
        }

    # Risk calculation methods

    def _calculate_position_risk(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate position size risk."""
        
        optimal_size = opportunity.get('optimal_size_usd', 0)
        portfolio_value = state.get('portfolio_performance', {}).get('total_value', 0)
        
        if portfolio_value <= 0:
            return 1.0  # Max risk if no portfolio data
        
        position_ratio = optimal_size / portfolio_value
        
        # Risk increases exponentially with position size
        if position_ratio > self.config.max_position_size:
            return 1.0
        
        # Sigmoid-like scaling
        risk = position_ratio / self.config.max_position_size
        return min(risk * 2, 1.0)  # Scale up to emphasize large positions

    def _calculate_market_risk(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate market volatility risk."""
        
        pool_data = opportunity.get('pool_data', {})
        volatility = pool_data.get('volatility', 0.5)
        
        # Market sentiment factor
        sentiment = state.get('market_sentiment', {})
        sentiment_type = sentiment.get('sentiment', 'neutral')
        
        sentiment_risk_multiplier = {
            'bullish': 0.8,
            'active': 0.9,
            'neutral': 1.0,
            'bearish': 1.3
        }.get(sentiment_type, 1.0)
        
        market_risk = volatility * sentiment_risk_multiplier
        
        return min(market_risk, 1.0)

    def _calculate_liquidity_risk(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate liquidity risk."""
        
        pool_data = opportunity.get('pool_data', {})
        tvl = pool_data.get('tvl', 0)
        volume_24h = pool_data.get('volume_24h', 0)
        optimal_size = opportunity.get('optimal_size_usd', 0)
        
        if tvl <= 0:
            return 1.0
        
        # Risk from position size relative to pool size
        size_impact = optimal_size / tvl
        
        # Risk from low trading volume
        volume_risk = 0.5 if volume_24h < tvl * 0.01 else 0.2  # Should have >1% volume/TVL
        
        liquidity_risk = (size_impact * 2) + volume_risk
        
        return min(liquidity_risk, 1.0)

    def _calculate_contract_risk(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate smart contract risk."""
        
        pool_data = opportunity.get('pool_data', {})
        
        # Pool age factor (newer pools are riskier)
        # This would be calculated from actual pool creation time
        pool_age_risk = 0.3  # Placeholder
        
        # Pool size factor (smaller pools are riskier)
        tvl = pool_data.get('tvl', 0)
        size_risk = max(0.1, 1.0 - (tvl / 1_000_000))  # Lower risk for >$1M TVL
        
        contract_risk = (pool_age_risk + size_risk) / 2
        
        return min(contract_risk, 1.0)

    def _calculate_timing_risk(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate timing risk based on market conditions."""
        
        gas_price = state.get('gas_price', 0)
        congestion = state.get('network_congestion', 0.5)
        
        # High gas price increases timing risk
        gas_risk = min(gas_price / self.config.max_gas_price, 1.0)
        
        # High congestion increases timing risk
        congestion_risk = congestion
        
        timing_risk = (gas_risk + congestion_risk) / 2
        
        return timing_risk

    def _calculate_il_risk(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate impermanent loss risk."""
        
        action_type = opportunity.get('action_type')
        
        if action_type != 'ADD_LIQUIDITY':
            return 0.0  # No IL risk for non-LP actions
        
        il_risk_data = opportunity.get('impermanent_loss_risk', {})
        il_risk = il_risk_data.get('risk_score', 0.5)
        
        return il_risk

    # Helper methods

    def _matches_failure_pattern(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> bool:
        """Check if opportunity matches known failure patterns."""
        
        failure_patterns = state.get('failure_patterns', [])
        
        for failure in failure_patterns:
            if failure.get('risk_score', 0) > 0.7:  # High risk failures
                failed_action = failure.get('action_taken', {})
                
                if (failed_action.get('action_type') == opportunity.get('action_type') and
                    failed_action.get('pool_address') == opportunity.get('pool_address')):
                    return True
        
        return False

    def _fits_portfolio_risk_profile(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> bool:
        """Check if opportunity fits portfolio risk profile."""
        
        # Check total portfolio risk wouldn't exceed limits
        current_positions = state.get('active_positions', [])
        total_position_risk = sum(pos.get('risk_score', 0.5) for pos in current_positions)
        
        new_risk = opportunity.get('risk_assessment', {}).get('total_risk', 0.5)
        
        if (total_position_risk + new_risk) > self.max_portfolio_risk:
            return False
        
        # Check individual position risk
        if new_risk > self.max_individual_position_risk:
            return False
        
        return True

    def _calculate_portfolio_fit_score(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate how well opportunity fits current portfolio."""
        
        # Diversification benefit
        current_positions = state.get('active_positions', [])
        pool_addresses = [pos.get('pool_address') for pos in current_positions]
        
        # Prefer new pools for diversification
        is_new_pool = opportunity.get('pool_address') not in pool_addresses
        diversification_score = 0.8 if is_new_pool else 0.4
        
        # Portfolio balance (prefer actions that balance the portfolio)
        balance_score = self._calculate_portfolio_balance_score(opportunity, state)
        
        portfolio_fit = (diversification_score + balance_score) / 2
        
        return portfolio_fit

    def _calculate_timing_score(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate timing score for the decision."""
        
        # Gas price factor
        gas_price = state.get('gas_price', 0)
        gas_score = 1.0 - min(gas_price / self.config.max_gas_price, 1.0)
        
        # Network congestion factor
        congestion = state.get('network_congestion', 0.5)
        congestion_score = 1.0 - congestion
        
        # Market sentiment factor
        sentiment = state.get('market_sentiment', {})
        sentiment_confidence = sentiment.get('confidence', 0.5)
        
        timing_score = (gas_score + congestion_score + sentiment_confidence) / 3
        
        return timing_score

    def _calculate_optimal_amounts(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> Dict[str, float]:
        """Calculate optimal amounts for the action."""
        
        optimal_size = opportunity.get('optimal_size_usd', 0)
        action_type = opportunity.get('action_type')
        pool_data = opportunity.get('pool_data', {})
        
        if action_type == 'ADD_LIQUIDITY':
            # Calculate token amounts for LP
            return self._calculate_lp_amounts(optimal_size, pool_data)
        elif action_type == 'SWAP':
            # Calculate swap amounts
            return self._calculate_swap_amounts(optimal_size, pool_data)
        else:
            return {'amount_usd': optimal_size}

    def _calculate_slippage_tolerance(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate appropriate slippage tolerance."""
        
        base_slippage = self.config.max_slippage
        
        # Adjust based on market conditions
        volatility = opportunity.get('pool_data', {}).get('volatility', 0.3)
        congestion = state.get('network_congestion', 0.5)
        
        # Higher volatility and congestion require higher slippage tolerance
        adjusted_slippage = base_slippage * (1 + volatility + congestion)
        
        return min(adjusted_slippage, 0.05)  # Cap at 5%

    def _determine_gas_strategy(self, state: BrainState) -> str:
        """Determine gas strategy based on market conditions."""
        
        gas_price = state.get('gas_price', 0)
        congestion = state.get('network_congestion', 0.5)
        
        if gas_price > self.config.max_gas_price * 0.8 or congestion > 0.7:
            return "slow"  # Wait for better conditions
        elif gas_price < self.config.max_gas_price * 0.3 and congestion < 0.3:
            return "fast"  # Take advantage of good conditions
        else:
            return "standard"

    def _generate_decision_reasoning(
        self, 
        opportunity: Dict[str, Any], 
        execution_plan: ActionPlan, 
        confidence_score: float, 
        state: BrainState
    ) -> List[str]:
        """Generate human-readable decision reasoning."""
        
        reasoning = []
        
        # Opportunity selection reasoning
        opp_score = opportunity.get('score', {}).get('total_score', 0)
        reasoning.append(f"Selected {opportunity['action_type']} with opportunity score {opp_score:.3f}")
        
        # Risk assessment reasoning
        risk_assessment = opportunity.get('risk_assessment', {})
        total_risk = risk_assessment.get('total_risk', 0)
        reasoning.append(f"Total risk assessment: {total_risk:.3f} (acceptable)")
        
        # Key factors
        key_factors = []
        score_details = opportunity.get('score', {})
        
        if score_details.get('yield_score', 0) > 0.7:
            key_factors.append(f"High yield potential ({score_details.get('yield_score', 0):.2f})")
        
        if score_details.get('volume_score', 0) > 0.7:
            key_factors.append("Strong volume activity")
        
        if score_details.get('historical_score', 0) > 0.7:
            key_factors.append("Positive historical performance")
        
        if key_factors:
            reasoning.append(f"Key factors: {', '.join(key_factors)}")
        
        # Confidence reasoning
        reasoning.append(f"Final confidence: {confidence_score:.3f}")
        
        # Risk mitigations
        mitigations = risk_assessment.get('mitigations', [])
        if mitigations:
            reasoning.append(f"Risk mitigations: {', '.join(mitigations[:3])}")
        
        return reasoning

    def _format_selected_action(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Format the selected action for execution."""
        
        return {
            'action_type': opportunity['action_type'],
            'pool_address': opportunity['pool_address'],
            'opportunity_score': opportunity.get('score', {}).get('total_score', 0),
            'risk_score': opportunity.get('risk_assessment', {}).get('total_risk', 0),
            'expected_returns': opportunity.get('expected_returns', {}),
            'pool_metadata': {
                'tvl': opportunity.get('pool_data', {}).get('tvl', 0),
                'volume_24h': opportunity.get('pool_data', {}).get('volume_24h', 0),
                'apr': opportunity.get('pool_data', {}).get('apr', 0),
                'is_stable': opportunity.get('pool_data', {}).get('is_stable', False)
            }
        }

    # Placeholder helper methods (would be implemented with actual calculations)

    def _identify_risk_factors(
        self, 
        position_risk: float,
        market_risk: float, 
        liquidity_risk: float,
        contract_risk: float, 
        timing_risk: float, 
        il_risk: float
    ) -> List[str]:
        """Identify primary risk factors."""
        
        factors = []
        
        if position_risk > 0.5:
            factors.append("Large position size")
        if market_risk > 0.5:
            factors.append("Market volatility")
        if liquidity_risk > 0.5:
            factors.append("Low liquidity")
        if contract_risk > 0.5:
            factors.append("Smart contract risk")
        if timing_risk > 0.5:
            factors.append("Poor timing conditions")
        if il_risk > 0.5:
            factors.append("Impermanent loss risk")
        
        return factors

    def _suggest_risk_mitigations(
        self, 
        opportunity: Dict[str, Any], 
        risk_factors: List[str]
    ) -> List[str]:
        """Suggest risk mitigation strategies."""
        
        mitigations = []
        
        if "Large position size" in risk_factors:
            mitigations.append("Reduce position size")
        if "Market volatility" in risk_factors:
            mitigations.append("Use tighter stop losses")
        if "Low liquidity" in risk_factors:
            mitigations.append("Use smaller trade sizes")
        if "Poor timing conditions" in risk_factors:
            mitigations.append("Wait for better market conditions")
        if "Impermanent loss risk" in risk_factors:
            mitigations.append("Consider stable pair alternatives")
        
        return mitigations

    def _calculate_portfolio_balance_score(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate how well opportunity balances the portfolio."""
        
        # Placeholder - would implement actual portfolio balance analysis
        return 0.7

    def _calculate_lp_amounts(
        self, 
        size_usd: float, 
        pool_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate token amounts for LP position."""
        
        # Placeholder - would implement actual LP amount calculation
        return {
            'token0_amount': size_usd / 2,
            'token1_amount': size_usd / 2,
            'total_usd': size_usd
        }

    def _calculate_swap_amounts(
        self, 
        size_usd: float, 
        pool_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate amounts for swap."""
        
        # Placeholder
        return {
            'amount_in': size_usd,
            'expected_amount_out': size_usd * 0.997  # Assume 0.3% fee
        }

    def _calculate_deadline(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> int:
        """Calculate transaction deadline in minutes."""
        
        # Base deadline
        base_deadline = 15  # 15 minutes
        
        # Adjust based on market conditions
        congestion = state.get('network_congestion', 0.5)
        
        # Higher congestion = longer deadline
        adjusted_deadline = base_deadline * (1 + congestion)
        
        return int(min(adjusted_deadline, 60))  # Cap at 1 hour

    def _calculate_max_acceptable_loss(
        self, 
        opportunity: Dict[str, Any], 
        state: BrainState
    ) -> float:
        """Calculate maximum acceptable loss."""
        
        optimal_size = opportunity.get('optimal_size_usd', 0)
        risk_tolerance = self.config.risk_threshold
        
        return optimal_size * risk_tolerance

    def _calculate_stop_loss_threshold(self, opportunity: Dict[str, Any]) -> Optional[float]:
        """Calculate stop loss threshold."""
        
        action_type = opportunity.get('action_type')
        
        if action_type in ['ADD_LIQUIDITY', 'STAKE_LP']:
            # No immediate stop loss for LP positions
            return None
        else:
            # 5% stop loss for trading positions
            return 0.05

    def _calculate_monitor_duration(self, action_type: str) -> int:
        """Calculate monitoring duration in seconds."""
        
        if action_type == 'SWAP':
            return 300  # 5 minutes
        elif action_type in ['ADD_LIQUIDITY', 'REMOVE_LIQUIDITY']:
            return 600  # 10 minutes
        else:
            return 1800  # 30 minutes

    def _define_success_criteria(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Define success criteria for the action."""
        
        return {
            'transaction_confirmed': True,
            'slippage_within_tolerance': True,
            'gas_cost_reasonable': True,
            'expected_outcome_achieved': True
        }

    def _calculate_market_condition_modifier(self, state: BrainState) -> float:
        """Calculate confidence modifier based on market conditions."""
        
        sentiment = state.get('market_sentiment', {})
        sentiment_confidence = sentiment.get('confidence', 0.5)
        
        gas_price = state.get('gas_price', 0)
        gas_factor = 1.0 - min(gas_price / self.config.max_gas_price, 1.0)
        
        return (sentiment_confidence + gas_factor) / 2

    def _extract_token_addresses(self, pool_data: Dict[str, Any]) -> List[str]:
        """Extract token addresses from pool data."""
        
        # Placeholder - would extract actual token addresses
        return [
            pool_data.get('token0', ''),
            pool_data.get('token1', '')
        ]