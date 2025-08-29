"""
Aerodrome Voting Analysis System

Comprehensive analysis system for Aerodrome's veAERO voting mechanism:
- veAERO voting patterns and trends
- Bribe effectiveness tracking and ROI analysis  
- Gauge weight distribution analysis
- Emission flow monitoring
- Voter behavior analytics
- Strategic voting insights

Integrates with AerodromeClient for real-time voting data access.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from decimal import Decimal
import json
from statistics import mean, median

import structlog
from .aerodrome_client import AerodromeClient, AERODROME_CONTRACTS

logger = structlog.get_logger(__name__)


class VoteType(Enum):
    """Types of votes in Aerodrome"""
    GAUGE_WEIGHT = "gauge_weight"
    WHITELIST = "whitelist"
    PROPOSAL = "proposal"


class BribeType(Enum):
    """Types of bribes"""
    TOKEN_BRIBE = "token_bribe"
    LP_BRIBE = "lp_bribe"
    CUSTOM_BRIBE = "custom_bribe"


class VoterCategory(Enum):
    """Categories of voters based on veAERO holdings"""
    WHALE = "whale"          # >1M veAERO
    LARGE = "large"          # 100k-1M veAERO
    MEDIUM = "medium"        # 10k-100k veAERO
    SMALL = "small"          # 1k-10k veAERO
    MICRO = "micro"          # <1k veAERO


@dataclass
class VoterInfo:
    """Information about a veAERO voter"""
    address: str
    ve_aero_balance: float
    voting_power: float
    category: VoterCategory
    total_votes_cast: int
    avg_vote_amount: float
    favorite_pools: List[str] = field(default_factory=list)
    last_vote_timestamp: Optional[datetime] = None
    total_bribes_received: float = 0.0


@dataclass
class GaugeInfo:
    """Information about a gauge"""
    address: str
    pool_address: str
    pool_symbol: str
    total_weight: float
    relative_weight: float
    votes_count: int
    total_votes_amount: float
    emissions_per_epoch: float
    bribes_total: float
    bribes_per_vote_ratio: float
    is_active: bool = True


@dataclass
class BribeInfo:
    """Information about a bribe"""
    gauge_address: str
    bribe_address: str
    token_address: str
    token_symbol: str
    amount: float
    amount_usd: float
    epoch: int
    timestamp: datetime
    bribe_type: BribeType
    provider_address: str


@dataclass
class VotingRound:
    """Information about a voting round/epoch"""
    epoch: int
    start_timestamp: datetime
    end_timestamp: datetime
    total_ve_aero_voted: float
    total_unique_voters: int
    gauges_voted: Set[str]
    total_bribes_usd: float
    avg_bribe_per_vote: float
    top_voted_gauges: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class VotingAnalytics:
    """Comprehensive voting analytics"""
    timestamp: datetime
    epoch: int
    
    # Overall metrics
    total_ve_aero_supply: float
    total_voting_power_used: float
    participation_rate: float
    
    # Gauge metrics
    total_active_gauges: int
    gauges_with_votes: int
    gauge_competition_index: float  # Measure of vote distribution
    
    # Bribe metrics
    total_bribes_usd: float
    avg_bribe_effectiveness: float
    bribe_vote_correlation: float
    
    # Voter behavior
    whale_vote_share: float
    voter_concentration: float  # Gini coefficient of voting power
    new_voters_count: int
    
    # Trends
    vote_weight_trend: str
    bribe_trend: str
    participation_trend: str


class VotingPatternAnalyzer:
    """Analyzes voting patterns and trends"""
    
    def __init__(self, lookback_epochs: int = 10):
        self.lookback_epochs = lookback_epochs
    
    def calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = sum((i + 1) * val for i, val in enumerate(sorted_values))
        total_sum = sum(sorted_values)
        
        if total_sum == 0:
            return 0.0
        
        return (2 * cumsum) / (n * total_sum) - (n + 1) / n
    
    def calculate_herfindahl_index(self, market_shares: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        return sum(share ** 2 for share in market_shares)
    
    def detect_voting_coalitions(
        self,
        voting_history: Dict[str, List[Tuple[str, float]]]  # voter -> [(gauge, amount)]
    ) -> List[List[str]]:
        """Detect potential voting coalitions based on similar voting patterns"""
        coalitions = []
        processed_voters = set()
        
        voters = list(voting_history.keys())
        
        for i, voter1 in enumerate(voters):
            if voter1 in processed_voters:
                continue
            
            coalition = [voter1]
            voter1_gauges = set(gauge for gauge, _ in voting_history[voter1])
            
            for j, voter2 in enumerate(voters[i+1:], i+1):
                if voter2 in processed_voters:
                    continue
                
                voter2_gauges = set(gauge for gauge, _ in voting_history[voter2])
                
                # Calculate Jaccard similarity
                intersection = len(voter1_gauges & voter2_gauges)
                union = len(voter1_gauges | voter2_gauges)
                
                if union > 0:
                    similarity = intersection / union
                    if similarity >= 0.7:  # High similarity threshold
                        coalition.append(voter2)
                        processed_voters.add(voter2)
            
            if len(coalition) > 1:
                coalitions.append(coalition)
                for voter in coalition:
                    processed_voters.add(voter)
        
        return coalitions
    
    def analyze_bribe_effectiveness(
        self,
        bribes: List[BribeInfo],
        votes: Dict[str, float]  # gauge -> vote amount
    ) -> Dict[str, float]:
        """Analyze effectiveness of bribes in attracting votes"""
        effectiveness = {}
        
        bribe_totals = defaultdict(float)
        for bribe in bribes:
            bribe_totals[bribe.gauge_address] += bribe.amount_usd
        
        for gauge, bribe_amount in bribe_totals.items():
            vote_amount = votes.get(gauge, 0.0)
            if bribe_amount > 0:
                effectiveness[gauge] = vote_amount / bribe_amount  # Votes per dollar
            else:
                effectiveness[gauge] = 0.0
        
        return effectiveness


class VotingAnalyzer:
    """
    Comprehensive Aerodrome voting analysis system
    
    Provides detailed analysis of:
    - Voting patterns and trends
    - Bribe effectiveness
    - Gauge weight distribution
    - Emission flows
    - Voter behavior
    """
    
    def __init__(self, aerodrome_client: AerodromeClient):
        """
        Initialize voting analyzer
        
        Args:
            aerodrome_client: AerodromeClient instance
        """
        self.client = aerodrome_client
        self.pattern_analyzer = VotingPatternAnalyzer()
        
        # Data caches
        self.voters_cache: Dict[str, VoterInfo] = {}
        self.gauges_cache: Dict[str, GaugeInfo] = {}
        self.voting_rounds_cache: Dict[int, VotingRound] = {}
        self.bribes_cache: Dict[int, List[BribeInfo]] = defaultdict(list)
        
        # Analysis cache
        self.analytics_cache: Dict[int, VotingAnalytics] = {}
        
        logger.info("Voting analyzer initialized")
    
    # ===== DATA COLLECTION =====
    
    async def get_current_epoch(self) -> int:
        """Get current voting epoch"""
        try:
            # Call Voter contract to get current epoch
            current_epoch = await self.client._make_rpc_call(
                "eth_call",
                [{
                    "to": AERODROME_CONTRACTS["voter"],
                    "data": "0x76671808"  # epoch() function signature
                }, "latest"]
            )
            return int(current_epoch, 16)
        except Exception as e:
            logger.error("Failed to get current epoch", error=str(e))
            # Fallback: estimate based on time
            genesis_timestamp = 1693526400  # Approximate Aerodrome launch
            current_timestamp = datetime.now().timestamp()
            epoch_duration = 7 * 24 * 3600  # 7 days
            return int((current_timestamp - genesis_timestamp) / epoch_duration)
    
    async def get_voter_info(self, voter_address: str, use_cache: bool = True) -> VoterInfo:
        """
        Get comprehensive voter information
        
        Args:
            voter_address: Voter's address
            use_cache: Whether to use cached data
            
        Returns:
            VoterInfo object
        """
        if use_cache and voter_address in self.voters_cache:
            return self.voters_cache[voter_address]
        
        try:
            # Get veAERO balance
            ve_balance_data = await self.client._make_rpc_call(
                "eth_call",
                [{
                    "to": AERODROME_CONTRACTS["voter"],
                    "data": f"0x70a08231{voter_address[2:].zfill(64)}"  # balanceOf(address)
                }, "latest"]
            )
            ve_balance = int(ve_balance_data, 16) / 1e18
            
            # Determine voter category
            if ve_balance >= 1_000_000:
                category = VoterCategory.WHALE
            elif ve_balance >= 100_000:
                category = VoterCategory.LARGE
            elif ve_balance >= 10_000:
                category = VoterCategory.MEDIUM
            elif ve_balance >= 1_000:
                category = VoterCategory.SMALL
            else:
                category = VoterCategory.MICRO
            
            # Get voting history (simplified - would need event logs in reality)
            voting_history = await self._get_voter_history(voter_address)
            
            voter_info = VoterInfo(
                address=voter_address,
                ve_aero_balance=ve_balance,
                voting_power=ve_balance,  # Simplified - actual voting power has time decay
                category=category,
                total_votes_cast=len(voting_history),
                avg_vote_amount=mean([v["amount"] for v in voting_history]) if voting_history else 0.0,
                favorite_pools=self._get_favorite_pools(voting_history),
                last_vote_timestamp=max([v["timestamp"] for v in voting_history]) if voting_history else None
            )
            
            if use_cache:
                self.voters_cache[voter_address] = voter_info
            
            return voter_info
            
        except Exception as e:
            logger.error("Failed to get voter info", address=voter_address, error=str(e))
            raise
    
    async def get_gauge_info(self, gauge_address: str, epoch: Optional[int] = None) -> GaugeInfo:
        """
        Get comprehensive gauge information
        
        Args:
            gauge_address: Gauge contract address
            epoch: Specific epoch (current if None)
            
        Returns:
            GaugeInfo object
        """
        if epoch is None:
            epoch = await self.get_current_epoch()
        
        try:
            # Get gauge weight
            weight_data = await self.client._make_rpc_call(
                "eth_call",
                [{
                    "to": AERODROME_CONTRACTS["voter"],
                    "data": f"0x{self._encode_function_call('weights', [gauge_address])}"
                }, "latest"]
            )
            total_weight = int(weight_data, 16) / 1e18
            
            # Get pool address for this gauge
            pool_data = await self.client._make_rpc_call(
                "eth_call",
                [{
                    "to": gauge_address,
                    "data": "0x16f0115b"  # stakingToken() - returns pool address
                }, "latest"]
            )
            pool_address = "0x" + pool_data[26:]
            
            # Get pool info for symbol
            try:
                pool_info = await self.client.get_pool_info(pool_address)
                pool_symbol = f"{pool_info.token0.symbol}/{pool_info.token1.symbol}"
            except:
                pool_symbol = "Unknown Pool"
            
            # Get total voting weight for relative calculation
            total_voting_weight = await self._get_total_voting_weight(epoch)
            relative_weight = total_weight / total_voting_weight if total_voting_weight > 0 else 0.0
            
            # Get votes data
            votes_data = await self._get_gauge_votes(gauge_address, epoch)
            
            # Get bribe data
            bribe_data = await self._get_gauge_bribes(gauge_address, epoch)
            
            gauge_info = GaugeInfo(
                address=gauge_address,
                pool_address=pool_address,
                pool_symbol=pool_symbol,
                total_weight=total_weight,
                relative_weight=relative_weight,
                votes_count=votes_data["count"],
                total_votes_amount=votes_data["total_amount"],
                emissions_per_epoch=self._calculate_emissions(relative_weight),
                bribes_total=bribe_data["total_usd"],
                bribes_per_vote_ratio=bribe_data["total_usd"] / total_weight if total_weight > 0 else 0.0,
                is_active=total_weight > 0
            )
            
            return gauge_info
            
        except Exception as e:
            logger.error("Failed to get gauge info", address=gauge_address, error=str(e))
            raise
    
    async def get_epoch_bribes(self, epoch: int) -> List[BribeInfo]:
        """
        Get all bribes for a specific epoch
        
        Args:
            epoch: Epoch number
            
        Returns:
            List of BribeInfo objects
        """
        if epoch in self.bribes_cache:
            return self.bribes_cache[epoch]
        
        try:
            # This would typically involve reading event logs from bribe contracts
            # Simplified implementation - would need to iterate through all gauges
            # and their associated bribe contracts
            
            bribes = []
            
            # Get all active gauges for this epoch
            active_gauges = await self._get_active_gauges(epoch)
            
            for gauge_address in active_gauges:
                gauge_bribes = await self._get_gauge_bribes_detailed(gauge_address, epoch)
                bribes.extend(gauge_bribes)
            
            self.bribes_cache[epoch] = bribes
            return bribes
            
        except Exception as e:
            logger.error("Failed to get epoch bribes", epoch=epoch, error=str(e))
            return []
    
    # ===== ANALYSIS METHODS =====
    
    async def analyze_voting_round(self, epoch: int) -> VotingRound:
        """
        Analyze a complete voting round
        
        Args:
            epoch: Epoch number to analyze
            
        Returns:
            VotingRound object with analysis results
        """
        if epoch in self.voting_rounds_cache:
            return self.voting_rounds_cache[epoch]
        
        try:
            # Get epoch timing
            epoch_start = await self._get_epoch_start_time(epoch)
            epoch_end = epoch_start + timedelta(days=7)
            
            # Get all votes for this epoch
            epoch_votes = await self._get_epoch_votes(epoch)
            
            # Calculate metrics
            total_ve_voted = sum(vote["amount"] for vote in epoch_votes)
            unique_voters = len(set(vote["voter"] for vote in epoch_votes))
            
            # Get gauges that received votes
            gauges_voted = set(vote["gauge"] for vote in epoch_votes)
            
            # Calculate top voted gauges
            gauge_votes = defaultdict(float)
            for vote in epoch_votes:
                gauge_votes[vote["gauge"]] += vote["amount"]
            
            top_gauges = sorted(gauge_votes.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Get bribes for this epoch
            epoch_bribes = await self.get_epoch_bribes(epoch)
            total_bribes_usd = sum(bribe.amount_usd for bribe in epoch_bribes)
            
            avg_bribe_per_vote = total_bribes_usd / total_ve_voted if total_ve_voted > 0 else 0.0
            
            voting_round = VotingRound(
                epoch=epoch,
                start_timestamp=epoch_start,
                end_timestamp=epoch_end,
                total_ve_aero_voted=total_ve_voted,
                total_unique_voters=unique_voters,
                gauges_voted=gauges_voted,
                total_bribes_usd=total_bribes_usd,
                avg_bribe_per_vote=avg_bribe_per_vote,
                top_voted_gauges=top_gauges
            )
            
            self.voting_rounds_cache[epoch] = voting_round
            return voting_round
            
        except Exception as e:
            logger.error("Failed to analyze voting round", epoch=epoch, error=str(e))
            raise
    
    async def analyze_bribe_effectiveness(self, epoch: int) -> Dict[str, Any]:
        """
        Analyze bribe effectiveness for an epoch
        
        Args:
            epoch: Epoch to analyze
            
        Returns:
            Dictionary with bribe effectiveness metrics
        """
        try:
            bribes = await self.get_epoch_bribes(epoch)
            voting_round = await self.analyze_voting_round(epoch)
            
            # Group bribes by gauge
            gauge_bribes = defaultdict(list)
            for bribe in bribes:
                gauge_bribes[bribe.gauge_address].append(bribe)
            
            # Calculate effectiveness metrics
            effectiveness_data = {}
            
            for gauge_address, gauge_bribes_list in gauge_bribes.items():
                total_bribe_usd = sum(b.amount_usd for b in gauge_bribes_list)
                
                # Find votes for this gauge
                gauge_votes = dict(voting_round.top_voted_gauges).get(gauge_address, 0.0)
                
                # Calculate effectiveness ratios
                votes_per_dollar = gauge_votes / total_bribe_usd if total_bribe_usd > 0 else 0.0
                bribe_per_vote = total_bribe_usd / gauge_votes if gauge_votes > 0 else float('inf')
                
                effectiveness_data[gauge_address] = {
                    "total_bribe_usd": total_bribe_usd,
                    "total_votes": gauge_votes,
                    "votes_per_dollar": votes_per_dollar,
                    "bribe_per_vote": bribe_per_vote,
                    "bribe_count": len(gauge_bribes_list),
                    "roi": self._calculate_bribe_roi(gauge_votes, total_bribe_usd)
                }
            
            # Overall effectiveness metrics
            total_bribes = sum(b.amount_usd for b in bribes)
            total_votes = voting_round.total_ve_aero_voted
            
            analysis = {
                "epoch": epoch,
                "total_bribes_usd": total_bribes,
                "total_votes": total_votes,
                "overall_votes_per_dollar": total_votes / total_bribes if total_bribes > 0 else 0.0,
                "gauge_effectiveness": effectiveness_data,
                "top_effective_gauges": sorted(
                    effectiveness_data.items(),
                    key=lambda x: x[1]["votes_per_dollar"],
                    reverse=True
                )[:5]
            }
            
            return analysis
            
        except Exception as e:
            logger.error("Failed to analyze bribe effectiveness", epoch=epoch, error=str(e))
            return {}
    
    async def analyze_voter_behavior(self, epochs: List[int]) -> Dict[str, Any]:
        """
        Analyze voter behavior patterns across multiple epochs
        
        Args:
            epochs: List of epochs to analyze
            
        Returns:
            Dictionary with voter behavior analysis
        """
        try:
            all_voters = set()
            voting_patterns = defaultdict(list)
            voter_loyalty = defaultdict(set)
            
            # Collect voting data across epochs
            for epoch in epochs:
                epoch_votes = await self._get_epoch_votes(epoch)
                
                for vote in epoch_votes:
                    voter = vote["voter"]
                    gauge = vote["gauge"]
                    amount = vote["amount"]
                    
                    all_voters.add(voter)
                    voting_patterns[voter].append({
                        "epoch": epoch,
                        "gauge": gauge,
                        "amount": amount
                    })
                    voter_loyalty[voter].add(gauge)
            
            # Analyze voter categories
            voter_analysis = {}
            category_stats = defaultdict(list)
            
            for voter in all_voters:
                try:
                    voter_info = await self.get_voter_info(voter)
                    voter_votes = voting_patterns[voter]
                    
                    # Calculate consistency metrics
                    unique_gauges = len(voter_loyalty[voter])
                    total_votes = len(voter_votes)
                    consistency = unique_gauges / total_votes if total_votes > 0 else 0.0
                    
                    voter_analysis[voter] = {
                        "category": voter_info.category.value,
                        "ve_balance": voter_info.ve_aero_balance,
                        "total_votes": total_votes,
                        "unique_gauges": unique_gauges,
                        "consistency_score": 1.0 - consistency,  # Lower is more consistent
                        "avg_vote_size": mean([v["amount"] for v in voter_votes])
                    }
                    
                    category_stats[voter_info.category.value].append(voter_analysis[voter])
                
                except Exception as e:
                    logger.warning("Failed to analyze voter", voter=voter, error=str(e))
                    continue
            
            # Detect voting coalitions
            voter_gauge_map = {
                voter: [(v["gauge"], v["amount"]) for v in votes]
                for voter, votes in voting_patterns.items()
            }
            coalitions = self.pattern_analyzer.detect_voting_coalitions(voter_gauge_map)
            
            # Category-level analysis
            category_summary = {}
            for category, voters_data in category_stats.items():
                if voters_data:
                    category_summary[category] = {
                        "count": len(voters_data),
                        "avg_ve_balance": mean([v["ve_balance"] for v in voters_data]),
                        "avg_votes_cast": mean([v["total_votes"] for v in voters_data]),
                        "avg_consistency": mean([v["consistency_score"] for v in voters_data]),
                        "total_voting_power": sum([v["ve_balance"] for v in voters_data])
                    }
            
            return {
                "total_unique_voters": len(all_voters),
                "epochs_analyzed": epochs,
                "voter_analysis": voter_analysis,
                "category_summary": category_summary,
                "detected_coalitions": [{"size": len(coalition), "members": coalition} for coalition in coalitions],
                "voting_concentration": self._calculate_voting_concentration(voter_analysis)
            }
            
        except Exception as e:
            logger.error("Failed to analyze voter behavior", error=str(e))
            return {}
    
    async def generate_voting_analytics(self, epoch: int) -> VotingAnalytics:
        """
        Generate comprehensive voting analytics for an epoch
        
        Args:
            epoch: Epoch to analyze
            
        Returns:
            VotingAnalytics object
        """
        if epoch in self.analytics_cache:
            return self.analytics_cache[epoch]
        
        try:
            # Get basic voting data
            voting_round = await self.analyze_voting_round(epoch)
            bribe_analysis = await self.analyze_bribe_effectiveness(epoch)
            
            # Get total veAERO supply
            total_supply = await self._get_total_ve_supply(epoch)
            participation_rate = voting_round.total_ve_aero_voted / total_supply if total_supply > 0 else 0.0
            
            # Get all active gauges
            all_gauges = await self._get_active_gauges(epoch)
            gauges_with_votes = len(voting_round.gauges_voted)
            
            # Calculate gauge competition index (inverse of HHI)
            gauge_weights = [weight for _, weight in voting_round.top_voted_gauges]
            total_weight = sum(gauge_weights)
            if total_weight > 0:
                market_shares = [w / total_weight for w in gauge_weights]
                hhi = self.pattern_analyzer.calculate_herfindahl_index(market_shares)
                competition_index = 1.0 - hhi  # Higher = more competitive
            else:
                competition_index = 0.0
            
            # Voter behavior analysis
            voter_behavior = await self.analyze_voter_behavior([epoch])
            whale_voters = [v for v in voter_behavior["voter_analysis"].values() 
                          if v["category"] == VoterCategory.WHALE.value]
            whale_vote_share = sum(v["ve_balance"] for v in whale_voters) / total_supply if total_supply > 0 else 0.0
            
            # Calculate trends (compare with previous epoch)
            trends = await self._calculate_trends(epoch)
            
            analytics = VotingAnalytics(
                timestamp=datetime.now(),
                epoch=epoch,
                total_ve_aero_supply=total_supply,
                total_voting_power_used=voting_round.total_ve_aero_voted,
                participation_rate=participation_rate,
                total_active_gauges=len(all_gauges),
                gauges_with_votes=gauges_with_votes,
                gauge_competition_index=competition_index,
                total_bribes_usd=voting_round.total_bribes_usd,
                avg_bribe_effectiveness=mean([g["votes_per_dollar"] for g in bribe_analysis["gauge_effectiveness"].values()]) if bribe_analysis["gauge_effectiveness"] else 0.0,
                bribe_vote_correlation=await self._calculate_bribe_correlation(epoch),
                whale_vote_share=whale_vote_share,
                voter_concentration=voter_behavior.get("voting_concentration", 0.0),
                new_voters_count=await self._count_new_voters(epoch),
                vote_weight_trend=trends["weight_trend"],
                bribe_trend=trends["bribe_trend"],
                participation_trend=trends["participation_trend"]
            )
            
            self.analytics_cache[epoch] = analytics
            return analytics
            
        except Exception as e:
            logger.error("Failed to generate voting analytics", epoch=epoch, error=str(e))
            raise
    
    # ===== HELPER METHODS =====
    
    async def _get_voter_history(self, voter_address: str) -> List[Dict[str, Any]]:
        """Get voting history for a voter (simplified)"""
        # In reality, this would parse event logs from the Voter contract
        return []
    
    def _get_favorite_pools(self, voting_history: List[Dict[str, Any]]) -> List[str]:
        """Get voter's favorite pools based on history"""
        if not voting_history:
            return []
        
        pool_votes = defaultdict(int)
        for vote in voting_history:
            pool_votes[vote.get("pool", "")] += 1
        
        return sorted(pool_votes.keys(), key=lambda x: pool_votes[x], reverse=True)[:5]
    
    async def _get_total_voting_weight(self, epoch: int) -> float:
        """Get total voting weight for an epoch"""
        try:
            total_weight_data = await self.client._make_rpc_call(
                "eth_call",
                [{
                    "to": AERODROME_CONTRACTS["voter"],
                    "data": "0x96c82e57"  # totalWeight() function signature
                }, "latest"]
            )
            return int(total_weight_data, 16) / 1e18
        except:
            return 0.0
    
    async def _get_gauge_votes(self, gauge_address: str, epoch: int) -> Dict[str, Any]:
        """Get vote data for a specific gauge"""
        # Simplified - would need to parse event logs
        return {"count": 0, "total_amount": 0.0}
    
    async def _get_gauge_bribes(self, gauge_address: str, epoch: int) -> Dict[str, float]:
        """Get bribe data for a gauge"""
        # Simplified - would need to query bribe contracts
        return {"total_usd": 0.0}
    
    async def _get_gauge_bribes_detailed(self, gauge_address: str, epoch: int) -> List[BribeInfo]:
        """Get detailed bribe information for a gauge"""
        # Simplified implementation
        return []
    
    def _calculate_emissions(self, relative_weight: float) -> float:
        """Calculate AERO emissions for a gauge based on relative weight"""
        # Simplified calculation - actual emission calculation is more complex
        weekly_emissions = 1_000_000  # Example weekly AERO emissions
        return weekly_emissions * relative_weight
    
    async def _get_active_gauges(self, epoch: int) -> List[str]:
        """Get all active gauges for an epoch"""
        # Simplified - would need to query gauge factory
        return []
    
    async def _get_epoch_start_time(self, epoch: int) -> datetime:
        """Get start time for an epoch"""
        genesis_timestamp = 1693526400  # Approximate Aerodrome launch
        epoch_duration = 7 * 24 * 3600  # 7 days
        return datetime.fromtimestamp(genesis_timestamp + epoch * epoch_duration)
    
    async def _get_epoch_votes(self, epoch: int) -> List[Dict[str, Any]]:
        """Get all votes for an epoch"""
        # Simplified - would parse event logs
        return []
    
    def _calculate_bribe_roi(self, votes_received: float, bribe_amount_usd: float) -> float:
        """Calculate ROI for bribes based on emissions gained"""
        if bribe_amount_usd == 0:
            return 0.0
        
        # Simplified ROI calculation
        # In reality, would need to consider AERO price and emission value
        aero_price = 1.0  # Placeholder
        emissions_value = votes_received * 0.001 * aero_price  # Simplified
        
        return (emissions_value - bribe_amount_usd) / bribe_amount_usd
    
    def _calculate_voting_concentration(self, voter_analysis: Dict[str, Any]) -> float:
        """Calculate voting power concentration using Gini coefficient"""
        ve_balances = [v["ve_balance"] for v in voter_analysis.values()]
        return self.pattern_analyzer.calculate_gini_coefficient(ve_balances)
    
    async def _get_total_ve_supply(self, epoch: int) -> float:
        """Get total veAERO supply for an epoch"""
        try:
            total_supply_data = await self.client._make_rpc_call(
                "eth_call",
                [{
                    "to": AERODROME_CONTRACTS["voter"],
                    "data": "0x18160ddd"  # totalSupply() function signature
                }, "latest"]
            )
            return int(total_supply_data, 16) / 1e18
        except:
            return 0.0
    
    async def _calculate_trends(self, current_epoch: int) -> Dict[str, str]:
        """Calculate trends by comparing with previous epoch"""
        if current_epoch == 0:
            return {"weight_trend": "unknown", "bribe_trend": "unknown", "participation_trend": "unknown"}
        
        try:
            current_round = await self.analyze_voting_round(current_epoch)
            previous_round = await self.analyze_voting_round(current_epoch - 1)
            
            # Weight trend
            weight_change = (current_round.total_ve_aero_voted - previous_round.total_ve_aero_voted) / previous_round.total_ve_aero_voted
            weight_trend = "increasing" if weight_change > 0.05 else "decreasing" if weight_change < -0.05 else "stable"
            
            # Bribe trend
            bribe_change = (current_round.total_bribes_usd - previous_round.total_bribes_usd) / previous_round.total_bribes_usd if previous_round.total_bribes_usd > 0 else 0
            bribe_trend = "increasing" if bribe_change > 0.05 else "decreasing" if bribe_change < -0.05 else "stable"
            
            # Participation trend
            participation_change = (current_round.total_unique_voters - previous_round.total_unique_voters) / previous_round.total_unique_voters
            participation_trend = "increasing" if participation_change > 0.05 else "decreasing" if participation_change < -0.05 else "stable"
            
            return {
                "weight_trend": weight_trend,
                "bribe_trend": bribe_trend,
                "participation_trend": participation_trend
            }
            
        except:
            return {"weight_trend": "unknown", "bribe_trend": "unknown", "participation_trend": "unknown"}
    
    async def _calculate_bribe_correlation(self, epoch: int) -> float:
        """Calculate correlation between bribes and votes received"""
        try:
            bribe_analysis = await self.analyze_bribe_effectiveness(epoch)
            
            bribes = []
            votes = []
            
            for gauge_data in bribe_analysis["gauge_effectiveness"].values():
                bribes.append(gauge_data["total_bribe_usd"])
                votes.append(gauge_data["total_votes"])
            
            if len(bribes) < 2:
                return 0.0
            
            # Calculate Pearson correlation coefficient
            n = len(bribes)
            sum_bribes = sum(bribes)
            sum_votes = sum(votes)
            sum_bribes_sq = sum(b**2 for b in bribes)
            sum_votes_sq = sum(v**2 for v in votes)
            sum_product = sum(bribes[i] * votes[i] for i in range(n))
            
            numerator = n * sum_product - sum_bribes * sum_votes
            denominator = ((n * sum_bribes_sq - sum_bribes**2) * (n * sum_votes_sq - sum_votes**2))**0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except:
            return 0.0
    
    async def _count_new_voters(self, epoch: int) -> int:
        """Count new voters in this epoch compared to previous"""
        if epoch == 0:
            return 0
        
        try:
            current_votes = await self._get_epoch_votes(epoch)
            previous_votes = await self._get_epoch_votes(epoch - 1)
            
            current_voters = set(vote["voter"] for vote in current_votes)
            previous_voters = set(vote["voter"] for vote in previous_votes)
            
            return len(current_voters - previous_voters)
            
        except:
            return 0
    
    def _encode_function_call(self, function_name: str, params: List[str]) -> str:
        """Encode function call for RPC (simplified)"""
        # This is a simplified version - would use web3.py's encode_abi in practice
        return "00000000"  # Placeholder


# ===== EXAMPLE USAGE =====

async def example_voting_analysis():
    """Example usage of VotingAnalyzer"""
    from .aerodrome_client import AerodromeClient
    
    # Initialize client and analyzer
    quicknode_url = "https://your-quicknode-endpoint.quiknode.pro/your-api-key/"
    
    async with AerodromeClient(quicknode_url) as client:
        analyzer = VotingAnalyzer(client)
        
        # Get current epoch
        current_epoch = await analyzer.get_current_epoch()
        print(f"Current epoch: {current_epoch}")
        
        # Analyze current voting round
        try:
            voting_round = await analyzer.analyze_voting_round(current_epoch)
            print(f"\nVoting Round Analysis (Epoch {current_epoch}):")
            print(f"Total veAERO voted: {voting_round.total_ve_aero_voted:,.2f}")
            print(f"Unique voters: {voting_round.total_unique_voters}")
            print(f"Gauges with votes: {len(voting_round.gauges_voted)}")
            print(f"Total bribes: ${voting_round.total_bribes_usd:,.2f}")
            
            # Top voted gauges
            print(f"\nTop Voted Gauges:")
            for i, (gauge, votes) in enumerate(voting_round.top_voted_gauges[:5], 1):
                try:
                    gauge_info = await analyzer.get_gauge_info(gauge)
                    print(f"{i}. {gauge_info.pool_symbol}: {votes:,.2f} votes")
                except:
                    print(f"{i}. {gauge[:10]}...: {votes:,.2f} votes")
            
        except Exception as e:
            print(f"Failed to analyze voting round: {e}")
        
        # Analyze bribe effectiveness
        try:
            bribe_effectiveness = await analyzer.analyze_bribe_effectiveness(current_epoch)
            print(f"\nBribe Effectiveness:")
            print(f"Total bribes: ${bribe_effectiveness['total_bribes_usd']:,.2f}")
            print(f"Overall votes per dollar: {bribe_effectiveness['overall_votes_per_dollar']:.4f}")
            
            print(f"\nTop Effective Bribed Gauges:")
            for i, (gauge, data) in enumerate(bribe_effectiveness['top_effective_gauges'][:3], 1):
                print(f"{i}. Gauge {gauge[:10]}...: {data['votes_per_dollar']:.4f} votes/$")
            
        except Exception as e:
            print(f"Failed to analyze bribe effectiveness: {e}")
        
        # Generate comprehensive analytics
        try:
            analytics = await analyzer.generate_voting_analytics(current_epoch)
            print(f"\nComprehensive Analytics (Epoch {current_epoch}):")
            print(f"Participation rate: {analytics.participation_rate:.1%}")
            print(f"Gauge competition index: {analytics.gauge_competition_index:.3f}")
            print(f"Whale vote share: {analytics.whale_vote_share:.1%}")
            print(f"Voter concentration (Gini): {analytics.voter_concentration:.3f}")
            print(f"Vote weight trend: {analytics.vote_weight_trend}")
            print(f"Bribe trend: {analytics.bribe_trend}")
            
        except Exception as e:
            print(f"Failed to generate analytics: {e}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_voting_analysis())