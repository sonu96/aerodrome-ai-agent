"""
Comprehensive Aerodrome Protocol Example Usage

This example demonstrates how to use all three main components:
1. AerodromeClient - Core protocol operations
2. PoolMonitor - Real-time pool monitoring
3. VotingAnalyzer - Voting analysis and insights

Run this example to see the full capabilities of the Aerodrome protocol client.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.protocol import (
    AerodromeClient,
    PoolMonitor, 
    VotingAnalyzer,
    MonitoringConfig,
    PoolAlert,
    AlertType
)

# QuickNode configuration
QUICKNODE_URL = "https://your-quicknode-endpoint.quiknode.pro/your-api-key/"
QUICKNODE_API_KEY = "your-api-key"  # Optional if included in URL


class AerodromeExample:
    """Comprehensive example of Aerodrome protocol usage"""
    
    def __init__(self):
        self.client = None
        self.monitor = None
        self.analyzer = None
    
    async def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing Aerodrome Protocol Client...")
        
        # Initialize main client
        self.client = AerodromeClient(
            quicknode_url=QUICKNODE_URL,
            quicknode_api_key=QUICKNODE_API_KEY,
            rate_limit_calls=100,
            rate_limit_window=60,
            enable_websocket=True
        )
        
        await self.client.connect()
        
        # Initialize monitoring with custom config
        monitor_config = MonitoringConfig(
            update_interval=60,  # 1 minute updates
            tvl_change_threshold=0.05,  # 5% TVL change alerts
            volume_spike_multiplier=2.0,  # 2x volume spike alerts
            liquidity_drop_threshold=0.10,  # 10% liquidity drop alerts
            apr_change_threshold=0.10,  # 10% APR change alerts
        )
        
        self.monitor = PoolMonitor(self.client, monitor_config)
        
        # Initialize voting analyzer
        self.analyzer = VotingAnalyzer(self.client)
        
        print("✅ All components initialized successfully!")
    
    async def demonstrate_basic_operations(self):
        """Demonstrate basic protocol operations"""
        print("\n" + "="*60)
        print("📊 BASIC PROTOCOL OPERATIONS")
        print("="*60)
        
        try:
            # Get network info
            network_info = await self.client.get_network_info()
            print(f"🌐 Network: Base Mainnet (Chain ID: {network_info['chain_id']})")
            print(f"📦 Current Block: {network_info['block_number']:,}")
            print(f"⛽ Gas Price: {network_info['gas_price'] / 1e9:.2f} gwei")
            
            # Get AERO token info
            aero_info = await self.client.get_token_info(
                "0x940181a94A35A4569E4529A3CDfB74e38FD98631"  # AERO token
            )
            print(f"\n💰 AERO Token:")
            print(f"   Price: ${aero_info.price_usd:.4f}")
            print(f"   Market Cap: ${aero_info.market_cap:,.0f}" if aero_info.market_cap else "   Market Cap: N/A")
            print(f"   24h Volume: ${aero_info.volume_24h:,.0f}" if aero_info.volume_24h else "   24h Volume: N/A")
            
            # Search for popular tokens
            print(f"\n🔍 Searching for popular tokens...")
            usdc_tokens = await self.client.search_tokens("USDC", limit=3)
            for i, token in enumerate(usdc_tokens, 1):
                print(f"   {i}. {token.symbol} - ${token.price_usd:.6f}" if token.price_usd else f"   {i}. {token.symbol}")
            
        except Exception as e:
            print(f"❌ Error in basic operations: {e}")
    
    async def demonstrate_pool_operations(self):
        """Demonstrate pool-related operations"""
        print("\n" + "="*60)
        print("🏊 POOL OPERATIONS")
        print("="*60)
        
        try:
            # Search for high TVL pools
            print("🔍 Finding high TVL pools...")
            high_tvl_pools = await self.client.search_pools(
                min_tvl=1_000_000,  # $1M minimum TVL
                sort_by="tvl",
                limit=10
            )
            
            print(f"📋 Top {len(high_tvl_pools)} High TVL Pools:")
            for i, pool in enumerate(high_tvl_pools, 1):
                print(f"   {i}. {pool.token0.symbol}/{pool.token1.symbol}")
                print(f"      TVL: ${pool.tvl_usd:,.0f} | APR: {pool.apr:.2f}% | Volume: ${pool.volume_24h:,.0f}")
            
            if high_tvl_pools:
                # Get detailed analytics for the top pool
                top_pool = high_tvl_pools[0]
                print(f"\n📈 Detailed Analytics for {top_pool.token0.symbol}/{top_pool.token1.symbol}:")
                
                analytics = await self.client.get_pool_analytics(top_pool.address, "24h")
                print(f"   Pool Type: {top_pool.pool_type.value}")
                print(f"   Fee Tier: {top_pool.fee:.3f}%")
                print(f"   24h Fees: ${top_pool.fees_24h:,.2f}")
                print(f"   Token0 Reserve: {top_pool.reserves['token0']:,.2f}")
                print(f"   Token1 Reserve: {top_pool.reserves['token1']:,.2f}")
                
                # Demonstrate swap quote
                print(f"\n💱 Swap Quote Example:")
                try:
                    quote = await self.client.get_swap_quote(
                        token_in=top_pool.token0.address,
                        token_out=top_pool.token1.address,
                        amount_in="1000000",  # 1 token (assuming 6 decimals)
                        slippage_bps=50  # 0.5% slippage
                    )
                    
                    print(f"   Input: {int(quote.amount_in) / 1e6:.2f} {top_pool.token0.symbol}")
                    print(f"   Output: {int(quote.amount_out) / 1e6:.2f} {top_pool.token1.symbol}")
                    print(f"   Price Impact: {quote.price_impact:.3f}%")
                    print(f"   Gas Estimate: {quote.gas_estimate:,}")
                    
                except Exception as e:
                    print(f"   ⚠️ Swap quote failed: {e}")
        
        except Exception as e:
            print(f"❌ Error in pool operations: {e}")
    
    async def demonstrate_pool_monitoring(self):
        """Demonstrate pool monitoring capabilities"""
        print("\n" + "="*60)
        print("📡 POOL MONITORING")
        print("="*60)
        
        try:
            # Set up alert handlers
            def alert_handler(alert: PoolAlert):
                emoji_map = {
                    AlertType.TVL_CHANGE: "💰",
                    AlertType.VOLUME_SPIKE: "📈", 
                    AlertType.LIQUIDITY_DROP: "📉",
                    AlertType.FEE_ANOMALY: "💸",
                    AlertType.APR_CHANGE: "📊",
                    AlertType.IL_RISK: "⚠️"
                }
                
                emoji = emoji_map.get(alert.alert_type, "🚨")
                print(f"{emoji} ALERT: {alert.message}")
                print(f"   Pool: {alert.pool_address[:10]}...")
                print(f"   Severity: {alert.severity.upper()}")
                print(f"   Current: {alert.current_value:.2f}")
                if alert.previous_value:
                    print(f"   Previous: {alert.previous_value:.2f}")
            
            def metrics_handler(pool_address: str, metrics):
                # Only print every 5th update to avoid spam
                if hasattr(metrics_handler, 'count'):
                    metrics_handler.count += 1
                else:
                    metrics_handler.count = 1
                
                if metrics_handler.count % 5 == 0:
                    print(f"📊 Pool {pool_address[:10]}... | TVL: ${metrics.tvl_usd:,.0f} | Volume: ${metrics.volume_24h:,.0f}")
            
            # Register callbacks
            self.monitor.add_alert_callback(alert_handler)
            self.monitor.add_metrics_callback(metrics_handler)
            
            # Find pools to monitor
            pools_to_monitor = await self.client.search_pools(min_tvl=500_000, limit=3)
            
            if pools_to_monitor:
                print(f"🔍 Starting monitoring for {len(pools_to_monitor)} pools...")
                
                # Start monitoring
                for pool in pools_to_monitor:
                    await self.monitor.start_monitoring(pool.address)
                    print(f"   ✅ Monitoring {pool.token0.symbol}/{pool.token1.symbol}")
                
                # Monitor for a short time
                print(f"\n⏱️ Monitoring for 2 minutes... (updates every {self.monitor.config.update_interval}s)")
                await asyncio.sleep(120)  # Monitor for 2 minutes
                
                # Show monitoring results
                print(f"\n📋 Monitoring Summary:")
                for pool in pools_to_monitor:
                    status = self.monitor.get_pool_status(pool.address)
                    analysis = self.monitor.get_pool_analysis(pool.address)
                    
                    print(f"\n   Pool: {pool.token0.symbol}/{pool.token1.symbol}")
                    print(f"   Status: {status['status']}")
                    print(f"   Metrics Collected: {status['metrics_count']}")
                    print(f"   Alerts (24h): {status['alerts_24h']}")
                    
                    if analysis and 'avg_tvl' in analysis:
                        print(f"   Avg TVL: ${analysis['avg_tvl']:,.0f}")
                        print(f"   TVL Trend: {analysis.get('tvl_trend', 'unknown')}")
                
                # Stop monitoring
                await self.monitor.stop_all_monitoring()
                print(f"\n✅ Monitoring stopped for all pools")
            
            else:
                print("⚠️ No pools found for monitoring")
        
        except Exception as e:
            print(f"❌ Error in pool monitoring: {e}")
    
    async def demonstrate_voting_analysis(self):
        """Demonstrate voting analysis capabilities"""
        print("\n" + "="*60)
        print("🗳️ VOTING ANALYSIS")
        print("="*60)
        
        try:
            # Get current epoch
            current_epoch = await self.analyzer.get_current_epoch()
            print(f"📅 Current Epoch: {current_epoch}")
            
            # Analyze voting round (might fail if no real data)
            try:
                print(f"\n🔍 Analyzing voting round for epoch {current_epoch}...")
                voting_round = await self.analyzer.analyze_voting_round(current_epoch)
                
                print(f"📊 Voting Round Summary:")
                print(f"   Total veAERO Voted: {voting_round.total_ve_aero_voted:,.2f}")
                print(f"   Unique Voters: {voting_round.total_unique_voters:,}")
                print(f"   Gauges with Votes: {len(voting_round.gauges_voted)}")
                print(f"   Total Bribes: ${voting_round.total_bribes_usd:,.2f}")
                print(f"   Avg Bribe/Vote: ${voting_round.avg_bribe_per_vote:.4f}")
                
                if voting_round.top_voted_gauges:
                    print(f"\n🏆 Top Voted Gauges:")
                    for i, (gauge, votes) in enumerate(voting_round.top_voted_gauges[:5], 1):
                        print(f"   {i}. {gauge[:10]}...: {votes:,.2f} votes")
            
            except Exception as e:
                print(f"⚠️ Voting round analysis failed (likely no data): {e}")
            
            # Try to analyze bribe effectiveness
            try:
                print(f"\n💰 Analyzing bribe effectiveness...")
                bribe_analysis = await self.analyzer.analyze_bribe_effectiveness(current_epoch)
                
                if bribe_analysis:
                    print(f"📈 Bribe Effectiveness Summary:")
                    print(f"   Total Bribes: ${bribe_analysis.get('total_bribes_usd', 0):,.2f}")
                    print(f"   Total Votes: {bribe_analysis.get('total_votes', 0):,.2f}")
                    print(f"   Overall Effectiveness: {bribe_analysis.get('overall_votes_per_dollar', 0):.4f} votes/$")
                    
                    top_effective = bribe_analysis.get('top_effective_gauges', [])
                    if top_effective:
                        print(f"\n🎯 Most Effective Bribed Gauges:")
                        for i, (gauge, data) in enumerate(top_effective[:3], 1):
                            print(f"   {i}. {gauge[:10]}...: {data.get('votes_per_dollar', 0):.4f} votes/$")
                else:
                    print("   ℹ️ No bribe data available for current epoch")
            
            except Exception as e:
                print(f"⚠️ Bribe analysis failed (likely no data): {e}")
            
            # Generate comprehensive analytics
            try:
                print(f"\n📋 Generating comprehensive analytics...")
                analytics = await self.analyzer.generate_voting_analytics(current_epoch)
                
                print(f"🔬 Comprehensive Analytics (Epoch {current_epoch}):")
                print(f"   Total veAERO Supply: {analytics.total_ve_aero_supply:,.2f}")
                print(f"   Participation Rate: {analytics.participation_rate:.1%}")
                print(f"   Active Gauges: {analytics.total_active_gauges}")
                print(f"   Gauges with Votes: {analytics.gauges_with_votes}")
                print(f"   Competition Index: {analytics.gauge_competition_index:.3f}")
                print(f"   Whale Vote Share: {analytics.whale_vote_share:.1%}")
                print(f"   Voter Concentration: {analytics.voter_concentration:.3f}")
                print(f"   Vote Trend: {analytics.vote_weight_trend}")
                print(f"   Bribe Trend: {analytics.bribe_trend}")
            
            except Exception as e:
                print(f"⚠️ Analytics generation failed (likely no data): {e}")
            
            # Example voter analysis (might fail with no data)
            try:
                print(f"\n👥 Analyzing voter behavior...")
                voter_behavior = await self.analyzer.analyze_voter_behavior([current_epoch])
                
                if voter_behavior:
                    print(f"🧑‍🤝‍🧑 Voter Behavior Summary:")
                    print(f"   Unique Voters: {voter_behavior.get('total_unique_voters', 0)}")
                    
                    category_summary = voter_behavior.get('category_summary', {})
                    for category, data in category_summary.items():
                        print(f"   {category.title()}: {data.get('count', 0)} voters, Avg veAERO: {data.get('avg_ve_balance', 0):,.2f}")
                    
                    coalitions = voter_behavior.get('detected_coalitions', [])
                    if coalitions:
                        print(f"   Detected {len(coalitions)} voting coalitions")
                        for i, coalition in enumerate(coalitions[:3], 1):
                            print(f"     Coalition {i}: {coalition['size']} members")
                else:
                    print("   ℹ️ No voter behavior data available")
            
            except Exception as e:
                print(f"⚠️ Voter behavior analysis failed (likely no data): {e}")
        
        except Exception as e:
            print(f"❌ Error in voting analysis: {e}")
    
    async def demonstrate_realtime_features(self):
        """Demonstrate real-time WebSocket features"""
        print("\n" + "="*60)
        print("🌐 REAL-TIME FEATURES")
        print("="*60)
        
        if not self.client.enable_websocket:
            print("⚠️ WebSocket disabled, skipping real-time features")
            return
        
        try:
            # Get some pools for price monitoring
            pools = await self.client.search_pools(min_tvl=100_000, limit=2)
            
            if not pools:
                print("⚠️ No pools found for real-time monitoring")
                return
            
            # Set up price update callback
            price_updates = []
            def price_update_handler(data):
                price_updates.append(data)
                print(f"💱 Price Update: {data}")
            
            # Set up pool update callback  
            pool_updates = []
            def pool_update_handler(data):
                pool_updates.append(data)
                print(f"🏊 Pool Update: {data}")
            
            # Subscribe to price updates for pool tokens
            token_addresses = []
            for pool in pools:
                token_addresses.extend([pool.token0.address, pool.token1.address])
            
            print(f"🔔 Subscribing to price updates for {len(set(token_addresses))} tokens...")
            
            # Note: These would fail without actual WebSocket endpoints
            try:
                await asyncio.wait_for(
                    self.client.subscribe_to_price_updates(
                        list(set(token_addresses)),
                        price_update_handler
                    ),
                    timeout=30
                )
            except asyncio.TimeoutError:
                print("   ⏱️ Price subscription timeout (expected without real WebSocket)")
            except Exception as e:
                print(f"   ⚠️ Price subscription failed: {e}")
            
            # Subscribe to pool updates
            print(f"🔔 Subscribing to pool updates...")
            try:
                await asyncio.wait_for(
                    self.client.subscribe_to_pool_updates(
                        pools[0].address,
                        pool_update_handler
                    ),
                    timeout=30
                )
            except asyncio.TimeoutError:
                print("   ⏱️ Pool subscription timeout (expected without real WebSocket)")
            except Exception as e:
                print(f"   ⚠️ Pool subscription failed: {e}")
            
            print(f"📊 Received {len(price_updates)} price updates and {len(pool_updates)} pool updates")
        
        except Exception as e:
            print(f"❌ Error in real-time features: {e}")
    
    async def cleanup(self):
        """Clean up all resources"""
        print("\n" + "="*60)
        print("🧹 CLEANUP")
        print("="*60)
        
        try:
            if self.monitor:
                await self.monitor.stop_all_monitoring()
                print("✅ Stopped all monitoring")
            
            if self.client:
                await self.client.disconnect()
                print("✅ Disconnected client")
            
            print("🎉 Cleanup completed successfully!")
        
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    async def run_full_example(self):
        """Run the complete example"""
        print("🚀 AERODROME PROTOCOL COMPREHENSIVE EXAMPLE")
        print("=" * 80)
        print("This example demonstrates all features of the Aerodrome protocol client.")
        print("Note: Some features require real QuickNode endpoint and may show limited data.")
        print("=" * 80)
        
        try:
            await self.initialize()
            await self.demonstrate_basic_operations()
            await self.demonstrate_pool_operations()
            await self.demonstrate_pool_monitoring()
            await self.demonstrate_voting_analysis()
            await self.demonstrate_realtime_features()
            
        except Exception as e:
            print(f"❌ Example failed: {e}")
        
        finally:
            await self.cleanup()


async def main():
    """Main example function"""
    example = AerodromeExample()
    await example.run_full_example()


if __name__ == "__main__":
    # Make sure to replace QUICKNODE_URL with your actual endpoint
    if "your-quicknode-endpoint" in QUICKNODE_URL:
        print("⚠️ Please update QUICKNODE_URL with your actual QuickNode endpoint!")
        print("Get one at: https://www.quicknode.com/")
        print("Make sure to enable the Aerodrome addon (1051) for full functionality.")
    else:
        asyncio.run(main())