"""
Contract Addresses - Aerodrome protocol addresses on Base network

Contains all contract addresses for the Aerodrome Finance protocol
deployed on Base mainnet.
"""

from typing import Dict

# Core Aerodrome Protocol Contracts
AERODROME_ROUTER = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
AERODROME_FACTORY = "0x420DD381b31aEf6683db96b3eff9d5cCD6c22c6"
AERODROME_VOTER = "0x16613524e02ad97eDfeF371bC883F2F5d6C480A5"
AERODROME_ROUTER_V2 = "0xAaAaAAddcC5aB1Af48c30Eb3678f5612f84F6c02"

# Token Addresses on Base
BASE_TOKENS: Dict[str, str] = {
    # Native and wrapped tokens
    "ETH": "0x0000000000000000000000000000000000000000",  # Native ETH
    "WETH": "0x4200000000000000000000000000000000000006",  # Wrapped ETH
    
    # Stablecoins
    "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USD Coin
    "USDT": "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2",  # Tether USD
    "DAI": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",   # Dai Stablecoin
    "USDbC": "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA", # USD Base Coin
    
    # Major tokens
    "WBTC": "0xc1CBa3fCea344f92078ef4938e4Ac0E4Dc4b2C0B",  # Wrapped Bitcoin
    "AERO": "0x940181a94A35A4569E4529A3CDfB74e38FD98631",  # Aerodrome token
    
    # DeFi tokens
    "cbETH": "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22", # Coinbase Wrapped Staked ETH
    "wstETH": "0xc1CBa3fCea344f92078ef4938e4Ac0E4Dc4b2C0B", # Wrapped Staked ETH
    "rETH": "0xB6fe221Fe9EeF5aBa221c348bA20A1Bf5e73624c",  # Rocket Pool ETH
    
    # Bridged tokens  
    "LINK": "0x88Fb150BDc53A65fe94Dea0c9BA0a6dAf8C6e196",  # Chainlink
    "UNI": "0xd3C52b2b8Cf5a8e6A2A4Eb9f1D2E4DbCabD2e2A8",   # Uniswap
    "COMP": "0x9e1028F5F1D5eDE59748FFceE5532509976840E0",  # Compound
    
    # Base ecosystem tokens
    "BSWAP": "0x78a087d713Be963Bf307b18F2Ff8122EF9A63ae9", # BaseSwap
    "PRIME": "0xfA980cEd6895AC314E7dE34Ef1bFAE90a5AdD21b", # Parallel Prime
}

# Token decimals mapping
TOKEN_DECIMALS: Dict[str, int] = {
    "ETH": 18,
    "WETH": 18,
    "USDC": 6,
    "USDT": 6, 
    "DAI": 18,
    "USDbC": 6,
    "WBTC": 8,
    "AERO": 18,
    "cbETH": 18,
    "wstETH": 18,
    "rETH": 18,
    "LINK": 18,
    "UNI": 18,
    "COMP": 18,
    "BSWAP": 18,
    "PRIME": 18,
}

# Common pool addresses (will be populated dynamically)
COMMON_POOLS: Dict[str, str] = {
    # Stable pools
    "USDC-USDbC": "",  # Will be fetched from factory
    "USDC-USDT": "",
    "DAI-USDC": "",
    
    # Volatile pools
    "WETH-USDC": "",
    "WETH-AERO": "",
    "AERO-USDC": "",
    "WBTC-WETH": "",
    "cbETH-WETH": "",
}

# Fee tiers
FEE_TIERS = {
    "STABLE": 5,      # 0.05% for stable pairs
    "VOLATILE": 30,   # 0.30% for volatile pairs
    "EXOTIC": 100,    # 1.00% for exotic pairs
}

# Network information
NETWORK_INFO = {
    "name": "Base",
    "chain_id": 8453,
    "rpc_url": "https://mainnet.base.org",
    "explorer_url": "https://basescan.org",
    "multicall_address": "0xcA11bde05977b3631167028862bE2a173976CA11",
}

# Utility functions
def get_token_address(symbol: str) -> str:
    """Get token address by symbol"""
    return BASE_TOKENS.get(symbol.upper(), "")

def get_token_decimals(symbol: str) -> int:
    """Get token decimals by symbol"""
    return TOKEN_DECIMALS.get(symbol.upper(), 18)

def is_stable_pair(token_a: str, token_b: str) -> bool:
    """Check if token pair should use stable pool"""
    stablecoins = {"USDC", "USDT", "DAI", "USDbC"}
    return token_a.upper() in stablecoins and token_b.upper() in stablecoins

def get_explorer_url(address: str, type: str = "address") -> str:
    """Get explorer URL for address or transaction"""
    base_url = NETWORK_INFO["explorer_url"]
    if type == "tx":
        return f"{base_url}/tx/{address}"
    else:
        return f"{base_url}/address/{address}"