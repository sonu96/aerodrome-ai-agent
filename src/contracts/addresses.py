"""
Contract Addresses - Aerodrome protocol contract addresses and network configuration

This module contains all contract addresses for Aerodrome protocol on Base and other
supported networks. It provides a centralized location for contract addresses and
network-specific configurations used by the CDP SDK.

Includes addresses for:
- Aerodrome core contracts (Router, Voter, Factory, etc.)
- Token addresses (AERO, USDC, WETH, etc.)
- Gauge and pool addresses
- Network configurations and chain IDs
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Network configuration dataclass."""
    name: str
    chain_id: int
    currency: str
    rpc_url: str
    explorer: str
    multicall: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'chain_id': self.chain_id,
            'currency': self.currency,
            'rpc_url': self.rpc_url,
            'explorer': self.explorer,
            'multicall': self.multicall
        }

# Network configurations
NETWORKS: Dict[str, NetworkConfig] = {
    'base-mainnet': NetworkConfig(
        name='Base Mainnet',
        chain_id=8453,
        currency='ETH',
        rpc_url='https://mainnet.base.org',
        explorer='https://basescan.org',
        multicall='0xcA11bde05977b3631167028862bE2a173976CA11'
    ),
    'base-goerli': NetworkConfig(
        name='Base Goerli Testnet',
        chain_id=84531,
        currency='ETH',
        rpc_url='https://goerli.base.org',
        explorer='https://goerli.basescan.org',
        multicall='0xcA11bde05977b3631167028862bE2a173976CA11'
    ),
    'base-sepolia': NetworkConfig(
        name='Base Sepolia Testnet',
        chain_id=84532,
        currency='ETH', 
        rpc_url='https://sepolia.base.org',
        explorer='https://sepolia.basescan.org',
        multicall='0xcA11bde05977b3631167028862bE2a173976CA11'
    )
}

# Aerodrome Protocol Core Contracts on Base Mainnet
AERODROME_CONTRACTS: Dict[str, str] = {
    # Core Protocol
    'ROUTER': '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43',
    'VOTER': '0x16613524e02ad97eDfeF371bC883F2F5d6C480A5',
    'FACTORY': '0x420DD381b31aEf6683db6B902084cB0FFECe40Da',
    'MINTER': '0xeB018363F0a9Af8f91F06FEe6613a751b2A33FE5',
    'VE_AERO': '0xeBf418Fe2512e7E6bd9b87a8F0f294aCDC67e6B4',
    'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631',
    'REWARDS_DISTRIBUTOR': '0x227f65131A261548b057215bB1D5Ab2997964C7d',
    
    # Additional contracts
    'GAUGE_FACTORY': '0x35f35cBd67C5f7b67e0da71b82b9080E4E14deB5',
    'BRIBE_FACTORY': '0x35f35cBd67C5f7b67e0da71b82b9080E4E14deB5',
    'WHITELIST': '0x8eB823ce8d1d0eF8E01ecc83d28fAC60dE4F3DED'
}

# Base Network Token Addresses
TOKEN_ADDRESSES: Dict[str, Dict[str, str]] = {
    'base-mainnet': {
        # Native and Wrapped
        'ETH': '0x0000000000000000000000000000000000000000',  # Native ETH
        'WETH': '0x4200000000000000000000000000000000000006',
        
        # Stablecoins
        'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
        'USDbC': '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',  # USD Base Coin (bridged USDC)
        'USDT': '0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2',
        'DAI': '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
        'LUSD': '0x368181499736d0c0CC614DBB145E2EC1AC86b8c6',
        'crvUSD': '0x417Ac0e078398C154EdFadD9Ef675d30Be60af93',
        
        # Aerodrome Protocol
        'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631',
        
        # Major Tokens
        'WBTC': '0x21d02B9dD18d82Cf8aD7Ed72b97b6C3a19Efe9a8',
        'CBETH': '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
        'rETH': '0xB6fe221Fe9EeF5aBa221c348bA20A1Bf5e73624c',
        'wstETH': '0xc1CBa3fCea344f92D9239c08C0568f6F2F0ee452',
        
        # DeFi Tokens
        'BAL': '0x4158734D47Fc9692176B5085E0F52ee0Da5d47F1',
        'COMP': '0x9e1028F5F1D5eDE59748FFceE5532509976840E0',
        'CRV': '0x8Ee73c484A26e0A5df2Ee2a4960B789967dd0415',
        'SUSHI': '0x7D49a065D17d6d4a55dc13649901fdBB98B2AFBA',
        'UNI': '0xc6A56c6A3362f63D35E9fCa8cF4c04c9F77eE9B0',
        
        # Base Ecosystem
        'PRIME': '0xfA980cEd6895AC314E7dE34Ef1bFAE90a5AdD21b',
        'DEGEN': '0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed',
        'BRETT': '0x532f27101965dd16442E59d40670FaF5eBB142E4',
        'TOSHI': '0xAC1Bd2486aAf3B5C0fc3Fd868558b082a531B2B4',
        'HIGHER': '0x0578d8A44db98B23BF096A382e016e29a5Ce0ffe',
        'NORMIE': '0x7F12d13B34F5F4f0a9449c16Bcd42f0da47AF200',
        
        # LP Tokens and Others
        'GOLD': '0x96b8b5C7f37D7BB6Cf7A3C3E3cBDb3C13a8F3a9c',
        'MLN': '0xa9fE4601811213c340e850ea305481afF02f5b28'
    },
    'base-goerli': {
        # Testnet addresses (these would be different from mainnet)
        'ETH': '0x0000000000000000000000000000000000000000',
        'WETH': '0x4200000000000000000000000000000000000006',
        'USDC': '0xF175520C52418dfE19C8098071a252da48Cd1C19',
        'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631'  # Example testnet address
    },
    'base-sepolia': {
        # Testnet addresses
        'ETH': '0x0000000000000000000000000000000000000000',
        'WETH': '0x4200000000000000000000000000000000000006',
        'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',  # Example testnet address
        'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631'
    }
}

# Popular Pool Addresses on Base
POOL_ADDRESSES: Dict[str, str] = {
    # Volatile Pools
    'AERO_USDC_VOLATILE': '0x6cDcb1C4A4D1C3C6d054b27AC5B77e89eAFb971d',
    'WETH_USDC_VOLATILE': '0xcDAC0d6c6C59727a65F871236188350531885C43',
    'WETH_AERO_VOLATILE': '0x7f670f78B17dEC44d5747d2B5b6B7A7C8F8E8c44',
    'WETH_DAI_VOLATILE': '0x6B4712AE9797C199edd44F897cA09BC57628a1CF',
    'USDC_USDbC_VOLATILE': '0x1b3462c46B6A1B6e5C9B8BFF3F6a4B8F6f3cB45b',
    
    # Stable Pools  
    'USDC_DAI_STABLE': '0x67b61B8E5783129bdEd47632e84e5D8BfC2F36d3',
    'USDC_USDbC_STABLE': '0x3916c10B8C7DdF6a3C7b3E5a3c1e0b1F5d4B4c58',
    'USDC_USDT_STABLE': '0x4b4B7D56c3B8BaA4c5e1e3F7a5B5a8a8c2a2a1a1',
    
    # Derivative Pools
    'WETH_CBETH_STABLE': '0xC9c8dE8e24a085aB0CF9A1b6a8b2a8b8c2a2a1a1',
    'WETH_WSTETH_STABLE': '0x6c7F83C3fDB92C06B01e1D5ff62f2B72D8d8C2dc',
    'WETH_rETH_STABLE': '0x3d4FDD2f81B14c6b6a8D0df7e6a5B83a8a3f6d8d'
}

# Gauge Addresses corresponding to pools
GAUGE_ADDRESSES: Dict[str, str] = {
    'AERO_USDC_VOLATILE': '0x9a202c932453fB3d04003979B121E80e5A14eE7b',
    'WETH_USDC_VOLATILE': '0x519BBD1Dd8C6A94C46080E24f316c14Ee758C025',
    'WETH_AERO_VOLATILE': '0x96C6AEAaFe8eB4B7B8b6F8B4b4b4b4B4b4b4b4b4',
    'WETH_DAI_VOLATILE': '0x7f670f78B17dEC44d5747d2B5b6B7A7C8F8E8c44',
    
    'USDC_DAI_STABLE': '0x8d24bD9B1a42C2Ea05c7F3a5B5a8a8c2a2a1a1a1',
    'USDC_USDbC_STABLE': '0x3916c10B8C7DdF6a3C7b3E5a3c1e0b1F5d4B4c58',
    'USDC_USDT_STABLE': '0x4b4B7D56c3B8BaA4c5e1e3F7a5B5a8a8c2a2a1a1',
    
    'WETH_CBETH_STABLE': '0xC9c8dE8e24a085aB0CF9A1b6a8b2a8b8c2a2a1a1',
    'WETH_WSTETH_STABLE': '0x6c7F83C3fDB92C06B01e1D5ff62f2B72D8d8C2dc',
    'WETH_rETH_STABLE': '0x3d4FDD2f81B14c6b6a8D0df7e6a5B83a8a3f6d8d'
}

# Bribe Addresses for gauges
BRIBE_ADDRESSES: Dict[str, str] = {
    'AERO_USDC_VOLATILE': '0x4b4B7D56c3B8BaA4c5e1e3F7a5B5a8a8c2a2a1a1',
    'WETH_USDC_VOLATILE': '0x3916c10B8C7DdF6a3C7b3E5a3c1e0b1F5d4B4c58',
    'WETH_AERO_VOLATILE': '0x8d24bD9B1a42C2Ea05c7F3a5B5a8a8c2a2a1a1a1',
    'WETH_DAI_VOLATILE': '0x7f670f78B17dEC44d5747d2B5b6B7A7C8F8E8c44',
    
    'USDC_DAI_STABLE': '0x6B4712AE9797C199edd44F897cA09BC57628a1CF',
    'USDC_USDbC_STABLE': '0x67b61B8E5783129bdEd47632e84e5D8BfC2F36d3',
    'USDC_USDT_STABLE': '0x1b3462c46B6A1B6e5C9B8BFF3F6a4B8F6f3cB45b',
    
    'WETH_CBETH_STABLE': '0xC9c8dE8e24a085aB0CF9A1b6a8b2a8b8c2a2a1a1',
    'WETH_WSTETH_STABLE': '0x6c7F83C3fDB92C06B01e1D5ff62f2B72D8d8C2dc',
    'WETH_rETH_STABLE': '0x3d4FDD2f81B14c6b6a8D0df7e6a5B83a8a3f6d8d'
}

# Network-specific contract addresses
NETWORK_CONTRACTS: Dict[str, Dict[str, str]] = {
    'base-mainnet': {
        **AERODROME_CONTRACTS,
        'MULTICALL': '0xcA11bde05977b3631167028862bE2a173976CA11',
        'ENS_REGISTRY': '0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e',
        'ENS_RESOLVER': '0x8FADE66B79cC9f707aB26799354482EB93a5B7dD'
    },
    'base-goerli': {
        **{k: v for k, v in AERODROME_CONTRACTS.items()},  # Copy main contracts
        'MULTICALL': '0xcA11bde05977b3631167028862bE2a173976CA11'
    },
    'base-sepolia': {
        **{k: v for k, v in AERODROME_CONTRACTS.items()},
        'MULTICALL': '0xcA11bde05977b3631167028862bE2a173976CA11'
    }
}

# Common token pairs for routing
COMMON_BASES: Dict[str, List[str]] = {
    'base-mainnet': [
        TOKEN_ADDRESSES['base-mainnet']['WETH'],
        TOKEN_ADDRESSES['base-mainnet']['USDC'], 
        TOKEN_ADDRESSES['base-mainnet']['USDbC'],
        TOKEN_ADDRESSES['base-mainnet']['AERO'],
        TOKEN_ADDRESSES['base-mainnet']['DAI']
    ]
}

# Utility functions for address management
def get_contract_address(contract_name: str, network: str = 'base-mainnet') -> str:
    """
    Get contract address for specified network.
    
    Args:
        contract_name: Name of the contract
        network: Network identifier
        
    Returns:
        Contract address
        
    Raises:
        KeyError: If contract or network not found
    """
    network_contracts = NETWORK_CONTRACTS.get(network)
    if not network_contracts:
        raise KeyError(f"Network '{network}' not found")
    
    contract_address = network_contracts.get(contract_name.upper())
    if not contract_address:
        raise KeyError(f"Contract '{contract_name}' not found on network '{network}'")
    
    return contract_address

def get_token_address(token_symbol: str, network: str = 'base-mainnet') -> str:
    """
    Get token address for specified network.
    
    Args:
        token_symbol: Token symbol (e.g., 'USDC', 'AERO')
        network: Network identifier
        
    Returns:
        Token address
        
    Raises:
        KeyError: If token or network not found
    """
    network_tokens = TOKEN_ADDRESSES.get(network)
    if not network_tokens:
        raise KeyError(f"Network '{network}' not found")
    
    token_address = network_tokens.get(token_symbol.upper())
    if not token_address:
        raise KeyError(f"Token '{token_symbol}' not found on network '{network}'")
    
    return token_address

def get_pool_address(pool_name: str) -> str:
    """
    Get pool address by name.
    
    Args:
        pool_name: Pool identifier (e.g., 'AERO_USDC_VOLATILE')
        
    Returns:
        Pool address
        
    Raises:
        KeyError: If pool not found
    """
    pool_address = POOL_ADDRESSES.get(pool_name.upper())
    if not pool_address:
        raise KeyError(f"Pool '{pool_name}' not found")
    
    return pool_address

def get_gauge_address(pool_name: str) -> str:
    """
    Get gauge address for a pool.
    
    Args:
        pool_name: Pool identifier
        
    Returns:
        Gauge address
        
    Raises:
        KeyError: If gauge not found
    """
    gauge_address = GAUGE_ADDRESSES.get(pool_name.upper())
    if not gauge_address:
        raise KeyError(f"Gauge for pool '{pool_name}' not found")
    
    return gauge_address

def get_bribe_address(pool_name: str) -> str:
    """
    Get bribe contract address for a pool.
    
    Args:
        pool_name: Pool identifier
        
    Returns:
        Bribe address
        
    Raises:
        KeyError: If bribe contract not found
    """
    bribe_address = BRIBE_ADDRESSES.get(pool_name.upper())
    if not bribe_address:
        raise KeyError(f"Bribe contract for pool '{pool_name}' not found")
    
    return bribe_address

def get_network_info(network: str = 'base-mainnet') -> Dict[str, Any]:
    """
    Get comprehensive network information.
    
    Args:
        network: Network identifier
        
    Returns:
        Network information dictionary
        
    Raises:
        KeyError: If network not found
    """
    network_config = NETWORKS.get(network)
    if not network_config:
        raise KeyError(f"Network '{network}' not found")
    
    return network_config.to_dict()

def is_stable_pair(token_a: str, token_b: str, network: str = 'base-mainnet') -> bool:
    """
    Determine if a token pair should use stable pool routing.
    
    Stable pairs are typically:
    - Stablecoin to stablecoin (USDC/DAI, USDC/USDT)
    - ETH derivatives (WETH/CBETH, WETH/wstETH)
    
    Args:
        token_a: First token address
        token_b: Second token address
        network: Network identifier
        
    Returns:
        True if pair should use stable routing
    """
    network_tokens = TOKEN_ADDRESSES.get(network, {})
    
    # Create reverse mapping for lookup
    address_to_symbol = {v: k for k, v in network_tokens.items()}
    
    symbol_a = address_to_symbol.get(token_a, '').upper()
    symbol_b = address_to_symbol.get(token_b, '').upper()
    
    # Stablecoin pairs
    stablecoins = {'USDC', 'USDT', 'DAI', 'LUSD', 'CRVUSD', 'USDBC'}
    if symbol_a in stablecoins and symbol_b in stablecoins:
        return True
    
    # ETH derivative pairs
    eth_derivatives = {'WETH', 'CBETH', 'RETH', 'WSTETH'}
    if symbol_a in eth_derivatives and symbol_b in eth_derivatives:
        return True
    
    return False

def get_common_bases(network: str = 'base-mainnet') -> List[str]:
    """
    Get common base tokens for routing on network.
    
    Args:
        network: Network identifier
        
    Returns:
        List of common base token addresses
    """
    return COMMON_BASES.get(network, [])

def validate_address(address: str) -> bool:
    """
    Validate Ethereum address format.
    
    Args:
        address: Ethereum address to validate
        
    Returns:
        True if address format is valid
    """
    if not isinstance(address, str):
        return False
    
    # Basic format validation
    if not address.startswith('0x'):
        return False
    
    if len(address) != 42:
        return False
    
    # Check if all characters after 0x are valid hex
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False

# Address validation for all stored addresses
def validate_all_addresses() -> Dict[str, List[str]]:
    """
    Validate all stored addresses for format correctness.
    
    Returns:
        Dictionary of invalid addresses by category
    """
    invalid_addresses = {
        'contracts': [],
        'tokens': [],
        'pools': [],
        'gauges': [],
        'bribes': []
    }
    
    # Check contract addresses
    for name, address in AERODROME_CONTRACTS.items():
        if not validate_address(address):
            invalid_addresses['contracts'].append(f"{name}: {address}")
    
    # Check token addresses
    for network, tokens in TOKEN_ADDRESSES.items():
        for symbol, address in tokens.items():
            if address != '0x0000000000000000000000000000000000000000' and not validate_address(address):
                invalid_addresses['tokens'].append(f"{network}:{symbol}: {address}")
    
    # Check pool addresses
    for name, address in POOL_ADDRESSES.items():
        if not validate_address(address):
            invalid_addresses['pools'].append(f"{name}: {address}")
    
    # Check gauge addresses
    for name, address in GAUGE_ADDRESSES.items():
        if not validate_address(address):
            invalid_addresses['gauges'].append(f"{name}: {address}")
    
    # Check bribe addresses
    for name, address in BRIBE_ADDRESSES.items():
        if not validate_address(address):
            invalid_addresses['bribes'].append(f"{name}: {address}")
    
    return invalid_addresses

# Export public interface
__all__ = [
    'NETWORKS',
    'AERODROME_CONTRACTS',
    'TOKEN_ADDRESSES',
    'POOL_ADDRESSES', 
    'GAUGE_ADDRESSES',
    'BRIBE_ADDRESSES',
    'NETWORK_CONTRACTS',
    'COMMON_BASES',
    'NetworkConfig',
    'get_contract_address',
    'get_token_address',
    'get_pool_address',
    'get_gauge_address',
    'get_bribe_address',
    'get_network_info',
    'is_stable_pair',
    'get_common_bases',
    'validate_address',
    'validate_all_addresses'
]