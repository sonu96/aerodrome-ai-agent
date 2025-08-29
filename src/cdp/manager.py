"""
CDP Manager - Main CDP SDK manager class with wallet initialization

This module provides the central CDP manager that handles:
- CDP client initialization
- Wallet management (development and MPC production wallets)
- Network configuration
- Secure wallet data management

Uses CDP SDK exclusively for all blockchain operations.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:
    from cdp_sdk import CDPClient, Wallet, SmartContract, readContract
    from cdp_sdk.wallet import WalletConfig
except ImportError:
    # Fallback for development - these would be the actual CDP SDK imports
    CDPClient = None
    Wallet = None
    SmartContract = None
    readContract = None
    WalletConfig = None

from .errors import CDPError, WalletInitializationError
from ..contracts.addresses import NETWORKS, get_network_info


class CDPManager:
    """
    Centralized CDP SDK manager for all blockchain operations.
    
    This class handles:
    - CDP client initialization and configuration
    - Wallet initialization with support for both dev and MPC wallets  
    - Network management and chain ID resolution
    - Secure storage of wallet credentials
    """
    
    def __init__(self, network: str = 'base-mainnet'):
        """
        Initialize CDP manager with specified network.
        
        Args:
            network: Network identifier (default: 'base-mainnet')
            
        Raises:
            CDPError: If CDP SDK is not available or configuration fails
            WalletInitializationError: If wallet initialization fails
        """
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Validate CDP SDK availability
        if CDPClient is None:
            raise CDPError("CDP SDK not available. Install with: pip install cdp-sdk")
        
        # Network configuration
        self.network = network
        self.chain_id = self._get_chain_id(network)
        
        # Initialize CDP client
        self.client = self._initialize_client()
        
        # Initialize wallet
        self.wallet = self._initialize_wallet(network)
        
        self.logger.info(f"CDP Manager initialized for network: {network}")
    
    def _initialize_client(self) -> 'CDPClient':
        """
        Initialize CDP client with API credentials.
        
        Returns:
            CDPClient: Configured CDP client
            
        Raises:
            CDPError: If API credentials are missing or invalid
        """
        api_key_id = os.getenv('CDP_API_KEY_ID')
        api_key_secret = os.getenv('CDP_API_KEY_SECRET')
        
        if not api_key_id or not api_key_secret:
            raise CDPError(
                "CDP API credentials not found. Set CDP_API_KEY_ID and CDP_API_KEY_SECRET environment variables."
            )
        
        try:
            client = CDPClient(
                api_key_id=api_key_id,
                api_key_secret=api_key_secret
            )
            return client
        except Exception as e:
            raise CDPError(f"Failed to initialize CDP client: {str(e)}")
    
    def _initialize_wallet(self, network: str) -> 'Wallet':
        """
        Initialize CDP wallet with MPC security for production.
        
        Args:
            network: Network identifier
            
        Returns:
            Wallet: Initialized CDP wallet
            
        Raises:
            WalletInitializationError: If wallet initialization fails
        """
        try:
            # Determine if running in production
            is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
            
            # Configure wallet
            wallet_config = WalletConfig(
                network_id=network,
                wallet_secret=os.getenv('CDP_WALLET_SECRET'),
                # MPC configuration for production
                use_mpc=is_production,
                server_signer_url=os.getenv('SERVER_SIGNER_URL') if is_production else None
            )
            
            # Create or load existing wallet
            wallet_data = os.getenv('CDP_WALLET_DATA')
            if wallet_data:
                # Load existing wallet
                self.logger.info("Loading existing CDP wallet")
                wallet = Wallet.import_wallet(wallet_data, wallet_config)
            else:
                # Create new wallet
                self.logger.info("Creating new CDP wallet")
                wallet = Wallet.create(wallet_config)
                
                # Save wallet data securely
                wallet_export = wallet.export()
                self._save_wallet_data(wallet_export)
            
            self.logger.info(f"Wallet initialized: {wallet.address}")
            return wallet
            
        except Exception as e:
            raise WalletInitializationError(f"Failed to initialize wallet: {str(e)}")
    
    def _save_wallet_data(self, wallet_data: str) -> None:
        """
        Save wallet data securely.
        
        In production, this should integrate with secure key management systems
        like AWS KMS, Azure Key Vault, or HashiCorp Vault.
        
        Args:
            wallet_data: Exported wallet data to save
        """
        # For development - save to environment variable
        # In production, use secure key management
        if os.getenv('ENVIRONMENT', '').lower() != 'production':
            self.logger.warning("Saving wallet data to environment - use secure storage in production")
            # In real implementation, would save to secure storage
        else:
            # Production: integrate with secure key management
            self._save_to_secure_storage(wallet_data)
    
    def _save_to_secure_storage(self, wallet_data: str) -> None:
        """
        Save wallet data to secure storage system.
        
        This method should be implemented to integrate with your chosen
        secure storage solution (AWS KMS, Azure Key Vault, etc.).
        
        Args:
            wallet_data: Encrypted wallet data to store
        """
        # TODO: Implement secure storage integration
        # Examples:
        # - AWS KMS: encrypt and store in Parameter Store
        # - Azure Key Vault: store as secret
        # - HashiCorp Vault: store in KV store
        self.logger.info("Saving wallet data to secure storage")
        pass
    
    def _get_chain_id(self, network: str) -> int:
        """
        Get chain ID for network.
        
        Args:
            network: Network identifier
            
        Returns:
            Chain ID integer
            
        Raises:
            CDPError: If network is not supported
        """
        network_config = NETWORKS.get(network)
        if not network_config:
            raise CDPError(f"Unsupported network: {network}")
        
        return network_config.chain_id
    
    @property
    def wallet_address(self) -> str:
        """Get wallet address."""
        return self.wallet.address if self.wallet else None
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv('ENVIRONMENT', '').lower() == 'production'
    
    @property
    def uses_mpc(self) -> bool:
        """Check if wallet uses MPC security."""
        return self.is_production and hasattr(self.wallet, 'use_mpc') and self.wallet.use_mpc
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get comprehensive network information.
        
        Returns:
            Dictionary containing network details
        """
        try:
            network_info = get_network_info(self.network)
            network_info.update({
                'wallet_address': self.wallet_address,
                'is_production': self.is_production,
                'uses_mpc': self.uses_mpc
            })
            return network_info
        except Exception as e:
            self.logger.warning(f"Could not get network info: {str(e)}")
            return {
                'network': self.network,
                'chain_id': self.chain_id,
                'wallet_address': self.wallet_address,
                'is_production': self.is_production,
                'uses_mpc': self.uses_mpc
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on CDP connections.
        
        Returns:
            Health check status and details
        """
        health_status = {
            'cdp_client': False,
            'wallet': False,
            'network_connection': False,
            'errors': []
        }
        
        # Check CDP client
        try:
            if self.client:
                health_status['cdp_client'] = True
        except Exception as e:
            health_status['errors'].append(f"CDP client error: {str(e)}")
        
        # Check wallet
        try:
            if self.wallet and self.wallet.address:
                health_status['wallet'] = True
        except Exception as e:
            health_status['errors'].append(f"Wallet error: {str(e)}")
        
        # Check network connection
        try:
            # Simple read operation to test network connectivity
            if readContract:
                # Test with a simple contract read (ETH balance check)
                balance = await readContract({
                    'network_id': self.network,
                    'contract_address': '0x0000000000000000000000000000000000000000',
                    'method': 'balanceOf',
                    'args': {'account': self.wallet_address}
                })
                health_status['network_connection'] = True
        except Exception as e:
            health_status['errors'].append(f"Network connection error: {str(e)}")
        
        health_status['overall'] = all([
            health_status['cdp_client'],
            health_status['wallet'],
            health_status['network_connection']
        ])
        
        return health_status


class MPCWallet:
    """
    MPC (2-of-2) wallet operations for production environments.
    
    This class handles Multi-Party Computation wallets which provide
    enhanced security through distributed key management.
    """
    
    def __init__(self):
        """Initialize MPC wallet configuration."""
        self.logger = logging.getLogger(__name__)
        self.server_signer_url = os.getenv('SERVER_SIGNER_URL')
        self.wallet = self._init_mpc_wallet()
    
    def _init_mpc_wallet(self) -> 'Wallet':
        """
        Initialize MPC wallet with server signer.
        
        Returns:
            MPC-enabled Wallet instance
            
        Raises:
            WalletInitializationError: If MPC wallet creation fails
        """
        if not self.server_signer_url:
            raise WalletInitializationError("SERVER_SIGNER_URL required for MPC wallet")
        
        try:
            config = {
                'network_id': 'base-mainnet',
                'use_mpc': True,
                'server_signer': {
                    'url': self.server_signer_url,
                    'api_key': os.getenv('SERVER_SIGNER_API_KEY')
                }
            }
            
            wallet = Wallet.create_mpc_wallet(config)
            self.logger.info(f"MPC wallet created: {wallet.address}")
            return wallet
            
        except Exception as e:
            raise WalletInitializationError(f"Failed to create MPC wallet: {str(e)}")
    
    async def sign_transaction(self, tx_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign transaction with MPC wallet.
        
        The server signer participates automatically in the signing process.
        
        Args:
            tx_params: Transaction parameters
            
        Returns:
            Dictionary containing signed transaction data
        """
        try:
            # Server signer participates automatically
            signed_tx = await self.wallet.sign_transaction(tx_params)
            
            return {
                'signed_tx': signed_tx,
                'requires_cosigner': False,  # CDP handles this automatically
                'status': 'signed',
                'mpc_used': True
            }
            
        except Exception as e:
            self.logger.error(f"MPC signing failed: {str(e)}")
            raise CDPError(f"MPC transaction signing failed: {str(e)}")
    
    def get_mpc_status(self) -> Dict[str, Any]:
        """
        Get MPC wallet status and configuration.
        
        Returns:
            MPC wallet status information
        """
        return {
            'mpc_enabled': True,
            'server_signer_configured': bool(self.server_signer_url),
            'wallet_address': self.wallet.address if self.wallet else None,
            'requires_server_signer': True,
            'security_level': 'production'
        }