"""
Google Cloud Secret Manager integration for secure credential management.

This module provides a centralized way to access secrets stored in Google Cloud
Secret Manager, eliminating the need for environment variables or .env files.
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from google.cloud import secretmanager
from google.api_core import exceptions

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Manages access to secrets stored in Google Cloud Secret Manager.
    
    This class provides a secure way to retrieve API keys and other sensitive
    configuration without storing them in environment variables or files.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize the Secret Manager client.
        
        Args:
            project_id: Google Cloud project ID. If not provided, will try to
                       get from environment or Application Default Credentials.
        """
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'aerodrome-brain-1756490979')
        
        try:
            self.client = secretmanager.SecretManagerServiceClient()
            self._available = True
            logger.info(f"Secret Manager initialized for project: {self.project_id}")
        except Exception as e:
            logger.warning(f"Secret Manager not available: {e}. Falling back to environment variables.")
            self.client = None
            self._available = False
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_id: str, version: str = "latest") -> Optional[str]:
        """
        Retrieve a secret value from Secret Manager.
        
        Args:
            secret_id: The ID of the secret to retrieve
            version: The version of the secret (default: "latest")
            
        Returns:
            The secret value as a string, or None if not found
        """
        if not self._available:
            # Fall back to environment variable
            env_key = secret_id.upper().replace('-', '_')
            return os.environ.get(env_key)
        
        try:
            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            
            # Access the secret
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            logger.debug(f"Successfully retrieved secret: {secret_id}")
            return secret_value
            
        except exceptions.NotFound:
            logger.warning(f"Secret not found: {secret_id}")
            # Fall back to environment variable
            env_key = secret_id.upper().replace('-', '_')
            return os.environ.get(env_key)
            
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_id}: {e}")
            # Fall back to environment variable
            env_key = secret_id.upper().replace('-', '_')
            return os.environ.get(env_key)
    
    def create_secret(self, secret_id: str, secret_value: str) -> bool:
        """
        Create a new secret in Secret Manager.
        
        Args:
            secret_id: The ID for the new secret
            secret_value: The value to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self._available:
            logger.error("Secret Manager not available")
            return False
        
        try:
            # Create the secret
            parent = f"projects/{self.project_id}"
            secret = self.client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            
            # Add the secret version
            self.client.add_secret_version(
                request={
                    "parent": secret.name,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            logger.info(f"Successfully created secret: {secret_id}")
            return True
            
        except exceptions.AlreadyExists:
            logger.warning(f"Secret already exists: {secret_id}")
            return self.update_secret(secret_id, secret_value)
            
        except Exception as e:
            logger.error(f"Error creating secret {secret_id}: {e}")
            return False
    
    def update_secret(self, secret_id: str, secret_value: str) -> bool:
        """
        Update an existing secret with a new version.
        
        Args:
            secret_id: The ID of the secret to update
            secret_value: The new value
            
        Returns:
            True if successful, False otherwise
        """
        if not self._available:
            logger.error("Secret Manager not available")
            return False
        
        try:
            parent = f"projects/{self.project_id}/secrets/{secret_id}"
            
            # Add a new version
            self.client.add_secret_version(
                request={
                    "parent": parent,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            # Clear cache for this secret
            self.get_secret.cache_clear()
            
            logger.info(f"Successfully updated secret: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating secret {secret_id}: {e}")
            return False
    
    def delete_secret(self, secret_id: str) -> bool:
        """
        Delete a secret from Secret Manager.
        
        Args:
            secret_id: The ID of the secret to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self._available:
            logger.error("Secret Manager not available")
            return False
        
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}"
            self.client.delete_secret(request={"name": name})
            
            # Clear cache
            self.get_secret.cache_clear()
            
            logger.info(f"Successfully deleted secret: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting secret {secret_id}: {e}")
            return False
    
    def list_secrets(self) -> Dict[str, str]:
        """
        List all secrets in the project.
        
        Returns:
            Dictionary of secret IDs and their creation times
        """
        if not self._available:
            logger.error("Secret Manager not available")
            return {}
        
        try:
            parent = f"projects/{self.project_id}"
            secrets = {}
            
            for secret in self.client.list_secrets(request={"parent": parent}):
                secret_id = secret.name.split('/')[-1]
                secrets[secret_id] = str(secret.create_time)
            
            return secrets
            
        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return {}
    
    def get_all_credentials(self) -> Dict[str, Optional[str]]:
        """
        Retrieve all credentials needed for the Aerodrome Brain.
        
        Returns:
            Dictionary containing all credential values
        """
        credentials = {
            'gemini_api_key': self.get_secret('gemini-api-key'),
            'quicknode_url': self.get_secret('quicknode-url'),
            'mem0_api_key': self.get_secret('mem0-api-key'),
            'neo4j_uri': self.get_secret('neo4j-uri'),
            'neo4j_username': self.get_secret('neo4j-username'),
            'neo4j_password': self.get_secret('neo4j-password'),
            'openai_api_key': self.get_secret('openai-api-key'),
            'cdp_api_key_name': self.get_secret('cdp-api-key-name'),
            'cdp_api_key_private': self.get_secret('cdp-api-key-private'),
        }
        
        # Filter out None values
        return {k: v for k, v in credentials.items() if v is not None}


# Global instance for easy access
_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """
    Get the global SecretManager instance.
    
    Returns:
        The global SecretManager instance
    """
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


def get_secret(secret_id: str) -> Optional[str]:
    """
    Convenience function to get a secret value.
    
    Args:
        secret_id: The ID of the secret to retrieve
        
    Returns:
        The secret value or None if not found
    """
    return get_secret_manager().get_secret(secret_id)


def get_credentials() -> Dict[str, Optional[str]]:
    """
    Get all credentials for the Aerodrome Brain.
    
    Returns:
        Dictionary containing all credential values
    """
    return get_secret_manager().get_all_credentials()


# For backward compatibility with .env files
class Settings:
    """
    Settings class that uses Secret Manager but falls back to environment variables.
    """
    
    def __init__(self):
        self.sm = get_secret_manager()
        
        # Load all credentials
        creds = self.sm.get_all_credentials()
        
        # Core settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # API Keys from Secret Manager
        self.gemini_api_key = creds.get('gemini_api_key', '')
        self.quicknode_url = creds.get('quicknode_url', '')
        self.mem0_api_key = creds.get('mem0_api_key', '')
        self.openai_api_key = creds.get('openai_api_key', '')
        
        # Neo4j
        self.neo4j_uri = creds.get('neo4j_uri', 'bolt://localhost:7687')
        self.neo4j_username = creds.get('neo4j_username', 'neo4j')
        self.neo4j_password = creds.get('neo4j_password', '')
        
        # CDP (optional)
        self.cdp_api_key_name = creds.get('cdp_api_key_name', '')
        self.cdp_api_key_private_key = creds.get('cdp_api_key_private', '')
        
        # Google Cloud
        self.google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT", "aerodrome-brain-1756490979")
        self.google_cloud_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Other settings from environment
        self.network = os.getenv("NETWORK", "base-mainnet")
        self.agent_name = os.getenv("AGENT_NAME", "aerodrome-brain")
        self.operation_mode = os.getenv("OPERATION_MODE", "autonomous")
        
        logger.info(f"Settings loaded. Using Secret Manager: {self.sm._available}")


# Create global settings instance
settings = Settings()