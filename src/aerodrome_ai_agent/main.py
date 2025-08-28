"""
Aerodrome AI Agent - Main Application Entry Point

This is the main entry point for the Aerodrome AI Agent. It provides:
- Command-line interface integration
- Direct programmatic API
- Environment setup and validation
- Graceful startup and shutdown procedures
- Error handling and recovery
- Multiple operation modes
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .orchestrator import AerodromeOrchestrator, OrchestratorManager, OperationMode, OrchestatorConfig
from .config.settings import Settings
from .utils.logger import setup_basic_logging, get_logger


# Setup basic logging for main module
setup_basic_logging()
logger = logging.getLogger(__name__)


class AerodromeAgent:
    """
    Main Aerodrome AI Agent class providing both CLI and programmatic interfaces
    
    This class serves as the primary interface for running the Aerodrome AI Agent
    in different contexts and with different configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize the Aerodrome AI Agent
        
        Args:
            config_path: Path to configuration file (optional)
            env_file: Path to environment file (optional)
        """
        self.config_path = config_path
        self.env_file = env_file
        
        # Load settings
        self.settings = Settings(env_file)
        
        # Validate configuration
        try:
            self.settings.validate()
            logger.info("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize orchestrator manager
        self.manager: Optional[OrchestratorManager] = None
        
        logger.info("Aerodrome AI Agent initialized")
    
    async def start(
        self,
        mode: str = "simulation",
        interval: Optional[int] = None,
        max_cycles: Optional[int] = None,
        auto_start: bool = True
    ) -> bool:
        """
        Start the agent
        
        Args:
            mode: Operation mode (simulation, testnet, mainnet, manual)
            interval: Brain cycle interval in seconds
            max_cycles: Maximum number of cycles to run
            auto_start: Whether to start immediately
            
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Parse operation mode
            operation_mode = OperationMode(mode.lower())
            logger.info(f"Starting agent in {operation_mode.value} mode")
            
            # Create orchestrator config
            orchestrator_config = OrchestatorConfig(
                operation_mode=operation_mode,
                auto_start=auto_start
            )
            
            # Override brain interval if specified
            if interval:
                self.settings.brain_observation_interval = interval
            
            # Create and start orchestrator manager
            self.manager = OrchestratorManager(self.settings, orchestrator_config)
            
            # Start the agent
            await self.manager.run(operation_mode, max_cycles)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the agent
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if self.manager:
                await self.manager.stop()
                self.manager = None
            
            logger.info("Agent stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get agent status
        
        Returns:
            Dictionary containing agent status information
        """
        if self.manager:
            return await self.manager.get_status()
        else:
            return {
                "state": "not_running",
                "message": "Agent is not currently running"
            }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate agent configuration
        
        Returns:
            Dictionary containing validation results
        """
        try:
            self.settings.validate()
            
            return {
                "valid": True,
                "environment": self.settings.environment,
                "network": self.settings.network,
                "operation_mode": self.settings.operation_mode,
                "risk_level": self.settings.risk_level,
                "api_keys": {
                    "openai": bool(self.settings.openai_api_key),
                    "cdp_name": bool(self.settings.cdp_api_key_name),
                    "cdp_key": bool(self.settings.cdp_api_key_private_key)
                }
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration (with sensitive values masked)
        
        Returns:
            Dictionary containing configuration
        """
        config = self.settings.to_dict()
        
        # Mask sensitive values
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'token', 'password']):
                config[key] = "***" if value else "Not set"
        
        return config


# Convenience functions for common operations
async def run_agent(
    mode: str = "simulation",
    interval: Optional[int] = None,
    max_cycles: Optional[int] = None,
    env_file: Optional[str] = None
):
    """
    Convenience function to run the agent with default settings
    
    Args:
        mode: Operation mode
        interval: Brain cycle interval
        max_cycles: Maximum cycles to run
        env_file: Path to environment file
    """
    agent = AerodromeAgent(env_file=env_file)
    await agent.start(mode, interval, max_cycles)


async def test_agent(cycles: int = 1, env_file: Optional[str] = None) -> bool:
    """
    Test the agent components
    
    Args:
        cycles: Number of test cycles to run
        env_file: Path to environment file
        
    Returns:
        True if tests pass, False otherwise
    """
    try:
        logger.info(f"Testing agent components ({cycles} cycles)")
        agent = AerodromeAgent(env_file=env_file)
        
        # Validate configuration
        validation = agent.validate_configuration()
        if not validation["valid"]:
            logger.error(f"Configuration invalid: {validation['error']}")
            return False
        
        # Run test cycles
        success = await agent.start(mode="simulation", max_cycles=cycles)
        
        if success:
            logger.info("Agent test completed successfully")
        else:
            logger.error("Agent test failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        return False


def validate_environment(env_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate the environment setup
    
    Args:
        env_file: Path to environment file
        
    Returns:
        Dictionary containing validation results
    """
    try:
        agent = AerodromeAgent(env_file=env_file)
        return agent.validate_configuration()
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def main():
    """
    Main function for direct execution
    
    This function is called when the module is executed directly.
    It provides a simple CLI interface for common operations.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aerodrome AI Agent - Autonomous DeFi Portfolio Manager"
    )
    
    parser.add_argument(
        "command",
        choices=["start", "test", "validate", "status"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--mode",
        choices=["simulation", "testnet", "mainnet", "manual"],
        default="simulation",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        help="Brain cycle interval in seconds"
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        help="Number of cycles to run (for test and limited runs)"
    )
    
    parser.add_argument(
        "--env-file",
        help="Path to environment file"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    if args.command == "validate":
        print("Validating environment...")
        result = validate_environment(args.env_file)
        
        if result["valid"]:
            print("✅ Configuration is valid")
            print(f"Environment: {result['environment']}")
            print(f"Network: {result['network']}")
            print(f"Operation Mode: {result['operation_mode']}")
            print(f"Risk Level: {result['risk_level']}")
            
            api_keys = result["api_keys"]
            print(f"API Keys:")
            print(f"  OpenAI: {'✅' if api_keys['openai'] else '❌'}")
            print(f"  CDP Name: {'✅' if api_keys['cdp_name'] else '❌'}")
            print(f"  CDP Key: {'✅' if api_keys['cdp_key'] else '❌'}")
        else:
            print(f"❌ Configuration invalid: {result['error']}")
            sys.exit(1)
    
    elif args.command == "test":
        print(f"Testing agent ({args.cycles or 1} cycles)...")
        
        async def run_test():
            return await test_agent(args.cycles or 1, args.env_file)
        
        success = asyncio.run(run_test())
        if success:
            print("✅ Tests passed")
        else:
            print("❌ Tests failed")
            sys.exit(1)
    
    elif args.command == "start":
        print(f"Starting agent in {args.mode} mode...")
        
        async def run_start():
            await run_agent(args.mode, args.interval, args.cycles, args.env_file)
        
        try:
            asyncio.run(run_start())
        except KeyboardInterrupt:
            print("\n⏹️ Agent stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    elif args.command == "status":
        print("Checking agent status...")
        
        async def check_status():
            agent = AerodromeAgent(env_file=args.env_file)
            status = await agent.get_status()
            return status
        
        status = asyncio.run(check_status())
        
        print(f"State: {status.get('state', 'unknown')}")
        if status.get('uptime_seconds'):
            uptime = status['uptime_seconds']
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            print(f"Uptime: {hours}h {minutes}m")
        
        if status.get('cycle_count'):
            print(f"Cycles: {status['cycle_count']}")


if __name__ == "__main__":
    main()