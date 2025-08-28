"""
CLI Interface for Aerodrome AI Agent

Command-line interface for starting, monitoring, and controlling the agent.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .config.settings import Settings
from .orchestrator import OrchestratorManager, OperationMode, OrchestatorConfig
from .main import AerodromeAgent, validate_environment, test_agent as run_test_agent
from .utils.logger import initialize_logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--env-file', default='.env', help='Path to environment file')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, env_file: str, debug: bool):
    """Aerodrome AI Agent - Autonomous DeFi Portfolio Manager"""
    
    # Load environment file if it exists
    if Path(env_file).exists():
        load_dotenv(env_file)
    
    # Set debug level
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('aerodrome_ai_agent').setLevel(logging.DEBUG)
    
    # Store settings in context
    ctx.ensure_object(dict)
    ctx.obj['settings'] = Settings(env_file if Path(env_file).exists() else None)
    ctx.obj['env_file'] = env_file if Path(env_file).exists() else None
    
    # Initialize logging
    initialize_logging(ctx.obj['settings'])


@cli.command()
@click.option('--mode', 
              type=click.Choice(['simulation', 'testnet', 'mainnet', 'manual']),
              default='simulation',
              help='Operation mode')
@click.option('--interval', 
              type=int, 
              help='Brain cycle interval in seconds')
@click.option('--max-cycles', 
              type=int,
              help='Maximum number of cycles (for testing)')
@click.pass_context
def start(ctx, mode: str, interval: Optional[int], max_cycles: Optional[int]):
    """Start the Aerodrome AI Agent using the orchestrator"""
    
    settings = ctx.obj['settings']
    
    click.echo(f"🚀 Starting Aerodrome AI Agent in {mode} mode...")
    
    try:
        # Validate configuration
        validation = validate_environment(ctx.obj.get('env_file'))
        if not validation['valid']:
            click.echo(f"❌ Configuration invalid: {validation['error']}")
            sys.exit(1)
        
        # Initialize agent
        agent = AerodromeAgent(env_file=ctx.obj.get('env_file'))
        
        # Run the agent
        asyncio.run(agent.start(mode, interval, max_cycles))
        
    except KeyboardInterrupt:
        click.echo("\n⏹️  Agent stopped by user")
    except Exception as e:
        click.echo(f"❌ Error starting agent: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Check agent status and health"""
    
    settings = ctx.obj['settings']
    
    click.echo("📊 Aerodrome AI Agent Status")
    click.echo("=" * 40)
    
    # Configuration validation
    validation = validate_environment(ctx.obj.get('env_file'))
    
    if validation['valid']:
        click.echo(f"Environment: {validation['environment']}")
        click.echo(f"Network: {validation['network']}")
        click.echo(f"Operation Mode: {validation['operation_mode']}")
        click.echo(f"Risk Level: {validation['risk_level']}")
        
        # API Key status
        api_keys = validation['api_keys']
        click.echo(f"\nAPI Keys:")
        click.echo(f"  OpenAI: {'✅' if api_keys['openai'] else '❌'}")
        click.echo(f"  CDP Name: {'✅' if api_keys['cdp_name'] else '❌'}")
        click.echo(f"  CDP Key: {'✅' if api_keys['cdp_key'] else '❌'}")
        
        click.echo("\n✅ Agent configuration is valid")
        
        # Try to get runtime status
        try:
            agent = AerodromeAgent(env_file=ctx.obj.get('env_file'))
            runtime_status = asyncio.run(agent.get_status())
            
            click.echo(f"\nRuntime Status:")
            click.echo(f"  State: {runtime_status.get('state', 'unknown')}")
            
            if runtime_status.get('uptime_seconds'):
                uptime = runtime_status['uptime_seconds']
                hours = int(uptime // 3600)
                minutes = int((uptime % 3600) // 60)
                click.echo(f"  Uptime: {hours}h {minutes}m")
            
            if runtime_status.get('cycle_count'):
                click.echo(f"  Cycles: {runtime_status['cycle_count']}")
            
            if runtime_status.get('health'):
                health = runtime_status['health']
                click.echo(f"  Health: {health.get('overall_status', 'unknown')}")
                click.echo(f"  Checks: {health.get('checks_passing', 0)}/{health.get('total_checks', 0)} passing")
                
                if runtime_status.get('alerts', 0) > 0:
                    click.echo(f"  ⚠️  Active alerts: {runtime_status['alerts']}")
                    
        except Exception as e:
            click.echo(f"\nRuntime Status: Not running or error getting status")
    else:
        click.echo(f"❌ Configuration invalid: {validation['error']}")
        click.echo("   Please check your .env file")


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration"""
    
    settings = ctx.obj['settings']
    config_dict = settings.to_dict()
    
    click.echo("⚙️  Agent Configuration")
    click.echo("=" * 40)
    
    for key, value in config_dict.items():
        # Hide sensitive values
        if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'token']):
            display_value = "***" if value else "Not set"
        else:
            display_value = value
        
        click.echo(f"{key}: {display_value}")


@cli.command()
@click.option('--cycles', default=1, help='Number of brain cycles to run')
@click.pass_context
def test(ctx, cycles: int):
    """Test agent components using orchestrator"""
    
    click.echo(f"🧪 Testing agent components ({cycles} cycles)...")
    
    try:
        success = asyncio.run(run_test_agent(cycles, ctx.obj.get('env_file')))
        if success:
            click.echo("✅ All tests completed successfully")
        else:
            click.echo("❌ Tests failed")
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def health(ctx):
    """Check system health and monitoring status"""
    
    click.echo("🏥 System Health Check")
    click.echo("=" * 40)
    
    try:
        agent = AerodromeAgent(env_file=ctx.obj.get('env_file'))
        status = asyncio.run(agent.get_status())
        
        if 'health' in status:
            health = status['health']
            overall_status = health.get('overall_status', 'unknown')
            
            # Overall health status
            status_emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '🚨',
                'unknown': '❓'
            }
            
            click.echo(f"Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
            click.echo(f"Health Checks: {health.get('checks_passing', 0)}/{health.get('total_checks', 0)} passing")
            click.echo(f"Critical Failures: {health.get('critical_failures', 0)}")
            click.echo(f"Active Alerts: {health.get('active_alerts', 0)}")
            
            # System uptime
            if health.get('system_uptime'):
                uptime_hours = health['system_uptime'] / 3600
                click.echo(f"System Uptime: {uptime_hours:.1f} hours")
            
            # Component status
            if 'component_status' in health:
                click.echo(f"\nComponent Status:")
                for component, comp_status in health['component_status'].items():
                    click.echo(f"  {component}: {comp_status.get('status', 'unknown')}")
        
        else:
            click.echo("❓ Health information not available (agent may not be running)")
    
    except Exception as e:
        click.echo(f"❌ Error checking health: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def alerts(ctx):
    """Show active alerts and warnings"""
    
    click.echo("🚨 System Alerts")
    click.echo("=" * 40)
    
    try:
        agent = AerodromeAgent(env_file=ctx.obj.get('env_file'))
        status = asyncio.run(agent.get_status())
        
        # This would need to be implemented in the agent to return alert details
        click.echo("Alert monitoring not yet implemented in runtime status")
        click.echo("Please check logs for detailed alert information")
        
    except Exception as e:
        click.echo(f"❌ Error getting alerts: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def metrics(ctx):
    """Show performance metrics"""
    
    click.echo("📈 Performance Metrics")
    click.echo("=" * 40)
    
    try:
        agent = AerodromeAgent(env_file=ctx.obj.get('env_file'))
        status = asyncio.run(agent.get_status())
        
        if 'performance' in status:
            perf = status['performance']
            
            # Show available metrics
            for metric_name, value in perf.items():
                if isinstance(value, (int, float)):
                    click.echo(f"{metric_name}: {value:.2f}")
                else:
                    click.echo(f"{metric_name}: {value}")
        else:
            click.echo("📊 No performance metrics available")
        
        # Show basic runtime metrics
        if status.get('cycle_count'):
            click.echo(f"\nRuntime Metrics:")
            click.echo(f"  Total Cycles: {status['cycle_count']}")
            
        if status.get('uptime_seconds'):
            uptime = status['uptime_seconds']
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            click.echo(f"  Uptime: {hours}h {minutes}m")
    
    except Exception as e:
        click.echo(f"❌ Error getting metrics: {e}")
        sys.exit(1)


@cli.command()
@click.option('--emergency', is_flag=True, help='Emergency stop (immediate)')
@click.pass_context
def stop(ctx, emergency: bool):
    """Stop the running agent"""
    
    if emergency:
        click.echo("🛑 Emergency stop requested...")
    else:
        click.echo("⏹️ Graceful stop requested...")
    
    try:
        agent = AerodromeAgent(env_file=ctx.obj.get('env_file'))
        success = asyncio.run(agent.stop())
        
        if success:
            click.echo("✅ Agent stopped successfully")
        else:
            click.echo("❌ Failed to stop agent")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error stopping agent: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context 
def pause(ctx):
    """Pause agent operations"""
    
    click.echo("⏸️ Pausing agent operations...")
    
    try:
        # This would need to be implemented in the agent
        click.echo("❓ Pause/resume functionality not yet implemented")
        click.echo("Use 'stop' command to stop the agent")
        
    except Exception as e:
        click.echo(f"❌ Error pausing agent: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def resume(ctx):
    """Resume paused agent operations"""
    
    click.echo("▶️ Resuming agent operations...")
    
    try:
        # This would need to be implemented in the agent
        click.echo("❓ Pause/resume functionality not yet implemented")
        click.echo("Use 'start' command to start the agent")
        
    except Exception as e:
        click.echo(f"❌ Error resuming agent: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def logs(ctx):
    """Show recent log entries"""
    
    click.echo("📋 Recent Log Entries")
    click.echo("=" * 40)
    
    settings = ctx.obj['settings']
    log_dir = Path(settings.log_directory)
    
    # Show recent entries from main log file
    main_log = log_dir / "agent.log"
    
    if main_log.exists():
        try:
            with open(main_log, 'r') as f:
                lines = f.readlines()
                # Show last 20 lines
                recent_lines = lines[-20:] if len(lines) > 20 else lines
                for line in recent_lines:
                    click.echo(line.rstrip())
        except Exception as e:
            click.echo(f"❌ Error reading log file: {e}")
    else:
        click.echo("📂 No log file found")
        click.echo(f"Expected location: {main_log}")


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate agent configuration and environment"""
    
    click.echo("🔍 Validating Configuration")
    click.echo("=" * 40)
    
    validation = validate_environment(ctx.obj.get('env_file'))
    
    if validation['valid']:
        click.echo("✅ Configuration is valid")
        click.echo(f"Environment: {validation['environment']}")
        click.echo(f"Network: {validation['network']}")
        click.echo(f"Operation Mode: {validation['operation_mode']}")
        click.echo(f"Risk Level: {validation['risk_level']}")
        
        api_keys = validation['api_keys']
        click.echo(f"\nAPI Keys:")
        click.echo(f"  OpenAI: {'✅' if api_keys['openai'] else '❌'}")
        click.echo(f"  CDP Name: {'✅' if api_keys['cdp_name'] else '❌'}")
        click.echo(f"  CDP Key: {'✅' if api_keys['cdp_key'] else '❌'}")
        
        if all(api_keys.values()):
            click.echo(f"\n🎯 Ready to start in {validation['operation_mode']} mode")
        else:
            click.echo(f"\n⚠️  Missing API keys - check your .env file")
    else:
        click.echo(f"❌ Configuration invalid: {validation['error']}")
        sys.exit(1)


# Enhanced CLI commands for orchestrator


def main():
    """Main entry point"""
    cli()