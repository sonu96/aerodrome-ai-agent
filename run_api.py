#!/usr/bin/env python3
"""
Local development server for the Aerodrome AI Agent API

This script starts the FastAPI application in development mode
with hot reloading and debug features enabled.

Usage:
    python run_api.py [--port PORT] [--host HOST] [--reload]
    
Examples:
    python run_api.py                    # Default: localhost:8080
    python run_api.py --port 8000        # Custom port
    python run_api.py --host 0.0.0.0     # Bind to all interfaces
    python run_api.py --reload           # Enable hot reloading
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="Run the Aerodrome AI Agent API server"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ.setdefault("ENV", "development")
    os.environ.setdefault("LOG_LEVEL", args.log_level.upper())
    
    # Load environment variables from .env if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        from dotenv import load_dotenv
        load_dotenv(env_file)
    else:
        print("No .env file found, using environment defaults")
        print("Copy .env.example to .env and configure your settings")
    
    # Validate required environment variables
    required_vars = [
        "API_KEY",
        "MEM0_API_KEY", 
        "GEMINI_API_KEY",
        "QUICKNODE_URL"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Warning: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nThe API may not function correctly without these variables.")
        print("Please check your .env file or environment configuration.\n")
    
    # Import and run the server
    try:
        import uvicorn
        from src.api.main import app
        
        print("🚀 Starting Aerodrome AI Agent API server...")
        print(f"📍 Server will be available at: http://{args.host}:{args.port}")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"📊 Health Check: http://{args.host}:{args.port}/health")
        
        if args.reload:
            print("🔄 Auto-reload enabled")
        
        print("\n" + "="*50)
        
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True,
            server_header=False,
            date_header=False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()