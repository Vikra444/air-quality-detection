"""
Main entry point for AirGuard system.
"""

import asyncio
import argparse
from src.utils.logger import get_logger
from src.api.main import app
import uvicorn

logger = get_logger("main")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AirGuard - Air Quality Monitoring System")
    parser.add_argument(
        "--mode",
        choices=["api", "dashboard"],
        default="api",
        help="Run mode"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        port = args.port or 8000
        logger.info(f"Starting AirGuard API server on {args.host}:{port}")
        uvicorn.run(
            app,
            host=args.host,
            port=port,
            reload=args.debug,
            log_level="info"
        )
    elif args.mode == "dashboard":
        logger.info("Dashboard mode not yet implemented")
        # Will be implemented in Phase 7


if __name__ == "__main__":
    main()

