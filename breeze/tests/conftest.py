"""Pytest configuration for Breeze tests."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure event loop for async tests
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session."""
    if sys.platform == "win32":
        # Windows requires a specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.get_event_loop_policy()


@pytest.fixture
def anyio_backend():
    """Configure anyio backend for async tests."""
    return "asyncio"


# Set environment variables for testing
os.environ["BREEZE_HOST"] = "127.0.0.1"
os.environ["BREEZE_PORT"] = "0"  # Use random port for testing