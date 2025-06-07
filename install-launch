#!/usr/bin/env python3
"""Install Breeze as a macOS LaunchAgent with full configuration support."""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict
import plistlib

# Try to import required dependencies
try:
    from dotenv import load_dotenv
except ImportError:
    print(
        "Error: python-dotenv is required. Install it with: pip install python-dotenv"
    )
    sys.exit(1)

try:
    from jinja2 import Template
except ImportError:
    print("Error: jinja2 is required. Install it with: pip install jinja2")
    sys.exit(1)


class BreezeInstaller:
    """Installer for Breeze MCP server as a macOS LaunchAgent."""

    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.working_dir = self.script_dir
        self.launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
        self.plist_template = self.script_dir / "com.breeze-mcp.server.plist.template"
        self.plist_file = self.script_dir / "com.breeze-mcp.server.plist"
        self.dest_plist = self.launch_agents_dir / "com.breeze-mcp.server.plist"

        # Load environment variables from .env file if it exists
        env_file = self.script_dir / ".env"
        if env_file.exists():
            print(f"Loading configuration from {env_file}")
            load_dotenv(env_file)

        # All supported configuration options with their defaults
        self.config = {
            # Server settings
            "BREEZE_HOST": os.environ.get("BREEZE_HOST", "127.0.0.1"),
            "BREEZE_PORT": os.environ.get("BREEZE_PORT", "9483"),
            # Data settings
            "BREEZE_DATA_ROOT": os.environ.get(
                "BREEZE_DATA_ROOT",
                str(Path.home() / "Library" / "Application Support" / "breeze"),
            ),
            "BREEZE_DB_NAME": os.environ.get("BREEZE_DB_NAME", "code_index"),
            # Model settings
            "BREEZE_EMBEDDING_MODEL": os.environ.get(
                "BREEZE_EMBEDDING_MODEL", "ibm-granite/granite-embedding-125m-english"
            ),
            "BREEZE_EMBEDDING_DEVICE": os.environ.get("BREEZE_EMBEDDING_DEVICE", "cpu"),
            "BREEZE_EMBEDDING_API_KEY": os.environ.get("BREEZE_EMBEDDING_API_KEY", ""),
            # Concurrency settings
            "BREEZE_CONCURRENT_READERS": os.environ.get(
                "BREEZE_CONCURRENT_READERS", "20"
            ),
            "BREEZE_CONCURRENT_EMBEDDERS": os.environ.get(
                "BREEZE_CONCURRENT_EMBEDDERS", "10"
            ),
            "BREEZE_CONCURRENT_WRITERS": os.environ.get(
                "BREEZE_CONCURRENT_WRITERS", "10"
            ),
            "BREEZE_VOYAGE_CONCURRENT_REQUESTS": os.environ.get(
                "BREEZE_VOYAGE_CONCURRENT_REQUESTS", "5"
            ),
            # Logging
            "LOG_DIR": os.environ.get("LOG_DIR", "/usr/local/var/log"),
            # Python settings
            "PYTHONUNBUFFERED": "1",
        }

        # Detect virtual environment
        self.venv_path = self._detect_venv()
        self.python_path = str(self.venv_path / "bin" / "python")
        self.venv_bin = str(self.venv_path / "bin")

    def _detect_venv(self) -> Path:
        """Detect the virtual environment path."""
        # Check if we're already in a virtual environment
        if os.environ.get("VIRTUAL_ENV"):
            return Path(os.environ["VIRTUAL_ENV"])

        # Check for .venv in the project directory
        venv_path = self.working_dir / ".venv"
        if venv_path.exists() and (venv_path / "bin" / "python").exists():
            return venv_path

        # Check for venv in the project directory
        venv_path = self.working_dir / "venv"
        if venv_path.exists() and (venv_path / "bin" / "python").exists():
            return venv_path

        print("Error: No virtual environment found.")
        print(
            "Please create or activate a virtual environment before running this script."
        )
        sys.exit(1)

    def check_requirements(self) -> bool:
        """Check if all requirements are met."""
        # Check for required API key if using cloud embedding models
        embedding_model = self.config["BREEZE_EMBEDDING_MODEL"].lower()

        # List of models that require API keys
        api_key_models = ["voyage", "openai", "cohere", "anthropic"]

        if any(provider in embedding_model for provider in api_key_models):
            if not self.config.get("BREEZE_EMBEDDING_API_KEY"):
                print(
                    f"Error: BREEZE_EMBEDDING_API_KEY is required for model '{embedding_model}'"
                )
                print("Please set it in your .env file or as an environment variable:")
                print("  export BREEZE_EMBEDDING_API_KEY='your-api-key'")
                return False

        # Check if template exists
        if not self.plist_template.exists():
            print(f"Error: Template file {self.plist_template} not found")
            return False

        return True

    def create_directories(self):
        """Create necessary directories."""
        # Create LaunchAgents directory
        self.launch_agents_dir.mkdir(parents=True, exist_ok=True)

        # Create log directory
        log_dir = Path(self.config["LOG_DIR"])
        try:
            # Try to create without sudo first
            log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fall back to sudo if needed
            subprocess.run(["sudo", "mkdir", "-p", str(log_dir)], check=True)
            subprocess.run(
                ["sudo", "chown", os.environ["USER"], str(log_dir)], check=True
            )

        # Create data directory
        data_dir = Path(self.config["BREEZE_DATA_ROOT"])
        data_dir.mkdir(parents=True, exist_ok=True)

    def generate_plist(self) -> Dict:
        """Generate the plist configuration using Jinja2."""
        # Read template
        with open(self.plist_template, "r") as f:
            template = Template(f.read())

        # Create context with all values
        context = {
            # Path values
            "PYTHON_PATH": self.python_path,
            "WORKING_DIR": str(self.working_dir),
            "VENV_BIN": self.venv_bin,
            "VIRTUAL_ENV": str(self.venv_path),
            "LOG_DIR": self.config["LOG_DIR"],
            # All config values
            **self.config,
        }

        # Render template
        plist_content = template.render(context)

        # Write generated plist
        with open(self.plist_file, "w") as f:
            f.write(plist_content)

        # Parse and return as dict for display
        with open(self.plist_file, "rb") as f:
            return plistlib.load(f)

    def install_service(self):
        """Install the LaunchAgent."""
        # Unload existing agent if present
        if (
            subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True
            ).stdout.find("com.breeze-mcp.server")
            != -1
        ):
            print("Unloading existing agent...")
            subprocess.run(
                ["launchctl", "unload", str(self.dest_plist)], stderr=subprocess.DEVNULL
            )

        # Copy plist file
        shutil.copy2(self.plist_file, self.dest_plist)

        # Load the agent
        subprocess.run(["launchctl", "load", str(self.dest_plist)], check=True)

    def print_configuration(self):
        """Print the current configuration."""
        print("\nBreeze MCP server installed and started as a LaunchAgent")
        print("\nConfiguration:")
        print(f"  Host: {self.config['BREEZE_HOST']}")
        print(f"  Port: {self.config['BREEZE_PORT']}")
        print(f"  Data Root: {self.config['BREEZE_DATA_ROOT']}")
        print(f"  Database Name: {self.config['BREEZE_DB_NAME']}")
        print(f"  Embedding Model: {self.config['BREEZE_EMBEDDING_MODEL']}")
        print(f"  Embedding Device: {self.config['BREEZE_EMBEDDING_DEVICE']}")
        print(f"  Python: {self.python_path}")
        print(f"  Logs: {self.config['LOG_DIR']}/breeze-mcp.log")

        print("\nCommands:")
        print("  Check status: launchctl list | grep breeze")
        print(f"  Stop: launchctl unload {self.dest_plist}")
        print(f"  Start: launchctl load {self.dest_plist}")
        print(f"  View logs: tail -f {self.config['LOG_DIR']}/breeze-mcp.log")

    def run(self):
        """Run the installation process."""
        print("Breeze MCP Server Installer")
        print("=" * 40)

        # Check requirements
        if not self.check_requirements():
            sys.exit(1)

        # Create directories
        print("\nCreating directories...")
        self.create_directories()

        # Generate plist
        print("Generating plist file from template...")
        self.generate_plist()

        # Install service
        print("Installing LaunchAgent...")
        self.install_service()

        # Print configuration
        self.print_configuration()

        # Clean up generated plist file (optional)
        # self.plist_file.unlink()


def main():
    """Main entry point."""
    installer = BreezeInstaller()
    try:
        installer.run()
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
