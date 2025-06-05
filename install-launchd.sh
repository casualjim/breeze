#!/bin/bash
set -e

PLIST_FILE="com.breeze-mcp.server.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
DEST_PLIST="$LAUNCH_AGENTS_DIR/$PLIST_FILE"

# Ensure the LaunchAgents directory exists
mkdir -p "$LAUNCH_AGENTS_DIR"

# Copy the plist file
cp "$PLIST_FILE" "$DEST_PLIST"

# Ensure log directory exists
sudo mkdir -p /usr/local/var/log
sudo chown "$USER" /usr/local/var/log

# Load the agent
launchctl load "$DEST_PLIST"

echo "Breeze MCP server installed and started as a LaunchAgent"
echo "To check status: launchctl list | grep breeze"
echo "To stop: launchctl unload $DEST_PLIST"
echo "To start: launchctl load $DEST_PLIST"
echo "Logs: /usr/local/var/log/breeze-mcp.log"