#!/usr/bin/env python3
"""Main entry point for Breeze MCP server."""

import os

# Disable progress bars early
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_DISABLE_PROGRESS_BAR"] = "1"

import sentence_transformers

sentence_transformers.SentenceTransformer.default_show_progress_bar = False

from breeze.cli import main

if __name__ == "__main__":
    main()
