#!/bin/bash
# Run MoCap app with GTK4 support on macOS
# This sets the library path for Homebrew-installed GTK4

export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

cd "$(dirname "$0")"
.venv/bin/python main.py "$@"
