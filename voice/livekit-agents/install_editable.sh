#!/bin/bash
set -e

# Activate the virtual environment
source venv/bin/activate

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "You are not in a virtual environment."
    exit 1
fi

pip install -e ./livekit-agents --config-settings editable_mode=strict
