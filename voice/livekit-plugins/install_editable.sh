#!/bin/bash

source venv/bin/activate

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Go to the parent directory of the script
cd "$SCRIPT_DIR/.." || exit

# Install the plugins using relative paths
pip install -e "./livekit-plugins/livekit-plugins-cartesia"
pip install -e "./livekit-plugins/livekit-plugins-deepgram"
pip install -e "./livekit-plugins/livekit-plugins-openai"
pip install -e "./livekit-plugins/livekit-plugins-silero"

