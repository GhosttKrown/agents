#!/bin/bash
set -e

# Function to ask for user confirmation
confirm() {
    read -r -p "${1:-Are you sure?} [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            true
            ;;
        *)
            false
            ;;
    esac
}

# Check if the virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    if confirm "Do you want to delete the existing environment and create a new one?"; then
        echo "Deleting existing virtual environment..."
        rm -rf venv
        echo "Creating new Python 3.12 virtual environment..."
        python3.12 -m venv venv
    else
        echo "Using existing virtual environment."
    fi
else
    echo "Creating Python 3.12 virtual environment..."
    python3.12 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Failed to activate the virtual environment."
    exit 1
fi

echo "Virtual environment activated successfully."

# Install requirements from the root folder
pip install -r requirements.txt

# Install livekit-agents in editable mode
pip install -e ./livekit-agents --config-settings editable_mode=strict

# Install livekit plugins in editable mode
pip install -e ./livekit-plugins/livekit-plugins-cartesia --config-settings editable_mode=strict
pip install -e ./livekit-plugins/livekit-plugins-deepgram --config-settings editable_mode=strict
pip install -e ./livekit-plugins/livekit-plugins-openai --config-settings editable_mode=strict
pip install -e ./livekit-plugins/livekit-plugins-silero --config-settings editable_mode=strict

echo "All installations completed successfully!"