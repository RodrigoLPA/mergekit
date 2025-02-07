#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
# Exit if any command in a pipeline fails
set -o pipefail
# Error on undefined variables
set -u

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status messages
print_status() {
    echo "📦 $1..."
}

# Update package lists
print_status "Updating package lists"
sudo apt update

# Install tmux if not already installed
if ! command_exists tmux; then
    print_status "Installing tmux"
    sudo apt install -y tmux
fi

# Install neovim if not already installed
if ! command_exists nvim; then
    print_status "Installing neovim"
    sudo apt-get install -y neovim
fi

# Git configuration
print_status "Configuring Git"
if ! command_exists git; then
    sudo apt install -y git
fi
git config --global user.name "roel"
git config --global user.email "rodrigo@macrocosmos.ai"

# Install NCDU if not already installed
if ! command_exists ncdu; then
    print_status "Installing NCDU"
    sudo apt install -y ncdu
fi
# Install htop if not already installed
if ! command_exists htop; then
    print_status "Installing htop"
    sudo apt install -y htop
fi

# Check if Python 3.10 is installed
if ! command_exists python3.10; then
    print_status "Installing Python 3.10"
    sudo apt install -y python3.10 python3.10-venv
fi

# Create and activate virtual environment
print_status "Setting up Python virtual environment"
python3.10 -m venv env
source env/bin/activate

# Install package in development mode
print_status "Installing package in development mode"
pip install -e .[evolve,vllm]
# Reinstall flash-attn to ensure compatibility with vLLM
# print_status "Reinstalling flash-attn"
# pip uninstall -y flash-attn
# pip cache purge
# pip install flash-attn

deactivate

echo "✅ Installation complete!"
