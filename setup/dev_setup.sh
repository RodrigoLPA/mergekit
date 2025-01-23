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

# Install Lazygit if not already installed
if ! command_exists lazygit; then
    print_status "Installing Lazygit"
    LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*')
    curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
    tar xf lazygit.tar.gz lazygit
    sudo install lazygit /usr/local/bin
    rm -f lazygit.tar.gz lazygit
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
pip install -e .[evolve]
deactivate

echo "✅ Installation complete!"
