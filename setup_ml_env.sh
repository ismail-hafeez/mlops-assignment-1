#!/bin/bash

set -e

echo "Updating system..."
sudo apt update -y

echo "Installing Python, pip and venv..."
sudo apt install -y python3 python3-pip python3-venv

VENV_PATH="/mnt/ml-data/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_PATH
else
    echo "Virtual environment already exists. Skipping creation."
fi

echo "Activating virtual environment..."
source $VENV_PATH/bin/activate

echo "Installing ML libraries..."
pip install --upgrade pip
pip install pandas scikit-learn numpy

echo "Setup completed successfully."
