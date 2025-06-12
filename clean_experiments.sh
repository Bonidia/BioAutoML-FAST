#!/usr/bin/env bash

# Define the base directory
BASE_DIR="App/datasets"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist."
    exit 1
fi

# Find and delete all 'runs' directories under dataset* folders
find "$BASE_DIR" -type d -path "$BASE_DIR/dataset*/runs" -exec echo "Deleting: {}" \; -exec rm -rf {} \;

echo "Cleanup completed."