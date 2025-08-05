#!/bin/bash
# filepath: /home/x_lab/workspace/roboracer/delete_empty_folders.sh

# Specify the directory to clean up (default is the current directory)
TARGET_DIR=${1:-.}

# Find and delete all empty directories
find "$TARGET_DIR" -type d -empty -delete

echo "Empty folders in '$TARGET_DIR' have been deleted."