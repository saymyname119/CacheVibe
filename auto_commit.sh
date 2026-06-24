#!/bin/bash

# Auto-commit script for SemCche project
# Usage: ./auto_commit.sh

cd /home/devansh/SemCche

echo "Starting auto-commit script..."
echo "Will commit every 20-25 minutes and 1 hour"
echo "Press Ctrl+C to stop"

commit_count=0

while true; do
    # Random interval between 20-25 minutes (1200-1500 seconds)
    if [ $((commit_count % 3)) -eq 2 ]; then
        # Every 3rd commit, wait 1 hour (3600 seconds)
        sleep_time=3600
        echo "Waiting 1 hour before next commit..."
    else
        # Random between 20-25 minutes
        sleep_time=$((1200 + RANDOM % 300))
        minutes=$((sleep_time / 60))
        echo "Waiting $minutes minutes before next commit..."
    fi
    
    sleep $sleep_time
    
    # Check if there are changes
    if [[ -n $(git status -s) ]]; then
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        git add .
        git commit -m "Auto-commit: $timestamp"
        echo "✓ Committed at $timestamp"
    else
        echo "No changes to commit"
    fi
    
    commit_count=$((commit_count + 1))
done
