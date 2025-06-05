#!/bin/bash

# Cleanup local workspace - remove large files and temporary data

echo "🧹 Cleaning up workspace..."

# Remove old training runs (keep last 3)
find experiments/*/runs -name "train*" -type d | sort -V | head -n -3 | xargs rm -rf 2>/dev/null || true

# Remove temporary files
rm -rf temp/downloads/*
rm -rf temp/processing/*
rm -rf temp/cache/*

# Remove old logs (keep last 7 days)
find logs/ -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true

# Remove cached Python files
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Show disk usage
echo ""
echo "📊 Current disk usage:"
du -sh datasets/ models/ experiments/ results/ logs/ temp/ 2>/dev/null || true

echo "✅ Cleanup complete!"
