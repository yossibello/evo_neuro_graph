#!/bin/bash

# Name of the output zip file
OUTPUT="neuro_graph_code_review.zip"

echo "📦 Creating compressed archive: $OUTPUT"
echo ""

# Remove old archive if exists
rm -f "$OUTPUT"

# Zip source files, excluding large or irrelevant dirs
zip -r "$OUTPUT" . \
    -x "*.zip" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x "env/*" \
    -x "venv/*" \
    -x "*.npz" \
    -x "artifacts/*" \
    -x ".git/*" \
    -x ".idea/*" \
    -x ".vscode/*" \
    -x "logs/*" \
    -x "*.log"

echo ""
echo "✅ Done!"
echo "Created: $OUTPUT"
echo "You can now upload it for final review."
