#!/bin/bash

echo "📦 Downloading model to .cache before app starts..."
mkdir -p .cache
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sentence-transformers/paraphrase-MiniLM-L6-v2', cache_dir='.cache')
"
echo "✅ Model cached!"
