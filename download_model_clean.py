from sentence_transformers import SentenceTransformer
import os

# Define your custom target path
target_dir = ".cache/sentence-transformers_all-MiniLM-L6-v2"

# Download the model (from HF hub)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Save it to the desired .cache directory
model.save(target_dir)
print(f"✅ Model saved to: {target_dir}")
