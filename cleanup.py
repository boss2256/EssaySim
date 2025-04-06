import os
import shutil

model_dir = ".cache/sentence-transformers/all-MiniLM-L6-v2"

# Files to KEEP
keep_files = {
    "config.json", "modules.json", "pytorch_model.bin", "model.safetensors",
    "sentence_bert_config.json", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json", "vocab.txt"
}

# Folders to REMOVE
remove_dirs = ["onnx", "openvino", "1_Pooling"]
remove_files = ["README.md", "train_script.py"]

# Delete extra folders
for d in remove_dirs:
    folder_path = os.path.join(model_dir, d)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"🗑️ Deleted folder: {folder_path}")

# Delete unnecessary files
for fname in remove_files:
    fpath = os.path.join(model_dir, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        print(f"🗑️ Deleted file: {fpath}")

# Double-check and remove anything not in keep_files
for file in os.listdir(model_dir):
    if os.path.isfile(os.path.join(model_dir, file)) and file not in keep_files:
        os.remove(os.path.join(model_dir, file))
        print(f"🗑️ Removed: {file}")
