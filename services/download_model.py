from huggingface_hub import snapshot_download
import os

# The ID of the model repository on the Hugging Face Hub
model_id = "RaguTeam/RAGU-lm"
# The target directory to save the model files, relative to this script
local_dir = "ragu-lm"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading model {model_id} to {local_dir}...")

# Download the model snapshot
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Model downloaded successfully!")
