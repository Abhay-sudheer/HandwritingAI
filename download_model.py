
from huggingface_hub import snapshot_download

print("Starting TrOCR model download...")

model_path = snapshot_download(
    repo_id="microsoft/trocr-base-handwritten",
    local_dir="./models/trocr-base-handwritten"
)

print("\nModel downloaded successfully!")
print("Model location:", model_path)
