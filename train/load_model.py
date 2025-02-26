from huggingface_hub import login, snapshot_download
import os
import sys

def download_model(repo_id):
    # Set environment variable for fast transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Login (this will prompt for your token or you can pass it directly)
    login(token=os.getenv('HF_API_KEY'))
    
    # Download the entire repository
    repo_path = snapshot_download(repo_id=repo_id)
    
    print(f"Repository downloaded to: {repo_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_model.py <repo_id>")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    download_model(repo_id)