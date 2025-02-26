from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    repo_id="thehunter911/test",
    folder_path="ckpts/s1-<your_timestamp>",
    commit_message="Upload trained model",
)
