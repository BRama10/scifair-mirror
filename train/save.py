from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    repo_id="openminderai/test",
    folder_path="ckpts/s1-<your_timestamp>",
    commit_message="Upload trained model",
)
