base_model=""Qwen/Qwen2.5-32B-Instruct"

python -m wandb login

python load_model.py ${base_model}
