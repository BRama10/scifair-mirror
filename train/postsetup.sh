base_model="Qwen/Qwen-7B"

python -m wandb login

python load_model.py ${base_model}