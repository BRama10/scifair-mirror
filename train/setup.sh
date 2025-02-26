base_model="Qwen/Qwen-7B"

python -m pip install wandb

python -m pip install transformers datasets torch accelerate trl

python -m pip install -U huggingface_hub hf_transfer

export WANDB_API_KEY=$(python env.py WANDB_API_KEY)
export HF_API_KEY=$(python env.py HF_API_KEY)

python -m wandb login

python load_model.py ${base_model}