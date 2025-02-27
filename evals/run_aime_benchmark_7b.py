from benchmark import Dataset
from loader import DatasetLoader
from huggingface_hub import login, snapshot_download
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import json

def download_model(repo_id):
    # Set environment variable for fast transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Login (this will prompt for your token or you can pass it directly)
    login(token=os.getenv('HF_API_KEY'))
    
    # Download the entire repository
    repo_path = snapshot_download(repo_id=repo_id)
    
    print(f"Repository downloaded to: {repo_path}")

download_model("thehunter911/test")

dataset_data = DatasetLoader.load_and_cache_dataset('aime24')

dataset = Dataset(dataset_data, name='AIME24')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "thehunter911/test"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def inference(query: str):
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=5012
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# Output JSON file
output_file = f"aime24_responses_{time.strftime('%Y%m%d_%H%M%S')}.json"

# Check if results file exists and load existing results
results = {}
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing results from {output_file}")

# Process each question and save after each one
for item in dataset.items:
    # Skip if already processed
    if str(item.index) in results:
        print(f"Skipping question {item.index} (already processed)")
        continue
        
    print(f"Processing question {item.index}/{len(dataset.items)}...")
    try:
        # Run inference
        response = inference(item.question)
        
        # Save result
        results[str(item.index)] = {
            "question": item.question,
            "response": response
        }
        
        # Update the file after each question
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved response for question {item.index}")
        
    except Exception as e:
        print(f"Error processing question {item.index}: {str(e)}")
        # Save the error in the results
        results[str(item.index)] = {
            "question": item.question,
            "error": str(e)
        }
        
        # Still save to file even on error
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

print(f"Complete! Processed {len(results)} questions. Results saved to {output_file}")