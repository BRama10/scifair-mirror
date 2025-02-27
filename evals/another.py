from benchmark import Dataset
from loader import DatasetLoader
from huggingface_hub import login, snapshot_download
import os
import time
import json
from tqdm import tqdm

# pip install openai

from openai import OpenAI

client = OpenAI(
		base_url = "https://r7pgla1bfg53sntx.us-east-1.aws.endpoints.huggingface.cloud/v1/",
		api_key = "hf_XXXXX"
	)


def download_model(repo_id):
    # Set environment variable for fast transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Login (this will prompt for your token or you can pass it directly)
    login(token=os.getenv('HF_API_KEY'))
    
    # Download the entire repository
    repo_path = snapshot_download(repo_id=repo_id)
    
    print(f"Repository downloaded to: {repo_path}")

# download_model("thehunter911/test")

dataset_data = DatasetLoader.load_and_cache_dataset('aime24')
dataset = Dataset(dataset_data, name='AIME24')
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "thehunter911/test"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)



def inference(query: str, stream_to_console=True):
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": query}
    ]
    
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            *messages
        ],
        top_p=None,
        temperature=0.7,
        max_tokens=32178,
        stream=False,
        seed=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None
    ).choices[0].message.content

# Output JSON file
output_file = f"aime24_responses_{time.strftime('%Y%m%d_%H%M%S')}.json"

# Check if results file exists and load existing results
results = {}
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing results from {output_file}")

# Get total number of items for tqdm
total_items = len(dataset.items)
already_processed = sum(1 for item in dataset.items if str(item.index) in results)
print(f"Already processed: {already_processed}/{total_items}")

# Process each question with tqdm progress bar and save after each one
for item in tqdm(dataset.items, desc="Processing questions", total=total_items, initial=already_processed):
    # Skip if already processed
    if str(item.index) in results:
        continue
        
    tqdm.write(f"Processing question {item.index}/{total_items}...")
    try:
        # Run inference with streaming
        response = inference(item.question, stream_to_console=True)
        
        # Save result
        results[str(item.index)] = {
            "question": item.question,
            "response": response
        }
        
        # Update the file after each question
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        tqdm.write(f"Saved response for question {item.index}")
        
    except Exception as e:
        tqdm.write(f"Error processing question {item.index}: {str(e)}")
        # Save the error in the results
        results[str(item.index)] = {
            "question": item.question,
            "error": str(e)
        }
        
        # Still save to file even on error
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

print(f"Complete! Processed {len(results)} questions. Results saved to {output_file}")