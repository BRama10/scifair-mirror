# dataset_loader.py
import json
from typing import List, Dict, Any
from datasets import load_dataset
from copy import deepcopy
import random
from config import BENCHMARK_CONFIGS

class DatasetLoader:
    @staticmethod
    def load_and_cache_dataset(benchmark_name: str) -> List[Dict[str, Any]]:
        config = BENCHMARK_CONFIGS.get(benchmark_name)
        if not config:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        cache_file = config["cache_file"]
        
        # Try to load from cache first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Load from HuggingFace and process
        if benchmark_name == "gpqa-diamond":
            dataset = load_dataset(config["dataset"], config["subset"])[config["split"]]
            processed_data = DatasetLoader._process_gpqa(dataset)
        elif benchmark_name == "aime24":
            dataset = load_dataset(config["dataset"])[config["split"]]
            processed_data = DatasetLoader._process_aime(dataset)
        else:  # math500
            dataset = load_dataset(config["dataset"])[config["split"]]
            processed_data = DatasetLoader._process_math500(dataset)

        # Cache the processed dataset
        with open(cache_file, 'w') as f:
            json.dump(processed_data, f)

        return processed_data

    @staticmethod
    def _process_gpqa(dataset) -> List[Dict[str, Any]]:
        processed_data = []
        for value in dataset:
            problem = value.get('problem')
            correct_ans = value.get('Correct Answer')
            
            answers = [
                value.get('Incorrect Answer 1'),
                value.get('Incorrect Answer 2'),
                value.get('Incorrect Answer 3'),
                deepcopy(correct_ans)
            ]
            
            numbers = list(range(4))
            random.shuffle(numbers)
            
            processed_data.append({
                'question': f"""
                {problem}
                
                Answer Choices:
                (a){answers[numbers[0]]}
                (b){answers[numbers[1]]}
                (c){answers[numbers[2]]}
                (d){answers[numbers[3]]}
                """,
                'answer': correct_ans
            })
        return processed_data

    @staticmethod
    def _process_aime(dataset) -> List[Dict[str, Any]]:
        return [{
            'question': value['question'],
            'answer': value['answer']
        } for value in dataset]

    @staticmethod
    def _process_math500(dataset) -> List[Dict[str, Any]]:
        return [{
            'question': value['problem'],
            'answer': value['answer']
        } for value in dataset]