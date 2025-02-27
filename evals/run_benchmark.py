# main.py
import asyncio
import argparse
from rich.console import Console
from benchmark import Dataset, BenchmarkManager
from loader import DatasetLoader
from config import BENCHMARK_CONFIGS
from huggingface_hub import login
import os

from dotenv import load_dotenv
load_dotenv()

def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run benchmarks on various datasets')
    parser.add_argument(
        '--benchmark',
        type=str,
        choices=list(BENCHMARK_CONFIGS.keys()),
        required=True,
        help='Benchmark dataset to use'
    )
    parser.add_argument(
        '--generation-concurrency',
        type=int,
        default=0,
        help='Number of concurrent generation tasks'
    )
    parser.add_argument(
        '--evaluation-concurrency',
        type=int,
        default=0,
        help='Number of concurrent evaluation tasks'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        required=False,
        help='HuggingFace API token',
        default=None
    )
    return parser

async def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Login to HuggingFace
    if args.hf_token:
        login(args.hf_token)
    elif os.getenv('HF_API_TOKEN'):
        login(os.getenv('HF_API_TOKEN'))
    
    # Load dataset
    console = Console()
    console.print(f"\n[bold blue]Loading {args.benchmark} dataset...[/bold blue]")
    dataset_data = DatasetLoader.load_and_cache_dataset(args.benchmark)

    if args.generation_concurrency == 0:
        args.generation_concurrency = args.batch_size

    if args.evaluation_concurrency == 0:
        args.evaluation_concurrency = args.batch_size
    
    # Setup benchmark
    dataset = Dataset(dataset_data, name=args.benchmark.upper())
    benchmark = BenchmarkManager(
        dataset=dataset,
        generation_batch_size=args.generation_concurrency,
        evaluation_batch_size=args.evaluation_concurrency,
        generation_model='thehunter911/test'
    )
    
    # Run benchmark
    console.print("[bold blue]Running benchmark...[/bold blue]")
    await benchmark.run()
    results_file = benchmark.save_results()
    
    # Print results
    console.print(f"\n[bold green]Benchmark Complete![/bold green]")
    console.print(f"Results saved to: {results_file}")
    console.print(f"Final Accuracy: {benchmark.calculate_current_accuracy():.1f}%")

if __name__ == "__main__":
    asyncio.run(main())