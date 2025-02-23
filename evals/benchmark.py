import asyncio
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

from .llm import inference


DATASET = [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Name a programming language that starts with 'P'.", "answer": "Python"},
    {"question": "What's the color of the sky on a clear day?", "answer": "blue"},
]


async def process_item(item):
    question = item["question"]
    expected_answer = item["answer"]
    
    predicted_answer = await inference(model_name="gpt-4o", prompt=question)
    
    is_correct = (predicted_answer.strip().lower() == expected_answer.strip().lower())
    
    return {
        "question": question,
        "expected": expected_answer,
        "predicted": predicted_answer,
        "correct": is_correct
    }


async def benchmark(dataset, concurrency: int = 5):
    """
    Benchmark model performance on a dataset with a given concurrency limit.
    
    :param dataset: list of dicts, each with "question" and "answer"
    :param concurrency: max number of concurrent requests
    :return: list of results
    """
    results = []
    console = Console()
    semaphore = asyncio.Semaphore(concurrency)

    async def sem_process(item):
        async with semaphore:
            return await process_item(item)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Benchmarking...", total=len(dataset))
        
        # Create tasks explicitly instead of passing coroutines
        tasks = [asyncio.create_task(sem_process(item)) for item in dataset]
        pending = set(tasks)

        while pending:
            # Wait for completed tasks
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                result = await task
                results.append(result)
                progress.update(task_id, advance=1)

    # Compute accuracy
    total_correct = sum(r["correct"] for r in results)
    accuracy = total_correct / len(dataset) if dataset else 0.0

    # Print final results & accuracy
    console.print("\n[bold green]Benchmark complete![/bold green]")
    console.print(f"Total questions: {len(dataset)}")
    console.print(f"Accuracy: {accuracy*100:.2f}%")

    return results


def main():
    results = asyncio.run(benchmark(DATASET, concurrency=3))
    
    # If you want to inspect all predictions:
    for item in results:
        print(item)


if __name__ == "__main__":
    main()