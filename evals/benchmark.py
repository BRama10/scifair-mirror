import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console, Group
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.live import Live
from rich.style import Style
from rich.theme import Theme

from dotenv import load_dotenv
from llm import inference
from prompts import (
    EVAL_SYSTEM_PROMPT,
    EVAL_USER_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
)
from utils import extract_tag_content

load_dotenv()

@dataclass
class DatasetItem:
    index: int
    question: str
    answer: str

@dataclass
class LLMResponse:
    content: str
    tokens: int
    model: str

@dataclass
class BenchmarkResult:
    index: int
    question: str
    is_correct: bool
    generation_time: float
    evaluation_time: float
    timestamp: str
    generation: LLMResponse
    evaluation: LLMResponse

class Dataset:
    def __init__(self, items: List[Dict[str, str]], name: Optional[str] = None):
        self.name = name or "Unnamed Dataset"
        self.items = [
            DatasetItem(index=idx, question=item["question"], answer=item["answer"])
            for idx, item in enumerate(items)
        ]
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> DatasetItem:
        return self.items[idx]
    
    def __iter__(self):
        return iter(self.items)

class BenchmarkManager:
    def __init__(
        self,
        dataset: Dataset,
        generation_concurrency: int = 3,
        evaluation_concurrency: int = 5,
        results_dir: str = "benchmark_results",
        batch_size: int = 10,
        generation_model: str = 'gpt-4o',
        eval_model: str = 'gpt-4o-mini'
    ):
        self.dataset = dataset
        self.generation_semaphore = asyncio.Semaphore(generation_concurrency)
        self.evaluation_semaphore = asyncio.Semaphore(evaluation_concurrency)
        self.batch_size = batch_size
        self.generation_model = generation_model
        self.eval_model = eval_model
        
        # Custom theme for better styling
        custom_theme = Theme({
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red bold",
            "progress.description": "cyan",
            "progress.percentage": "green",
            "table.header": "cyan bold",
        })

        self.console = Console(theme=custom_theme)
        self.results: List[BenchmarkResult] = []
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.results_table = Table(
            show_header=True,
            header_style="table.header",
            show_lines=True,
            expand=True,
            box=None,
            pad_edge=False,
            min_width=60
        )
        self.results_table.add_column("#", style="dim", width=4)
        self.results_table.add_column("Correct", justify="center", width=3)
        self.results_table.add_column("Gen", justify="right", width=8)
        self.results_table.add_column("Eval", justify="right", width=8)

        # Create progress bar that will stay at bottom
        self.progress = Progress(
            SpinnerColumn(style="info"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="info", finished_style="success"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("[success]Acc: {task.fields[accuracy]:.1f}%"),
            TextColumn("[info]Time: {task.fields[time]}s"),
            TextColumn("[info]{task.fields[dataset]}"),
            refresh_per_second=10,
            expand=True,
            console=self.console
        )

    async def generate_answer(self, question: str) -> Tuple[LLMResponse, float]:
        async with self.generation_semaphore:
            start_time = asyncio.get_event_loop().time()
            response = await inference(
                model_name=self.generation_model,
                system=GENERATION_SYSTEM_PROMPT.substitute(),
                user=GENERATION_USER_PROMPT.substitute(question=question)
            )
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                tokens=response.usage.total_tokens,
                model=response.model
            ), generation_time

    async def evaluate_answer(
        self, expected_answer: str, predicted_answer: str
    ) -> Tuple[bool, LLMResponse, float]:
        async with self.evaluation_semaphore:
            start_time = asyncio.get_event_loop().time()
            response = await inference(
                model_name=self.eval_model,
                system=EVAL_SYSTEM_PROMPT.substitute(),
                user=EVAL_USER_PROMPT.substitute(
                    correct_answer=expected_answer,
                    student_answer=predicted_answer
                )
            )
            evaluation_time = asyncio.get_event_loop().time() - start_time
            
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                tokens=response.usage.total_tokens,
                model=response.model
            )
            
            is_correct = 'yes' in [x.lower() for x in extract_tag_content(llm_response.content, 'answer')]
            return is_correct, llm_response, evaluation_time

    async def process_batch(self, batch: List[DatasetItem]) -> List[BenchmarkResult]:
        generation_tasks = [self.generate_answer(item.question) for item in batch]
        generation_results = await asyncio.gather(*generation_tasks)
        
        evaluation_tasks = [
            self.evaluate_answer(item.answer, gen_response.content)
            for item, (gen_response, _) in zip(batch, generation_results)
        ]
        evaluation_results = await asyncio.gather(*evaluation_tasks)
        
        return [
            BenchmarkResult(
                index=item.index,
                question=item.question,
                is_correct=is_correct,
                generation_time=gen_time,
                evaluation_time=eval_time,
                timestamp=datetime.now().isoformat(),
                generation=gen_response,
                evaluation=eval_response
            )
            for item, (gen_response, gen_time), (is_correct, eval_response, eval_time)
            in zip(batch, generation_results, evaluation_results)
        ]

    def save_results(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        
        summary = {
            "dataset_name": self.dataset.name,
            "timestamp": timestamp,
            "total_questions": len(self.dataset),
            "accuracy": self.calculate_current_accuracy(),
            "avg_generation_time": sum(r.generation_time for r in self.results) / len(self.results),
            "avg_evaluation_time": sum(r.evaluation_time for r in self.results) / len(self.results),
            "results": [
                {
                    "index": r.index,
                    "question": r.question,
                    "is_correct": r.is_correct,
                    "generation": {
                        "content": r.generation.content,
                        "tokens": r.generation.tokens,
                        "model": r.generation.model
                    },
                    "evaluation": {
                        "content": r.evaluation.content,
                        "tokens": r.evaluation.tokens,
                        "model": r.evaluation.model
                    },
                    "times": {
                        "generation": r.generation_time,
                        "evaluation": r.evaluation_time
                    }
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return results_file

    def update_results_table(self, result: BenchmarkResult):
        status_style = "success" if result.is_correct else "error"
        self.results_table.add_row(
            str(result.index),
            "✓" if result.is_correct else "✗",
            f"{result.generation_time:.1f}s",
            f"{result.evaluation_time:.1f}s",
            style=status_style if not result.is_correct else None
        )

    async def run(self) -> List[BenchmarkResult]:
        # Clear the screen once at the start
        self.console.clear()
        
        task = self.progress.add_task(
            "Processing",
            total=len(self.dataset),
            accuracy=0.0,
            time=0.0,
            dataset=f"Dataset: {self.dataset.name}"
        )

        start_time = asyncio.get_event_loop().time()
        
        # Create a Live display just for the progress bar
        progress_live = Live(
            self.progress,
            console=self.console,
            refresh_per_second=10,
            transient=True
        )
        
        with progress_live:
            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset[i:i + self.batch_size]
                batch_results = await self.process_batch(batch)
                
                for result in batch_results:
                    self.results.append(result)
                    self.update_results_table(result)
                    
                    # Update progress
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self.progress.update(
                        task,
                        advance=1,
                        accuracy=self.calculate_current_accuracy(),
                        time=f"{elapsed:.1f}"
                    )
                    
                    # Clear screen and redraw everything
                    self.console.clear()
                    self.console.print(self.results_table)
                    
                # Small delay to prevent screen flicker
                await asyncio.sleep(0.05)

        return self.results

    def calculate_current_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.is_correct for r in self.results) / len(self.results) * 100

async def main():
    sample_data = [
        {"question": "What is 1+1?", "answer": "2"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Name a programming language that starts with 'P'.", "answer": "Python"},
        {"question": "What's the color of the sky on a clear day?", "answer": "blue"},
    ]

    dataset = Dataset(sample_data, name="Basic Knowledge Test")
    benchmark = BenchmarkManager(
        dataset=dataset,
        generation_concurrency=1,
        evaluation_concurrency=1,
        batch_size=1  # Smaller batch size for moxre frequent updates

    )
    
    results = await benchmark.run()
    results_file = benchmark.save_results()
    
    console = Console()
    console.print(f"\n[bold green]Benchmark Complete![/bold green]")
    console.print(f"Results saved to: {results_file}")
    console.print(f"Final Accuracy: {benchmark.calculate_current_accuracy():.1f}%")

if __name__ == "__main__":
    asyncio.run(main())