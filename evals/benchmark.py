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
class GenerationResult:
    index: int
    question: str
    generation: LLMResponse
    generation_time: float
    timestamp: str

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
        generation_concurrency: int = 1, # Default to sequential generation
        evaluation_concurrency: int = 5,
        results_dir: str = "benchmark_results",
        generation_batch_size: int = 1, # Default to one at a time
        evaluation_batch_size: int = 50, # Default to 50 at a time
        generation_model: str = 'gpt-4o',
        eval_model: str = 'gpt-4o-mini'
    ):
        self.dataset = dataset
        self.generation_semaphore = asyncio.Semaphore(generation_concurrency)
        self.evaluation_semaphore = asyncio.Semaphore(evaluation_concurrency)
        self.generation_batch_size = generation_batch_size
        self.evaluation_batch_size = evaluation_batch_size
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
        self.generation_results: List[GenerationResult] = []
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

        # Create progress bar for generation phase
        self.generation_progress = Progress(
            SpinnerColumn(style="info"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="info", finished_style="success"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("[info]Time: {task.fields[time]}s"),
            TextColumn("[info]{task.fields[dataset]}"),
            refresh_per_second=10,
            expand=True,
            console=self.console
        )
        
        # Create progress bar for evaluation phase
        self.evaluation_progress = Progress(
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

    async def generate_answer(self, item: DatasetItem) -> GenerationResult:
        async with self.generation_semaphore:
            start_time = asyncio.get_event_loop().time()
            response = await inference(
                model_name=self.generation_model,
                system=GENERATION_SYSTEM_PROMPT.substitute(),
                user=GENERATION_USER_PROMPT.substitute(question=item.question)
            )
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return GenerationResult(
                index=item.index,
                question=item.question,
                generation=LLMResponse(
                    content=response.choices[0].message.content,
                    tokens=response.usage.total_tokens,
                    model=response.model
                ),
                generation_time=generation_time,
                timestamp=datetime.now().isoformat()
            )

    async def evaluate_answer(
        self, 
        item: DatasetItem, 
        gen_result: GenerationResult
    ) -> BenchmarkResult:
        async with self.evaluation_semaphore:
            start_time = asyncio.get_event_loop().time()
            response = await inference(
                model_name=self.eval_model,
                system=EVAL_SYSTEM_PROMPT.substitute(),
                user=EVAL_USER_PROMPT.substitute(
                    correct_answer=item.answer,
                    student_answer=gen_result.generation.content
                )
            )
            evaluation_time = asyncio.get_event_loop().time() - start_time
            
            eval_response = LLMResponse(
                content=response.choices[0].message.content,
                tokens=response.usage.total_tokens,
                model=response.model
            )
            
            is_correct = 'yes' in [x.lower() for x in extract_tag_content(eval_response.content, 'answer')]
            
            return BenchmarkResult(
                index=item.index,
                question=item.question,
                is_correct=is_correct,
                generation_time=gen_result.generation_time,
                evaluation_time=evaluation_time,
                timestamp=datetime.now().isoformat(),
                generation=gen_result.generation,
                evaluation=eval_response
            )

    async def run_generation_phase(self) -> List[GenerationResult]:
        self.console.clear()
        
        gen_task = self.generation_progress.add_task(
            "Generating Answers",
            total=len(self.dataset),
            time=0.0,
            dataset=f"Dataset: {self.dataset.name}"
        )

        start_time = asyncio.get_event_loop().time()
        
        generation_live = Live(
            self.generation_progress,
            console=self.console,
            refresh_per_second=10,
            transient=True
        )
        
        with generation_live:
            for i in range(0, len(self.dataset), self.generation_batch_size):
                batch = self.dataset[i:i + self.generation_batch_size]
                
                # Create and await generation tasks for the current batch
                gen_tasks = [self.generate_answer(item) for item in batch]
                batch_results = await asyncio.gather(*gen_tasks)
                
                # Save generation results
                for result in batch_results:
                    self.generation_results.append(result)
                    
                    # Update progress
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self.generation_progress.update(
                        gen_task,
                        advance=1,
                        time=f"{elapsed:.1f}"
                    )
                
                # Small delay to prevent screen flicker
                await asyncio.sleep(0.05)
        
        self.console.print("[bold green]Generation phase complete![/bold green]")
        return self.generation_results
    
    async def run_evaluation_phase(self) -> List[BenchmarkResult]:
        # Match generation results with dataset items
        paired_data = []
        for gen_result in self.generation_results:
            # Find the corresponding dataset item
            dataset_item = self.dataset[gen_result.index]
            paired_data.append((dataset_item, gen_result))
        
        self.console.clear()
        
        eval_task = self.evaluation_progress.add_task(
            "Evaluating Answers",
            total=len(paired_data),
            accuracy=0.0,
            time=0.0,
            dataset=f"Dataset: {self.dataset.name}"
        )

        start_time = asyncio.get_event_loop().time()
        
        evaluation_live = Live(
            Group(
                self.results_table,
                self.evaluation_progress
            ),
            console=self.console,
            refresh_per_second=10,
            transient=True
        )
        
        with evaluation_live:
            for i in range(0, len(paired_data), self.evaluation_batch_size):
                batch = paired_data[i:i + self.evaluation_batch_size]
                
                # Create and await evaluation tasks for the current batch
                eval_tasks = [self.evaluate_answer(item, gen_result) for item, gen_result in batch]
                batch_results = await asyncio.gather(*eval_tasks)
                
                # Process evaluation results
                for result in batch_results:
                    self.results.append(result)
                    self.update_results_table(result)
                    
                    # Update progress
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self.evaluation_progress.update(
                        eval_task,
                        advance=1,
                        accuracy=self.calculate_current_accuracy(),
                        time=f"{elapsed:.1f}"
                    )
                    
                # Clear screen and redraw everything
                self.console.clear()
                self.console.print(self.results_table)
                
                # Small delay to prevent screen flicker
                await asyncio.sleep(0.05)
        
        self.console.print("[bold green]Evaluation phase complete![/bold green]")
        return self.results

    def update_results_table(self, result: BenchmarkResult):
        status_style = "success" if result.is_correct else "error"
        self.results_table.add_row(
            str(result.index),
            "✓" if result.is_correct else "✗",
            f"{result.generation_time:.1f}s",
            f"{result.evaluation_time:.1f}s",
            style=status_style if not result.is_correct else None
        )

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

    def calculate_current_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.is_correct for r in self.results) / len(self.results) * 100

    async def run(self) -> List[BenchmarkResult]:
        """Run the full benchmark process with decoupled generation and evaluation phases"""
        # Phase 1: Generate all answers
        self.console.print("[bold cyan]Starting Generation Phase[/bold cyan]")
        await self.run_generation_phase()
        
        # Phase 2: Evaluate all answers
        self.console.print("[bold cyan]Starting Evaluation Phase[/bold cyan]")
        await self.run_evaluation_phase()
        
        # Save results
        results_file = self.save_results()
        
        self.console.print(f"\n[bold green]Benchmark Complete![/bold green]")
        self.console.print(f"Results saved to: {results_file}")
        self.console.print(f"Final Accuracy: {self.calculate_current_accuracy():.1f}%")
        
        return self.results

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
        generation_concurrency=1,  # Sequential generation (one at a time)
        evaluation_concurrency=5,  # Parallel evaluation
        generation_batch_size=1,   # Generate one at a time
        evaluation_batch_size=50   # Evaluate in batches of 50
    )
    
    results = await benchmark.run()

if __name__ == "__main__":
    asyncio.run(main())