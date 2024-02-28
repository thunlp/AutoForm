import random

from .dataloader import DataLoader
from . import dataloader_registry
import json
import re


@dataloader_registry.register("tasksolving/coin_flip/autoform-gemini.py")
@dataloader_registry.register("tasksolving/coin_flip/autoform-gpt-3.5.py")
@dataloader_registry.register("tasksolving/coin_flip/autoform-gpt-4.py")
@dataloader_registry.register("tasksolving/coin_flip/cot-gemini.py")
@dataloader_registry.register("tasksolving/coin_flip/cot-gpt-3.5.py")
@dataloader_registry.register("tasksolving/coin_flip/cot-gpt-4.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-instance-gemini.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-instance-gpt-3.5.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-instance-gpt-4.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/autoform-gemini.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/autoform-gpt-3.5.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/autoform-gpt-4.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/cot-gemini.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/cot-gpt-3.5.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/cot-gpt-4.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-instance-gemini.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-instance-gpt-3.5.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-instance-gpt-4.py")
class StandardDataloader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                self.examples.append(
                    {
                        "input": line["input"],
                        "answer": line["answer"],
                    }
                )


@dataloader_registry.register("tasksolving/coin_flip/twostep-task-gemini.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-task-gpt-3.5.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-task-gpt-3.5-gpt-4.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-task-gpt-4.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-task-gpt-4-gemini.py")
@dataloader_registry.register("tasksolving/coin_flip/twostep-task-gpt-4-gpt-3.5.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-task-gemini.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-task-gpt-3.5.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-task-gpt-3.5-gpt-4.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-task-gpt-4.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-task-gpt-4-gemini.py")
@dataloader_registry.register("tasksolving/minute_mysteries_qa/twostep-task-gpt-4-gpt-3.5.py")
class StandardDataloader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        tasks = []
        random.seed(0)
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                tasks.append(
                    {
                        "input": line["input"],
                        "answer": line["answer"],
                    }
                )
        for task in tasks:
            self.examples.append(
                {
                    "input": (
                        "\n---\n".join(
                            [t['input'] for t in random.sample(tasks, k=5)]
                        ),
                        task["input"],
                    ),
                    "answer": task["answer"],
                }
            )
