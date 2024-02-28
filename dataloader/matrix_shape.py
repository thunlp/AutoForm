from .dataloader import DataLoader
from . import dataloader_registry
import json
import re
import random


@dataloader_registry.register("tasksolving/evaluating_information_essentiality/autoform-gemini.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/autoform-gpt-3.5.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/autoform-gpt-4.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/cot-gemini.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/cot-gpt-3.5.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/cot-gpt-4.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-instance-gemini.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-instance-gpt-3.5.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-instance-gpt-4.py")
class MatrixShapeLoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            data = json.load(f)
            for example in data['examples']:
                self.examples.append(
                    {
                        "input": example['input'],
                        "answer": example['target'],
                    }
                )


@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-task-gemini.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-task-gpt-3.5.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-task-gpt-3.5-gpt-4.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-task-gpt-4.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-task-gpt-4-gemini.py")
@dataloader_registry.register("tasksolving/evaluating_information_essentiality/twostep-task-gpt-4-gpt-3.5.py")
class MatrixShapeManyTaskDescriptionLoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        tasks = []
        random.seed(0)
        with open(self.path) as f:
            data = json.load(f)
            for example in data['examples']:
                tasks.append(
                    {
                        "input": example['input'],
                        "answer": example['target'],
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


