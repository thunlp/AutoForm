import random

from .dataloader import DataLoader
from . import dataloader_registry
import json
import re


@dataloader_registry.register("tasksolving/logic_grid/autoform-gemini.py")
@dataloader_registry.register("tasksolving/logic_grid/autoform-gpt-3.5.py")
@dataloader_registry.register("tasksolving/logic_grid/autoform-gpt-4.py")
@dataloader_registry.register("tasksolving/logic_grid/cot-gemini.py")
@dataloader_registry.register("tasksolving/logic_grid/cot-gpt-3.5.py")
@dataloader_registry.register("tasksolving/logic_grid/cot-gpt-4.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-instance-gemini.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-instance-gpt-3.5.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-instance-gpt-4.py")
class LogicGridLoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                question = line["inputs"].strip().strip("A:").strip("Q:")
                # question = re.sub(r"choice: ?\d+", "", question).strip()
                self.examples.append(
                    {
                        "input": question,
                        "answer": line["targets"][0],
                    }
                )


@dataloader_registry.register("tasksolving/logic_grid/twostep-task-gemini.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-task-gpt-3.5.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-task-gpt-3.5-gpt-4.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-task-gpt-4.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-task-gpt-4-gemini.py")
@dataloader_registry.register("tasksolving/logic_grid/twostep-task-gpt-4-gpt-3.5.py")
class LogicGridManyTaskDescriptionLoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        random.seed(0)
        tasks = []
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                question = line["inputs"].strip().strip("A:").strip("Q:")
                # question = re.sub(r"choice: ?\d+", "", question).strip()
                tasks.append(
                    {
                        "input": question,
                        "answer": line["targets"][0],
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
