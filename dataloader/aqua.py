from .dataloader import DataLoader
from . import dataloader_registry
import json
import re
import random


@dataloader_registry.register("tasksolving/aqua/cot-gpt-4")
@dataloader_registry.register("tasksolving/aqua/autoform-gpt-4")
@dataloader_registry.register("tasksolving/aqua/cot-gpt-3.5")
@dataloader_registry.register("tasksolving/aqua/autoform-gpt-3.5")
@dataloader_registry.register("tasksolving/aqua/twostep-instance-gpt-3.5.py")
@dataloader_registry.register("tasksolving/aqua/twostep-instance-gpt-4.py")
@dataloader_registry.register("tasksolving/aqua/twostep-instance-gemini.py")
@dataloader_registry.register("tasksolving/aqua/cot-gemini")
@dataloader_registry.register("tasksolving/aqua/autoform-gemini")
class AquaDataloader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                # choices = ""
                # for i, choice in enumerate(line["options"]):
                #     choices += f"{choice}\n"
                choices = "\n".join(line["options"])
                question = f"{line['question']}\n\nChoices:\n{choices.strip()}"
                self.examples.append(
                    {
                        "input": question,
                        "answer": line["correct"],
                    }
                )


@dataloader_registry.register("tasksolving/aqua/twostep-task-gpt-3.5.py")
@dataloader_registry.register("tasksolving/aqua/twostep-task-gpt-4.py")
@dataloader_registry.register("tasksolving/aqua/twostep-task-gemini.py")
@dataloader_registry.register("tasksolving/aqua/twostep-task-gpt-4-gpt-3.5.py")
@dataloader_registry.register("tasksolving/aqua/twostep-task-gpt-4-gemini.py")
@dataloader_registry.register("tasksolving/aqua/twostep-task-gpt-3.5-gpt-4.py")
class AquaManyTaskDescriptionDataloader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        random.seed(0)
        tasks = []
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                # choices = ""
                # for i, choice in enumerate(line["options"]):
                #     choices += f"{choice}\n"
                choices = "\n".join(line["options"])
                question = f"{line['question']}\n\nChoices:\n{choices.strip()}"
                tasks.append(
                    {
                        "input": question,
                        "answer": line["correct"],
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

