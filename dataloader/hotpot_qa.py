from .dataloader import DataLoader
from . import dataloader_registry
import json
import re


@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-comm")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-comm-internal")
class HotpotQALoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            data = json.load(f)
            for example in data['examples']:
                self.examples.append(
                    {
                        "input": "Relevant Context:\n" + example["supporting_paragraphs"] + "\n\nQuestion: " + example["question"],
                        "answer": example["answer"],
                    }
                )

@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-comm-unrelated")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-comm-unrelated-internal")
class HotpotQAUnrelatedLoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            data = json.load(f)
            for example in data['examples']:
                self.examples.append(
                    {
                        "input": (
                            "Context:\n" + example["supporting_paragraphs"] + "\n\nQuestion: " + example["question"],
                            "Context:\n" + example["unrelated_paragraphs"] + "\n\nQuestion: " + example["question"]
                        ),
                        "answer": example["answer"],
                    }
                )
