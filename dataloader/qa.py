from .dataloader import DataLoader
from . import dataloader_registry
import json
import re
from random import shuffle
from copy import deepcopy


@dataloader_registry.register("tasksolving/search_qa/gpt-3.5-cot-0301")
@dataloader_registry.register("tasksolving/search_qa/gpt-3.5-cot-model-0301")
@dataloader_registry.register("tasksolving/search_qa/gpt-4-cot")
@dataloader_registry.register("tasksolving/search_qa/gpt-4-cot-model")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-3.5-cot")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-3.5-cot-model")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-cot")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-cot-model")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-cot-model-2")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-cot-model-kqml")
@dataloader_registry.register("tasksolving/hotpot_qa/gpt-4-cot-model-json")
@dataloader_registry.register("tasksolving/wiki_hop_qa/gpt-4-cot")
@dataloader_registry.register("tasksolving/wiki_hop_qa/gpt-3.5-cot")
@dataloader_registry.register("tasksolving/wiki_hop_qa/gpt-4-cot-model")
@dataloader_registry.register("tasksolving/wiki_hop_qa/gpt-3.5-cot-model")
@dataloader_registry.register("tasksolving/wiki_hop_qa/gpt-4-cot-model-kqml")
@dataloader_registry.register("tasksolving/wiki_hop_qa/gpt-4-cot-model-json")
@dataloader_registry.register("tasksolving/narrative_qa/gpt-4-cot")
@dataloader_registry.register("tasksolving/narrative_qa/gpt-3.5-cot")
@dataloader_registry.register("tasksolving/narrative_qa/gpt-4-cot-model")
@dataloader_registry.register("tasksolving/narrative_qa/gpt-3.5-cot-model")
@dataloader_registry.register("tasksolving/narrative_qa/gpt-4-cot-model-kqml")
@dataloader_registry.register("tasksolving/narrative_qa/gpt-4-cot-model-json")
class QALoader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                example = json.loads(line)
                # contexts = [
                #     context
                #     for context in example["context"]
                #     if context is not None and len(context) > 0
                # ]
                # shuffle(contexts)
                # contexts = [contexts[i::2] for i in range(2)]
                # inputs = tuple(
                #     [
                #         "# Context\n"
                #         + "\n".join(
                #             [
                #                 "- " + context.replace("\n", "")
                #                 for context in contexts[i]
                #             ]
                #         )
                #         + "\n\n# Clue (Prompt)\n"
                #         + example["input"]
                #         for i in range(2)
                #     ]
                # )
                self.examples.append(
                    {
                        "input": example["input"],
                        "answer": example["answer"],
                        "context": example["context"],
                    }
                )
