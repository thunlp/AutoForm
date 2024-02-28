import logging
import os
import json
import time
import shutil
import random
from copy import deepcopy

# from agentverse.agentverse import AgentVerse
from agentverse.utils import AGENT_TYPES
from agentverse.tasksolving import TaskSolving
from agentverse.logging import get_logger
from argparse import ArgumentParser
import asyncio
import threading
from dataloader import dataloader_registry

parser = ArgumentParser()

parser.add_argument(
    "--task", type=str, default="tasksolving/wiki_hop_qa/gpt-3.5-cot-model"
)
parser.add_argument(
    "--tasks_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "..", "agentverse", "tasks"),
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/wiki_hop_qa/test_processed.jsonl",
)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--num_thread", type=int, default=1)
parser.add_argument("--num_examples", type=int, default=0)
parser.add_argument("--tool_tmp_path", type=str)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


logger = get_logger()
logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def get_dataloader(task, dataset_path):
    return dataloader_registry.build(task, path=dataset_path)


def cli_main():
    dataloader = get_dataloader(args.task, args.dataset_path)
    if args.output_path is None:
        os.makedirs(f"./results/{args.task}", exist_ok=True)
        args.output_path = f"./results/{args.task}"
    else:
        os.makedirs(args.output_path, exist_ok=True)
    shutil.copyfile(
        f"{args.tasks_dir}/{args.task}/config.yaml",
        f"{args.output_path}/config.yaml",
    )

    skip_cnt = 0
    exist_input = set()
    if args.overwrite:
        f = open(f"{args.output_path}/results.jsonl", "w")
        f.close()
    if not args.overwrite and os.path.exists(f"{args.output_path}/results.jsonl"):
        with open(f"{args.output_path}/results.jsonl", "r") as f:
            for line in f:
                # if line.strip():
                #     skip_cnt += 1
                exist_input.add(json.loads(line)["input"])
    # f = open(f"{args.output_path}/results.jsonl", "w" if args.overwrite else "a")

    threads = []
    # results = [None] * len(dataloader)
    semaphore = threading.Semaphore(args.num_thread)
    writing_block = threading.Lock()
    # threads.append(
    #     threading.Thread(
    #         target=_write,
    #         args=(
    #             deepcopy(dataloader),
    #             results,
    #             skip_cnt,
    #             f"{args.output_path}/results.jsonl",
    #             args.overwrite,
    #         ),
    #     )
    # )
    for i, example in enumerate(dataloader):
        if i < skip_cnt:
            continue
        t = threading.Thread(
            target=_step,
            args=(
                i,
                semaphore,
                writing_block,
                example,
                # results,
                exist_input,
                f"{args.output_path}/results.jsonl",
            ),
        )
        threads.append(t)
        if i + 1 == args.num_examples:
            break
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _step(index, semaphore, writing_block, example, exist_input, file):
    if example["input"] in exist_input:
        # print(f"skip {index}")
        return
    with semaphore:
        logger.info(f"Input: {example['input']}\nAnswer: {example['answer']}")
        agentverse = TaskSolving.from_task(args.task, args.tasks_dir)
        agentverse.environment.set_task_description(example["input"])
        if "_qa" in args.task:
            random.seed(0)
            agent_num = len(agentverse.environment.agents[AGENT_TYPES.CRITIC])
            contexts = [
                context
                for context in example["context"]
                if context is not None and len(context) > 0
            ]
            random.shuffle(contexts)
            contexts = [contexts[i::agent_num] for i in range(agent_num)]
            contexts = [
                "\n".join(["- " + context.replace("\n", "") for context in contexts[i]])
                for i in range(agent_num)
            ]
            for agent, context in zip(
                agentverse.environment.agents[AGENT_TYPES.CRITIC], contexts
            ):
                agent.knowledge = context

        # print(args.single_agent)
        # print(args.discussion_mode)
        # exit()
        plan, result, logs = agentverse.run()
        # results[index] = [plan, result, logs]
    with writing_block:
        f = open(file, "a")
        f.write(
            json.dumps(
                {
                    "input": example["input"],
                    "response": plan,
                    "label": example["answer"],
                    "logs": logs,
                }
            )
            + "\n"
        )
        f.flush()


def _write(dataloader, results, skip_cnt, file, overwrite):
    f = open(file, "w" if overwrite else "a")
    for i, example in enumerate(dataloader):
        if i < skip_cnt:
            continue
        while True:
            if results[i] is None:
                time.sleep(1)
                continue
            f.write(
                json.dumps(
                    {
                        "input": example["input"],
                        "response": results[i][0],
                        "label": example["answer"],
                        "logs": results[i][2],
                    }
                )
                + "\n"
            )
            f.flush()
            results[i] = None
            break
    f.close()


if __name__ == "__main__":
    cli_main()
