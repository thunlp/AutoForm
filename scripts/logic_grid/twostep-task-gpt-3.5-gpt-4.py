import logging
import os
import json
import re
import shutil

# from agentverse.agentverse import AgentVerse
from agentverse.tasksolving import TaskSolving
from agentverse.logging import get_logger
from argparse import ArgumentParser
import asyncio
from dataloader import dataloader_registry

parser = ArgumentParser()

parser.add_argument("--task", type=str, default="tasksolving/logic_grid/twostep-task-gpt-3.5-gpt-4")
parser.add_argument(
    "--tasks_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "..", "agentverse", "tasks"),
)
parser.add_argument("--dataset_path", type=str, default="data/logic_grid/logic_grid_puzzle_200.jsonl")
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--has_tools", action="store_true")
parser.add_argument("--tool_tmp_path", type=str)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


logger = get_logger()
logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def get_dataloader(task, dataset_path):
    return dataloader_registry.build(task, path=dataset_path)


def check_answer(response, label):
    ans = re.findall(r'the answer is: *\*?\*? *\n* *(\d+)', response, re.IGNORECASE)
    if len(ans) == 0:
        return False
    ret_res = label[0] == ans[-1]
    if not ret_res:
        print(f"Label: {label}")
        print(f"Answer: {ans[-1]}")
    return ret_res


def cli_main():
    res = []

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
    if not args.overwrite and os.path.exists(f"{args.output_path}/results.jsonl"):
        with open(f"{args.output_path}/results.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    res.append(check_answer(data["response"], data["label"]))
                    skip_cnt += 1
    print(f"Skip {skip_cnt} examples")
    f = open(f"{args.output_path}/results.jsonl", "w" if args.overwrite else "a")
    for i, example in enumerate(dataloader):
        if i < skip_cnt:
            continue
        if res:
            print(f"Accuracy: {sum(res) / len(res)}, total: {len(res)}")
        logger.info(f"Input: {example['input']}\nAnswer: {example['answer']}")
        if args.has_tools:
            assert args.tool_tmp_path is not None
            with open(args.tool_tmp_path, "w") as f:
                f.write(json.dumps(example["tools"]))
        agentverse = TaskSolving.from_task(args.task, args.tasks_dir)
        agentverse.environment.set_task_description(example["input"])
        # print(args.single_agent)
        # print(args.discussion_mode)
        # exit()
        plan, result, logs = agentverse.run()
        res.append(check_answer(plan, example["answer"]))
        f.write(
            json.dumps(
                {
                    "input": example["input"],
                    "response": plan,
                    "label": example["answer"],
                    "logs": logs,
                    "costs": agentverse.environment.get_spend(),
                }
            )
            + "\n"
        )
        f.flush()
    f.close()
    print(f"Accuracy: {sum(res) / len(res)}, total: {len(res)}")


if __name__ == "__main__":
    cli_main()
