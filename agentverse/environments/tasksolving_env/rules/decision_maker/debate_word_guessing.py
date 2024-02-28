from __future__ import annotations
import asyncio
import re
from itertools import cycle
from colorama import Fore

from typing import TYPE_CHECKING, List, Union

from . import decision_maker_registry
from .base import BaseDecisionMaker
from agentverse.logging import logger

from agentverse.message import Message, SolverMessage

if TYPE_CHECKING:
    from agentverse.agents import BaseAgent, CriticAgent
    from agentverse.message import CriticMessage


@decision_maker_registry.register("debate-word-guessing")
class DebateWordGuessingDecisionMaker(BaseDecisionMaker):
    """
    Discuss in a debate manner.
    """

    name: str = "debate-word-guessing"
    max_turns: int = 20

    async def astep(
        self,
        agents: List[BaseAgent],
        task_description: str,
        previous_plan: str = "No solution yet.",
        advice: str = "No advice yet.",
        **kwargs,
    ) -> List[SolverMessage]:
        if advice != "No advice yet.":
            self.broadcast_messages(
                agents, [Message(content=advice, sender="Evaluator")]
            )
        cnt = 0
        agree_num = 0
        last_answer = None
        all_messages = []

        for idx in range(self.max_turns):
            turn_id = idx + 1
            question_message: SolverMessage = agents[0].step(turn_id=turn_id)

            logger.info(
                "",
                f"[{question_message.sender}]: {question_message.content}",
                Fore.YELLOW,
            )
            import pdb

            pdb.set_trace()
            self.broadcast_messages(agents, [question_message])

            all_messages.append(SolverMessage(content=question_message.content))
            if re.findall(r"Answer: ?(.+)", question_message.content) != []:
                break

            answer_message: CriticMessage = await agents[1].astep(
                previous_plan, advice, task_description, turn_id=turn_id
            )
            logger.info(
                "",
                f"[{answer_message.sender}]: {answer_message.content}",
                Fore.YELLOW,
            )
            import pdb

            pdb.set_trace()
            self.broadcast_messages(agents, [answer_message])
            all_messages.append(SolverMessage(content=answer_message.content))

        return all_messages

    def reset(self):
        pass
