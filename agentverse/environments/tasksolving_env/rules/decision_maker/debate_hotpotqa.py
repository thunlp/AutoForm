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


@decision_maker_registry.register("debate_hotpotqa")
class DebateHotpotqaDecisionMaker(BaseDecisionMaker):
    """
    Discuss in a debate manner.
    """

    name: str = "debate_hotpotqa"
    max_turns: int = 20

    async def astep(
            self,
            agents: List[BaseAgent],
            task_description: str,
            previous_plan: str = "No solution yet.",
            advice: str = "No advice yet.",
            **kwargs,
    ) -> List[SolverMessage]:
        # assert (
        #     len(agents) == 3
        # ), "Debate Decision Maker only works with 2 critic agents."
        if advice != "No advice yet.":
            self.broadcast_messages(
                agents, [Message(content=advice, sender="Evaluator")]
            )
        cnt = 0
        # all_roles = "\n".join(
        #     [f"{agent.name}: {agent.role_description}" for agent in agents[1:]]
        # )
        agree_num = 0
        last_answer = None
        all_messages = []
        for agent_id, agent in enumerate(cycle(agents[1:])):
            agent_id %= len(agents[1:])
            all_roles = ", ".join(
                [agent.name for id, agent in enumerate(agents[1:]) if id != agent_id]
            )
            message: CriticMessage = await agent.astep(
                previous_plan, advice, task_description[agent_id], all_roles
            )

            logger.info(
                "",
                f"[{message.sender}]: {message.content}",
                Fore.YELLOW,
            )
            self.broadcast_messages(agents[1:], [message])
            cnt += 1
            if message.is_agree and (
                    re.findall(r"Answer: ?(.+)", message.content)[-1] == last_answer or (last_answer is None)):
                last_answer = re.findall(r"Answer: ?(.+)", message.content)[-1]
                agree_num += 1
            else:
                last_answer = None
                agree_num = 0
            all_messages.append(SolverMessage(content=message.content))
            if cnt == self.max_turns or agree_num == len(agents) - 1:
                break

        return all_messages

    def reset(self):
        pass
