import logging
import json
import ast
import os

import numpy as np
from aiohttp import ClientSession
from typing import Dict, List, Optional, Union, Iterable
from tenacity import retry, stop_after_attempt, wait_exponential

from pydantic import BaseModel, Field, validator

from agentverse.llms.base import LLMResult
from agentverse.logging import logger
from agentverse.message import Message

from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs
from .utils.jsonrepair import JsonRepair

import google.generativeai as genai
from google.generativeai.types import content_types
from google.ai.generativelanguage_v1beta import GenerateContentResponse

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class GeminiChatArgs(BaseModelArgs):
    candidate_count: int | None = None
    stop_sequences: Iterable[str] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = 1
    top_p: float | None = None
    top_k: int | None = None


model: genai.GenerativeModel = genai.GenerativeModel('gemini-pro')


@llm_registry.register("gemini")
class GeminiChat(BaseChatModel):
    args: GeminiChatArgs = Field(default_factory=GeminiChatArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        global model
        model = genai.GenerativeModel(kwargs.pop("model", 'gemini-pro'))

        args = GeminiChatArgs()
        args = args.dict()
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def generate_response(
            self,
            prepend_prompt: str = "",
            history: List[dict] = [],
            append_prompt: str = "",
            functions: List[dict] = [],
            flatten: bool = False,
    ) -> LLMResult:
        messages = self.construct_messages(
            prepend_prompt, history, append_prompt, flatten
        )
        logger.log_prompt(messages)
        try:
            response = model.generate_content(
                contents=messages,
                generation_config=genai.GenerationConfig(**self.args.dict())
            )
            try:
                if (response.prompt_feedback.block_reason !=
                        GenerateContentResponse.PromptFeedback.BlockReason.BLOCK_REASON_UNSPECIFIED):
                    logger.warn(f"Block reason: {response.prompt_feedback.block_reason}")
                    result_text = ""
                else:
                    result_text = response.text
            except ValueError:
                result_text = "\n".join(response.parts)
            return LLMResult(
                content=result_text,
                send_tokens=0,
                recv_tokens=0,
                total_tokens=0,
            )
        except (Exception, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            logger.error(error)
            raise

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def agenerate_response(
            self,
            prepend_prompt: str = "",
            history: List[dict] = [],
            append_prompt: str = "",
            functions: List[dict] = [],
            flatten: bool = False,
    ) -> LLMResult:
        messages = self.construct_messages(
            prepend_prompt, history, append_prompt, flatten
        )
        logger.log_prompt(messages)
        try:
            response = model.generate_content(
                contents=messages,
                generation_config=genai.GenerationConfig(**self.args.dict())
            )
            try:
                if (response.prompt_feedback.block_reason !=
                        GenerateContentResponse.PromptFeedback.BlockReason.BLOCK_REASON_UNSPECIFIED):
                    logger.warn(f"Block reason: {response.prompt_feedback.block_reason}")
                    result_text = ""
                else:
                    result_text = response.text
            except ValueError:
                result_text = "\n".join(response.parts)
            return LLMResult(
                content=result_text,
                send_tokens=0,
                recv_tokens=0,
                total_tokens=0,
            )
        except (Exception, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            logger.error(error)
            raise

    def construct_messages(
            self,
            prepend_prompt: str,
            history: List[dict],
            append_prompt: str,
            flatten=False,
    ):
        messages = []
        if not flatten:
            if prepend_prompt != "":
                messages.append({"role": "user", "parts": [prepend_prompt]})
            if len(history) > 0:
                for m in history:
                    messages.append({
                        "role": "user" if m["role"] in ["user", "system"] else "model",
                        "parts": [m["content"]]
                    })
            if append_prompt != "":
                messages.append({"role": "user", "parts": [append_prompt]})
        if flatten:
            content = prepend_prompt
            content += "\n" + "\n".join([m["content"] for m in history])
            if history:
                content += "\n"
            content += append_prompt
            messages.append({"role": "user", "parts": [content]})
        return content_types.to_contents(messages)

    def get_spend(self) -> int:
        return 0
