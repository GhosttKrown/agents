# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, MutableSet

import httpx
from livekit.agents import llm

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice

from .log import logger
from .models import (
    ChatModels,
    GroqChatModels,
    OctoChatModels,
    PerplexityChatModels,
    TogetherChatModels,
)
from .utils import AsyncAzureADTokenProvider, build_oai_message

import aiohttp
from livekit.agents import llm
from typing import Any
import logging
import asyncio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DIFYLLM(llm.LLM):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1",
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp.ClientSession()
        self._conversation_id = None

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> "DIFYLLMStream":
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        last_user_message = next(
            (msg for msg in reversed(chat_ctx.messages) if msg.role == "user"), None
        )

        if last_user_message is None:
            raise ValueError("No user message found in chat context")

        data = {
            "inputs": {},
            "query": last_user_message.content,
            "response_mode": "streaming",
            "conversation_id": self._conversation_id,
            "user": "agent",
        }

        if temperature is not None:
            data["temperature"] = temperature

        logger.debug(f"Sending request to Dify: {data}")
        logger.debug(f"Using API key: {self._api_key[:5]}...{self._api_key[-5:]}")
        logger.debug(f"Using base URL: {self._base_url}")
        logger.debug(f"Headers: {headers}")

        try:
            response = await self._session.post(
                f"{self._base_url}/chat-messages",
                headers=headers,
                json=data,
                timeout=30,
            )
            logger.debug(f"Received response from Dify: status={response.status}")
            logger.debug(f"Response headers: {response.headers}")

            if response.status != 200:
                error_content = await response.text()
                logger.error(f"Dify API error: {response.status} - {error_content}")
                raise Exception(f"Dify API error: {response.status} - {error_content}")

            return DIFYLLMStream(
                response=response, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, llm=self
            )
        except aiohttp.ClientError as e:
            logger.error(f"Network error in DIFYLLM.chat: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DIFYLLM.chat: {e}")
            raise

    async def aclose(self) -> None:
        await self._session.close()


class DIFYLLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        response: aiohttp.ClientResponse,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        llm: DIFYLLM,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._response = response
        self._buffer = ""
        self._llm = llm
        self._tool_call_in_progress = False
        self._tool_call_result = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> llm.ChatChunk:
        try:
            async for line in self._response.content:
                line = line.decode("utf-8").strip()
                logger.debug(f"Raw line from Dify: {line}")

                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    logger.debug(f"Parsed data from Dify: {data}")

                    event = data.get("event")
                    if event == "agent_message":
                        content = data.get("answer", "")
                        if content:
                            # Update conversation_id if present
                            if "conversation_id" in data:
                                self._llm._conversation_id = data["conversation_id"]
                                logger.debug(
                                    f"Updated conversation_id: {self._llm._conversation_id}"
                                )
                            return llm.ChatChunk(
                                choices=[
                                    llm.Choice(
                                        delta=llm.ChoiceDelta(
                                            role="assistant",
                                            content=content,
                                        ),
                                        index=0,
                                    )
                                ]
                            )
                    elif event == "agent_thought":
                        logger.debug(f"Agent thought: {data.get('thought', '')}")
                    elif event == "tool_call":
                        self._tool_call_in_progress = True
                        logger.debug(f"Tool call: {data.get('tool', '')}")
                        # Here you might want to yield a chunk indicating a tool is being called
                        return llm.ChatChunk(
                            choices=[
                                llm.Choice(
                                    delta=llm.ChoiceDelta(
                                        role="assistant",
                                        content="[Tool being called...]",
                                    ),
                                    index=0,
                                )
                            ]
                        )
                    elif event == "tool_output":
                        self._tool_call_in_progress = False
                        self._tool_call_result = data.get("output", "")
                        logger.debug(f"Tool output: {self._tool_call_result}")
                        # Here you might want to yield a chunk with the tool output
                        return llm.ChatChunk(
                            choices=[
                                llm.Choice(
                                    delta=llm.ChoiceDelta(
                                        role="assistant",
                                        content=f"[Tool output: {self._tool_call_result}]",
                                    ),
                                    index=0,
                                )
                            ]
                        )
                    elif event == "error":
                        logger.error(f"Error from Dify: {data.get('data', '')}")
                        raise Exception(f"Dify API error: {data.get('data', '')}")
                    elif event == "message_end":
                        logger.debug("Received message_end event from Dify")
                        raise StopAsyncIteration
        except asyncio.CancelledError:
            logger.warning("Stream reading was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in DIFYLLMStream.__anext__: {e}")
            raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._response and not self._response.closed:
            await self._response.release()
        logger.debug("Closed DIFYLLMStream")


@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
        client: openai.AsyncClient | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """

        # throw an error on our end
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key is required")

        self._opts = LLMOptions(model=model, user=user)
        self._client = client or openai.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=5.0,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()

    @staticmethod
    def with_azure(
        *,
        model: str | ChatModels = "gpt-4o",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
    ) -> LLM:
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """

        azure_client = openai.AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
        )  # type: ignore

        return LLM(model=model, client=azure_client, user=user)

    @staticmethod
    def with_fireworks(
        *,
        model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
        api_key: str | None = None,
        base_url: str | None = "https://api.fireworks.ai/inference/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
    ) -> LLM:
        """
        Create a new instance of Fireworks LLM.

        ``api_key`` must be set to your Fireworks API key, either using the argument or by setting
        the ``FIREWORKS_API_KEY`` environmental variable.
        """

        # shim for not using OPENAI_API_KEY
        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if api_key is None:
            raise ValueError("Fireworks API key is required")

        return LLM(
            model=model, api_key=api_key, base_url=base_url, client=client, user=user
        )

    @staticmethod
    def with_groq(
        *,
        model: str | GroqChatModels = "llama3-8b-8192",
        api_key: str | None = None,
        base_url: str | None = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
    ) -> LLM:
        """
        Create a new instance of Groq LLM.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
        """

        # shim for not using OPENAI_API_KEY
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("Groq API key is required")

        return LLM(
            model=model, api_key=api_key, base_url=base_url, client=client, user=user
        )

    @staticmethod
    def with_octo(
        *,
        model: str | OctoChatModels = "llama-2-13b-chat",
        api_key: str | None = None,
        base_url: str | None = "https://text.octoai.run/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
    ) -> LLM:
        """
        Create a new instance of OctoAI LLM.

        ``api_key`` must be set to your OctoAI API key, either using the argument or by setting
        the ``OCTOAI_TOKEN`` environmental variable.
        """

        # shim for not using OPENAI_API_KEY
        api_key = api_key or os.environ.get("OCTOAI_TOKEN")
        if api_key is None:
            raise ValueError("OctoAI API key is required")

        return LLM(
            model=model, api_key=api_key, base_url=base_url, client=client, user=user
        )

    @staticmethod
    def with_ollama(
        *,
        model: str = "llama3.1",
        base_url: str | None = "http://localhost:11434/v1",
        client: openai.AsyncClient | None = None,
    ) -> LLM:
        """
        Create a new instance of Ollama LLM.
        """

        return LLM(model=model, api_key="ollama", base_url=base_url, client=client)

    @staticmethod
    def with_perplexity(
        *,
        model: str | PerplexityChatModels = "llama-3.1-sonar-small-128k-chat",
        api_key: str | None = None,
        base_url: str | None = "https://api.perplexity.ai",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
    ) -> LLM:
        return LLM(
            model=model, api_key=api_key, base_url=base_url, client=client, user=user
        )

    @staticmethod
    def with_together(
        *,
        model: str | TogetherChatModels = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: str | None = None,
        base_url: str | None = "https://api.together.xyz/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
    ) -> LLM:
        """
        Create a new instance of TogetherAI LLM.

        ``api_key`` must be set to your TogetherAI API key, either using the argument or by setting
        the ``TOGETHER_API_KEY`` environmental variable.
        """

        # shim for not using OPENAI_API_KEY
        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError("TogetherAI API key is required")

        return LLM(
            model=model, api_key=api_key, base_url=base_url, client=client, user=user
        )

    @staticmethod
    def create_azure_client(
        *,
        model: str | ChatModels = "gpt-4o",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
    ) -> LLM:
        logger.warning("This alias is deprecated. Use LLM.with_azure() instead")
        return LLM.with_azure(
            model=model,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            user=user,
        )

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream":
        opts: dict[str, Any] = dict()
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            fncs_desc = []
            for fnc in fnc_ctx.ai_functions.values():
                fncs_desc.append(llm._oai_api.build_oai_function_description(fnc))

            opts["tools"] = fncs_desc

            if fnc_ctx and parallel_tool_calls is not None:
                opts["parallel_tool_calls"] = parallel_tool_calls

        user = self._opts.user or openai.NOT_GIVEN

        messages = _build_oai_context(chat_ctx, id(self))
        cmp = self._client.chat.completions.create(
            messages=messages,
            model=self._opts.model,
            n=n,
            temperature=temperature,
            stream=True,
            user=user,
            **opts,
        )

        return LLMStream(oai_stream=cmp, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        oai_stream: Awaitable[openai.AsyncStream[ChatCompletionChunk]],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_oai_stream = oai_stream
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None

    async def aclose(self) -> None:
        if self._oai_stream:
            await self._oai_stream.close()

        return await super().aclose()

    async def __anext__(self):
        if not self._oai_stream:
            self._oai_stream = await self._awaitable_oai_stream

        async for chunk in self._oai_stream:
            for choice in chunk.choices:
                chat_chunk = self._parse_choice(choice)
                if chat_chunk is not None:
                    return chat_chunk

        raise StopAsyncIteration

    def _parse_choice(self, choice: Choice) -> llm.ChatChunk | None:
        delta = choice.delta

        if delta.tool_calls:
            # check if we have functions to calls
            for tool in delta.tool_calls:
                if not tool.function:
                    continue  # oai may add other tools in the future

                call_chunk = None
                if self._tool_call_id and tool.id and tool.id != self._tool_call_id:
                    call_chunk = self._try_run_function(choice)

                if tool.function.name:
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._fnc_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._fnc_raw_arguments += tool.function.arguments  # type: ignore

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason == "tool_calls":
            # we're done with the tool calls, run the last one
            return self._try_run_function(choice)

        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
                    index=choice.index,
                )
            ]
        )

    def _try_run_function(self, choice: Choice) -> llm.ChatChunk | None:
        if not self._fnc_ctx:
            logger.warning("oai stream tried to run function without function context")
            return None

        if self._tool_call_id is None:
            logger.warning(
                "oai stream tried to run function but tool_call_id is not set"
            )
            return None

        if self._fnc_name is None or self._fnc_raw_arguments is None:
            logger.warning(
                "oai stream tried to call a function but raw_arguments and fnc_name are not set"
            )
            return None

        fnc_info = llm._oai_api.create_ai_function_info(
            self._fnc_ctx, self._tool_call_id, self._fnc_name, self._fnc_raw_arguments
        )
        self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=[fnc_info]),
                    index=choice.index,
                )
            ]
        )


def _build_oai_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[ChatCompletionMessageParam]:
    return [build_oai_message(msg, cache_key) for msg in chat_ctx.messages]  # type: ignore
