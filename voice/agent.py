from asyncio.log import logger
import os
import asyncio
import aiohttp
import uvloop
import logging

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant, AssistantTranscriptionOptions
from livekit.plugins import deepgram, silero, cartesia
from livekit.agents import tokenize
from livekit.plugins.openai.llm import DIFYLLM

from dotenv import load_dotenv

load_dotenv()

# Use uvloop for better asyncio performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Create a shared aiohttp session
async def create_client_session():
    connector = aiohttp.TCPConnector(limit=100)  # Adjust the limit as needed
    return aiohttp.ClientSession(connector=connector)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["session"] = asyncio.get_event_loop().run_until_complete(
        create_client_session()
    )


async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="Você é um atendente de portaria remota. atenda pessoas que estão entrando no prédio, deixando encomendas, informando horários de visitas, etc. pergunte bloco e apartamento se necessário. e faça ligação para confirmar.",
            )
        ]
    )

    dify_api_key = os.environ.get("DIFY_API_KEY")
    dify_base_url = os.environ.get("DIFY_BASE_URL", "http://localhost/v1")

    if not dify_api_key:
        logger.error("DIFY_API_KEY is not set in the environment variables")
        return

    logger.debug(f"Using Dify API key: {dify_api_key[:5]}...{dify_api_key[-5:]}")
    logger.debug(f"Using Dify base URL: {dify_base_url}")

    try:
        dify_llm = DIFYLLM(
            api_key=dify_api_key,
            base_url=dify_base_url,
        )

        assistant = VoiceAssistant(
            vad=ctx.proc.userdata["vad"],
            stt=deepgram.STT(
                api_key=os.environ.get("DEEPGRAM_API_KEY"),
                model="nova-2-general",
                language="pt-BR",
            ),
            llm=dify_llm,
            tts=cartesia.TTS(
                api_key=os.environ.get("CARTESIA_API_KEY"),
                model="sonic-multilingual",
                language="pt",
                voice="700d1ee3-a641-4018-ba6e-899dcadc9e2b",
            ),
            chat_ctx=initial_ctx,
            transcription=AssistantTranscriptionOptions(
                sentence_tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=5),
                word_tokenizer=tokenize.basic.WordTokenizer(ignore_punctuation=False),
            ),
            preemptive_synthesis=True,
            interrupt_min_words=3,
            allow_interruptions=True,
            interrupt_speech_duration=0.5,
        )

        await ctx.connect()
        assistant.start(ctx.room)

        chat = rtc.ChatManager(ctx.room)

        async def answer_from_text(txt: str):
            logger.debug(f"Received text: {txt}")
            chat_ctx = assistant.chat_ctx.copy()
            chat_ctx.append(role="user", text=txt)
            try:
                async with dify_llm.chat(chat_ctx=chat_ctx) as stream:
                    full_response = ""
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            # Send each chunk to the assistant for immediate synthesis
                            if content.startswith("[Tool being called...]"):
                                await assistant.say(
                                    "Estou consultando uma ferramenta para obter mais informações. Um momento, por favor.",
                                    allow_interruptions=True,
                                )
                            elif content.startswith("[Tool output:"):
                                await assistant.say(
                                    "Recebi informações adicionais. Deixe-me processar isso para você.",
                                    allow_interruptions=True,
                                )
                            else:
                                await assistant.say(content, allow_interruptions=True)

                    logger.debug(f"Full response from Dify: {full_response}")
                    logger.debug(
                        f"Current conversation ID: {dify_llm._conversation_id}"
                    )

                    # Add the assistant's response to the chat context
                    chat_ctx.append(role="assistant", text=full_response)
                    assistant.chat_ctx = chat_ctx

            except Exception as e:
                logger.error(f"Error in answer_from_text: {e}")
                await assistant.say(
                    "Desculpe, ocorreu um erro ao processar sua solicitação. Posso tentar ajudar com outra coisa?",
                    allow_interruptions=True,
                )

        @chat.on("message_received")
        def on_chat_received(msg: rtc.ChatMessage):
            if msg.message:
                asyncio.create_task(answer_from_text(msg.message))

        try:
            await assistant.say("Inicio", allow_interruptions=True)
        except Exception as e:
            logger.error(f"Error in initial greeting: {e}")

        # Keep the agent running
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error initializing DIFYLLM: {e}")
        return


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
