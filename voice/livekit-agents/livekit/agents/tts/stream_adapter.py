from __future__ import annotations

import asyncio
import time

from .. import tokenize, utils
from ..log import logger
from .tts import (
    TTS,
    ChunkedStream,
    SynthesizeStream,
    TTSCapabilities,
)


class StreamAdapter(TTS):
    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(
                streaming=True,
            ),
            sample_rate=tts.sample_rate,
            num_channels=tts.num_channels,
        )
        self._tts = tts
        self._sentence_tokenizer = sentence_tokenizer

    def synthesize(self, text: str) -> ChunkedStream:
        return self._tts.synthesize(text=text)

    def stream(self) -> SynthesizeStream:
        return StreamAdapterWrapper(
            tts=self._tts,
            sentence_tokenizer=self._sentence_tokenizer,
        )


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__()
        self._tts = tts
        self._sent_stream = sentence_tokenizer.stream()
        self._text_buffer = ""
        self._synthesize_lock = asyncio.Lock()
        self._last_input_time = 0
        self._synthesis_timer = None

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async def _forward_input():
            async for input in self._input_ch:
                if isinstance(input, self._FlushSentinel):
                    self._sent_stream.flush()
                    await self._synthesize_buffer()
                    continue
                self._text_buffer += input
                self._last_input_time = time.time()
                if len(self._text_buffer) >= 20:
                    await self._synthesize_buffer()
                else:
                    self._schedule_synthesis()
            
            self._sent_stream.end_input()
            await self._synthesize_buffer()

        async def _synthesize_buffer():
            async with self._synthesize_lock:
                if self._text_buffer:
                    self._sent_stream.push_text(self._text_buffer)
                    async for audio in self._tts.synthesize(self._text_buffer):
                        self._event_ch.send_nowait(audio)
                    self._text_buffer = ""
                if self._synthesis_timer:
                    self._synthesis_timer.cancel()
                    self._synthesis_timer = None

        def _schedule_synthesis():
            if self._synthesis_timer:
                self._synthesis_timer.cancel()
            self._synthesis_timer = asyncio.create_task(self._delayed_synthesis())

        async def _delayed_synthesis():
            await asyncio.sleep(0.5)  # Espera 500ms antes de sintetizar
            if time.time() - self._last_input_time >= 0.5:
                await self._synthesize_buffer()

        await _forward_input()
