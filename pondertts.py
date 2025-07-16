from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass, replace
import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("livekit.plugins.pondertts")

BASE_URL = "inf.useponder.ai" # <- or your self-hosted Ponder instance
NUM_CHANNELS = 1


def _to_ponder_url(api_key: str, voice_id: str, websocket: bool = True) -> str:
    if websocket:
        return f"wss://{BASE_URL}/v1/ws/tts?api_key={api_key}&voice_id={voice_id}"
    else:
        return f"https://{BASE_URL}/v1/tts"



@dataclass
class _TTSOptions:
    model: str
    encoding: str
    sample_rate: int
    voice_id: str
    use_websocket: bool
    word_tokenizer: tokenize.WordTokenizer
    api_key: str
    mip_opt_out: bool = False


class PonderTTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str | None = None,
        use_websocket: bool = False,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        mip_opt_out: bool = False,
    ) -> None:
        """
        Create a new instance of Ponder TTS.

        Args:
            api_key (str): Ponder API key.
            voice_id (str): Voice ID to use.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text.
            http_session (aiohttp.ClientSession): Optional aiohttp session to use for requests.
            mip_opt_out (bool): Whether to opt out of MIP.
        """
        
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_websocket),
            sample_rate=24000,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("PONDER_API_KEY")
        if not api_key:
            raise ValueError("Ponder API key required. Set PONDER_API_KEY or provide api_key.")

        voice_id = voice_id or os.environ.get("PONDER_VOICE_ID")
        if not voice_id:
            raise ValueError("Ponder voice ID required. Set PONDER_VOICE_ID or provide voice_id.")

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            model="pondertts",
            encoding="linear16",
            sample_rate=24000,
            voice_id=voice_id,
            use_websocket=use_websocket,
            word_tokenizer=word_tokenizer,
            api_key=api_key,
            mip_opt_out=mip_opt_out,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        if use_websocket:
            self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
                connect_cb=self._connect_ws,
                close_cb=self._close_ws,
                max_session_duration=3600,  # 1 hour
                mark_refreshed_on_get=False,
            )

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()

        return await asyncio.wait_for(
            session.ws_connect(
                _to_ponder_url(self._opts.api_key, self._opts.voice_id, websocket=True),

            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice_id (str): Voice ID to use.
        """
        if is_given(voice_id):
            self._opts.voice_id = voice_id
    

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)



    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        if self._opts.use_websocket:
            self._pool.prewarm()
        else:
            pass

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

        await self._pool.aclose()



class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                _to_ponder_url(
                    self._opts.api_key,
                    self._opts.voice_id,
                    websocket=False,
                ),
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                },
                json={"text": self._input_text, "voice_id": self._opts.voice_id},
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into WordStreams and sends them into _segments_ch
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            self._segments_ch.close()

        async def _run_segments() -> None:
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, word_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for word in word_stream:
                speak_msg = {"type": "text", "text": f"{word.token} "}
                self._mark_started()
                await ws.send_str(json.dumps(speak_msg))

            flush_msg = {"type": "flush"}
            await ws.send_str(json.dumps(flush_msg))

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError("Ponder websocket connection closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    resp = json.loads(msg.data)
                    if resp.get("type") == "flushed":
                        output_emitter.end_segment()
                        break
                    logger.info(f"Ponder websocket text response: {resp}")

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)