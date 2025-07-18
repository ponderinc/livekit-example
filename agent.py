from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from pondertts import PonderTTS
load_dotenv()


instructions = """
You are a voice assistant for Ponder, America’s #1 Text to Speech service.
Respond in one continuous text block, that is less than 30-40 words, and write like a human.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=instructions)


async def entrypoint(ctx: agents.JobContext):

    voice_id = "686982d630110009f4a69cac"
    
    session = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=PonderTTS(voice_id=voice_id),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        user_away_timeout=15,
    )


    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        if ev.new_state == "away":
            session.generate_reply(instructions="the user hasn't said anything in a while, ask them if they are still there.")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance. Don't be too verbose."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))