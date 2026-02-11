from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentServer,
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    room_io,
)
from livekit.plugins import openai, silero

from rag import query_pinecone

load_dotenv()  # works locally, ignored in cloud


# --------------------------------------------------
# RAG VOICE AGENT
# --------------------------------------------------

class RAGVoiceAgent(Agent):

    def __init__(self, chat_ctx: ChatContext, user_id: str):
        self.user_id = user_id

        super().__init__(
            chat_ctx=chat_ctx,
            instructions=(
                "You are a voice AI assistant. "
                "Always speak only in English. "
                "Answer strictly from the provided knowledge base context. "
                "If the answer is not found in the context, say exactly: "
                "'I do not know.' "
                "Use short spoken sentences."
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:

        query = new_message.text_content

        print(f"\n🔎 Searching Pinecone namespace: {self.user_id}")
        print("User query:", query)

        docs = query_pinecone(query, user_id=self.user_id)

        if not docs:
            print("❌ No results from Pinecone")

            turn_ctx.add_message(
                role="system",
                content="No relevant knowledge base information was found.",
            )
            return

        rag_content = "\n\n".join(d.page_content for d in docs)

        print("✅ Injecting RAG context")

        turn_ctx.add_message(
            role="assistant",
            content=f"Knowledge base information:\n\n{rag_content}",
        )


# --------------------------------------------------
# SERVER
# --------------------------------------------------

server = AgentServer()


@server.rtc_session(agent_name="voice-agent")
async def my_agent(ctx: agents.JobContext):

    print("✅ Agent job received")

    # 🔥 STEP 1: CONNECT TO ROOM
    await ctx.connect()

    print("🔗 Room connected")

    # 🔥 STEP 2: Wait for participant
    participant = await ctx.wait_for_participant()

    if not participant:
        print("❌ No participant joined")
        return

    user_id = participant.identity
    print("🔥 Using namespace:", user_id)

    initial_ctx = ChatContext()

    # STT
    base_stt = openai.STT(model="gpt-4o-transcribe")

    # VAD
    vad_model = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.5,
    )

    streaming_stt = agents.stt.StreamAdapter(
        stt=base_stt,
        vad=vad_model,
    )

    # Agent session
    session = AgentSession(
        stt=streaming_stt,
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
    )

    await session.start(
        room=ctx.room,
        agent=RAGVoiceAgent(
            chat_ctx=initial_ctx,
            user_id=user_id,
        ),
        room_options=room_io.RoomOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user and offer assistance."
    )

# --------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------

if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(server)
