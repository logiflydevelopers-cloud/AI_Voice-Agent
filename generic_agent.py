from livekit.agents import Agent
from livekit.agents.llm import ChatContext


class GenericAgent(Agent):
    def __init__(
        self,
        instructions: str,
        llm,
        chat_ctx: ChatContext | None = None,
    ):
        super().__init__(
            instructions=instructions,
            llm=llm,
            chat_ctx=chat_ctx,
        )
