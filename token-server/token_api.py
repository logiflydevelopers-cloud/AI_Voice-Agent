import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from livekit.api import (
    AccessToken,
    VideoGrants,
    RoomConfiguration,
    RoomAgentDispatch,
)

load_dotenv()

app = FastAPI()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# ----------------------------
# Validate Environment
# ----------------------------

if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise RuntimeError("LiveKit environment variables are missing")


# ----------------------------
# Request Schema
# ----------------------------

class TokenRequest(BaseModel):
    userId: str


# ----------------------------
# Token Endpoint
# ----------------------------

@app.post("/getToken", status_code=201)
async def get_token(body: TokenRequest):
    try:
        user_id = body.userId.strip()

        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid userId")

        # 🔥 One room per user
        room_name = f"user-{user_id}"

        token = (
            AccessToken(
                LIVEKIT_API_KEY,
                LIVEKIT_API_SECRET,
            )
            .with_identity(user_id)
            .with_name(user_id)
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
            # 🔥 AUTO DISPATCH YOUR AGENT
            .with_room_config(
                RoomConfiguration(
                    agents=[
                        RoomAgentDispatch(
                            agent_name="voice-agent",  # MUST match livekit.toml
                            metadata=f'{{"user_id": "{user_id}"}}',
                        )
                    ]
                )
            )
        )

        return {
            "server_url": LIVEKIT_URL,
            "room_name": room_name,
            "participant_token": token.to_jwt(),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
