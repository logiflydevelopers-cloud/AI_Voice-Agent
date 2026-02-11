import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from livekit import api
from dotenv import load_dotenv

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
            api.AccessToken(
                LIVEKIT_API_KEY,
                LIVEKIT_API_SECRET,
            )
            .with_identity(user_id)
            .with_name(user_id)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            # 🔥 AUTO DISPATCH YOUR AGENT
            .with_room_config(
                api.RoomConfiguration(
                    agents=[
                        api.RoomAgentDispatch(
                            agent_name="voice-agent"  # must match rtc_session name
                        )
                    ]
                )
            )
        )

        jwt_token = token.to_jwt()

        return {
            "server_url": LIVEKIT_URL,
            "room_name": room_name,
            "participant_token": jwt_token,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
