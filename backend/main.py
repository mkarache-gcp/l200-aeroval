"""
FastAPI application for AeroEval Test Engineering Assistant.

Starts the AeroEval Agent and FastAPI server with a single /chat endpoint
supporting Gemini 3.8 Flash & Anthropic Claude toggling.
"""

import os
import uvicorn
from typing import Optional
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agent import create_agent
from backend.memory import session_store

# Initialize the AeroEval Agent
agent = create_agent()

# Initialize FastAPI application
app = FastAPI(
    title="AeroEval",
    description="Hardware Test Engineering Telemetry Evaluation Agent",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default-session"
    provider: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    provider: str


@app.get("/healthz")
def healthz():
    """Health check probe for Cloud Run container orchestration."""
    return {"status": "healthy", "service": "aeroeval-backend"}


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """Single chat endpoint: routes to agent and persists memory asynchronously via BackgroundTasks.

    Prevents UI blocking by returning the response immediately while Firestore persistence executes
    out-of-band as a background task.
    """
    chosen_provider = req.provider or agent.default_provider
    reply = agent.chat(message=req.message, session_id=req.session_id, provider=chosen_provider)

    # Asynchronous memory persistence via FastAPI BackgroundTasks to eliminate UI blocking
    session = session_store.get_or_create(req.session_id)
    background_tasks.add_task(session.save)

    return ChatResponse(response=reply, session_id=req.session_id, provider=chosen_provider)


# Mount static frontend files for the chat window UI
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    print("🛰️  Starting AeroEval Agent & FastAPI Server...")
    print("🌐  Open your browser at: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
