"""
FastAPI application for AeroEval Test Engineering Assistant.

Starts the AeroEval Agent and FastAPI server with a single /chat endpoint
supporting Gemini 3.8 Flash & Anthropic Claude toggling.
"""

import os
import uvicorn
from typing import Any, Dict, Optional
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agent import create_agent
from backend.memory import session_store

# Initialize the AeroEval Multi-Agent Orchestrator
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
    agent: Optional[str] = None
    model: Optional[str] = None
    routing: Optional[Dict[str, Any]] = None
    requires_approval: bool = False
    pending_draft: Optional[Dict[str, Any]] = None


class ApprovalRequest(BaseModel):
    session_id: str
    draft_id: str
    approved: bool = True


class ApprovalResponse(BaseModel):
    status: str
    message: str
    ticket_id: Optional[str] = None
    draft_id: Optional[str] = None


@app.get("/healthz")
def healthz():
    """Health check probe for Cloud Run container orchestration."""
    return {"status": "healthy", "service": "aeroeval-backend"}


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """Chat endpoint with multi-agent strategic routing and async Firestore persistence.

    Routes between AeroEvalAgent (Telemetry) and DocGenAgent (Reports/Tickets) autonomously,
    and returns pending Human-In-The-Loop approval state if action requires signoff.
    """
    chosen_provider = req.provider or agent.default_provider
    orchestrated = agent.chat_orchestrated(message=req.message, session_id=req.session_id, provider=chosen_provider)

    # Asynchronous memory persistence via FastAPI BackgroundTasks to eliminate UI blocking
    session = session_store.get_or_create(req.session_id)
    background_tasks.add_task(session.save)

    return ChatResponse(
        response=orchestrated["response"],
        session_id=req.session_id,
        provider=orchestrated.get("provider", chosen_provider),
        agent=orchestrated.get("agent"),
        model=orchestrated.get("model"),
        routing=orchestrated.get("routing"),
        requires_approval=orchestrated.get("requires_approval", False),
        pending_draft=orchestrated.get("pending_draft"),
    )


@app.post("/approve-action", response_model=ApprovalResponse)
async def handle_approval(req: ApprovalRequest, background_tasks: BackgroundTasks) -> ApprovalResponse:
    """Human-In-The-Loop (HITL) approval endpoint for one-click action signoff."""
    res = agent.handle_approval(session_id=req.session_id, draft_id=req.draft_id, approved=req.approved)

    # Persist updated session state in background
    session = session_store.get_or_create(req.session_id)
    background_tasks.add_task(session.save)

    return ApprovalResponse(
        status=res.get("status", "ERROR"),
        message=res.get("message", ""),
        ticket_id=res.get("ticket_id"),
        draft_id=res.get("draft_id", req.draft_id),
    )


# Mount static frontend files for the chat window UI
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    print("🛰️  Starting AeroEval Agent & FastAPI Server...")
    print("🌐  Open your browser at: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
