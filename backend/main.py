"""FastAPI backend for LLM Council."""

import logging
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from . import storage
from . import groq
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage2_5_debate,
    stage3_stream,
    calculate_aggregate_rankings,
    _build_conversation_history,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await groq.close_client()
    logger.info("HTTP client closed")


app = FastAPI(title="LLM Council API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    pass


class SendMessageRequest(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > 5000:
            raise ValueError("Message too long (max 5000 characters)")
        return v


class ConversationMetadata(BaseModel):
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    conversation_id = str(uuid.uuid4())
    return storage.create_conversation(conversation_id)


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """Send a message and run the full council process (non-streaming)."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0
    history = _build_conversation_history(conversation["messages"])

    storage.add_user_message(conversation_id, request.content)

    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    stage1_results, stage2_results, stage2_5_result, stage3_result, metadata = (
        await run_full_council(request.content, history)
    )

    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        metadata,
        stage2_5_result,
    )

    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage2_5": stage2_5_result,
        "stage3": stage3_result,
        "metadata": metadata,
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """Send a message and stream each stage via SSE."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0
    # Capture history before adding the new user message
    history = _build_conversation_history(conversation["messages"])

    async def event_generator():
        try:
            storage.add_user_message(conversation_id, request.content)

            # Kick off title generation in parallel with Stage 1 (first message only)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(
                    generate_conversation_title(request.content)
                )

            # --- Stage 1 ---
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(request.content, history)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # --- Stage 2 ---
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(
                request.content, stage1_results
            )
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # --- Stage 2.5: Debate ---
            yield f"data: {json.dumps({'type': 'stage2_5_start'})}\n\n"
            stage2_5_result = await stage2_5_debate(
                request.content, stage1_results, aggregate_rankings
            )
            yield f"data: {json.dumps({'type': 'stage2_5_complete', 'data': stage2_5_result})}\n\n"

            # --- Stage 3: stream word by word ---
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_content = ""
            async for chunk in stage3_stream(
                request.content, stage1_results, stage2_results, stage2_5_result
            ):
                stage3_content += chunk
                yield f"data: {json.dumps({'type': 'stage3_chunk', 'content': chunk})}\n\n"

            stage3_result = {"model": "chairman", "response": stage3_content}
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # --- Persist ---
            metadata = {
                "label_to_model": label_to_model,
                "aggregate_rankings": aggregate_rankings,
            }

            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                metadata,
                stage2_5_result,
            )

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming endpoint: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
