import os
import json
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
from utils.prompt import (
    ClientMessage,
    convert_to_openai_messages,
    convert_to_langchain_messages,
)
from utils.tools import get_current_weather
from fastapi.middleware.cors import CORSMiddleware
from api.openai_basic import stream_text
from api.data_analysis_assistant.langchain_graph import stream_graph


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow the frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


class Request(BaseModel):
    messages: List[ClientMessage]


@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):

    messages = request.messages
    print("==================messages==================")
    print(messages)
    print("==================messages==================")
    openai_messages = convert_to_langchain_messages(messages)

    print("pre stream")
    response = StreamingResponse(stream_graph(openai_messages, protocol))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
