import asyncio
from .kql_expert_bot import KqlExpertBotTool
from pbyc.types import Action, Response
from fastapi import FastAPI
from typing import Dict
import os
from dotenv import load_dotenv
import aiohttp
load_dotenv()

app = FastAPI()


@app.post("/kql-expert-bot", response_model=Response)
async def root(action: Action):

    response_url = action.response_url
    correlation_id = action.correlation_id
    session_id = action.session_id

    credentials = {
        "AZURE_OPENAI_API_DEPLOYMENT_NAME": os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"],
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "AZURE_OPENAI_API_BASE": os.environ["AZURE_OPENAI_API_BASE"],
    }

    async def progress(data: Response):
        data.correlation_id = correlation_id
        data.session_id = session_id

        print("Sending response to: " + response_url)
        data = data.dict()
        print(data)
        async with aiohttp.ClientSession() as session:
            async with session.post(response_url, json=data) as resp:
                print(resp.status)
                print(await resp.text())

    engine = KqlExpertBotTool(
        action.project, progress, credentials=credentials)

    if action.action == "utterance":
        asyncio.create_task(engine.take_utterance(
            action.utterance, chat_history=action.chat_history, files=action.files))
    elif action.action == "representation_edit":
        asyncio.create_task(engine.take_representation_edit(
            action.changed_representation))
    elif action.action == "output":
        asyncio.create_task(engine.get_output(
            action.utterance, chat_history=action.chat_history, files=action.files))

    return Response(
        type="thought",
        message="",
        correlation_id=correlation_id,
        session_id=session_id
    )
