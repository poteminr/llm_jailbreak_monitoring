from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from datetime import datetime
from enum import Enum
from models import PromptInput, Prompt, PromptInputModel
from models import engine, Base, async_session
from detector import Detector

app = FastAPI()
detector = Detector()


@app.on_event("startup")
async def create_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/api/prompts/")
async def track_prompt(prompt_data: str):
    print("Main part")

    result = detector.check_input(prompt_data)

    print(f"{result = }")

    '''
        results = {
            "input_text": input_text,
            "input_score": input_check_results[0],
            "is_input_jailbreak": input_check_results[1]
        }
        
    prompt: str
    categorised_as: CategorisedAs = Field(..., description="Возможные значения: injection, unsafe, jailbreak, toxic")
    generation: str = None
    score: float = None
    response_refused: str = None

    '''

    async with async_session() as session:
        new_prompt = PromptInputModel(
            prompt=prompt_data,
            categorised_as='jailbreak' if result['is_input_jailbreak'] else 'regular_text',
            generation='', # TODO
            score=result['input_score'],
            response_refused=result['is_input_jailbreak']
        )
        session.add(new_prompt)
        await session.commit()
    return {"message": "Prompt tracked successfully"}
