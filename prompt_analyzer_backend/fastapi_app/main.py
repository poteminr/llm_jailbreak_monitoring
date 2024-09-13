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
from typing import Optional

app = FastAPI()
detector = Detector()


@app.on_event("startup")
async def create_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def mocked_generate_answer(value: str) -> str:
    return value

@app.post("/api/prompts/")
async def track_prompt(prompt_data: str, output_data: Optional[str] = None):
    if output_data is None:
        output_data = mocked_generate_answer(prompt_data)

    result = detector.check_model_artefacts(prompt_data, output_data)

    '''
            results = {
            "input_text" : input_text,
            "input_score": input_check_results[0],
            "is_input_jailbreak": input_check_results[1],
            "generated_text" : generated_text,
            "output_label": output_check_results[0],
            "is_unsafe": output_check_results[1]
        }
    '''

    async with async_session() as session:
        new_prompt = PromptInputModel(
            prompt=prompt_data,
            categorised_as='jailbreak' if result['is_input_jailbreak'] else 'regular_text',
            generation=output_data,
            score=result['input_score'],
            response_refused=result['is_input_jailbreak'],
            output_label=result['output_label'],
            is_unsafe=result['is_unsafe'],
        )
        session.add(new_prompt)
        await session.commit()
    return {"message": f"Prompt tracked successfully\n{result}"}
