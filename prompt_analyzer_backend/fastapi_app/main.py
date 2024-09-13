from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from datetime import datetime
from enum import Enum

DATABASE_URL = "postgresql+asyncpg://username:password@db:5432/mydatabase"

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    categorised_as = Column(String, nullable=False)
    generation = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_refused = Column(String, nullable=True)

class CategorisedAs(str, Enum):
    injection = "injection"
    unsafe = "unsafe"
    jailbreak = "jailbreak"
    toxic = "toxic"

class PromptInput(BaseModel):
    prompt: str
    categorised_as: CategorisedAs = Field(..., description="Возможные значения: injection, unsafe, jailbreak, toxic")
    generation: str = None
    score: float = None
    response_refused: str = None

app = FastAPI()

@app.on_event("startup")
async def create_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.post("/api/prompts/")
async def track_prompt(prompt_data: PromptInput):
    async with async_session() as session:
        new_prompt = Prompt(
            prompt=prompt_data.prompt,
            categorised_as=prompt_data.categorised_as,
            generation=prompt_data.generation,
            score=prompt_data.score,
            response_refused=prompt_data.response_refused
        )
        session.add(new_prompt)
        await session.commit()
    return {"message": "Prompt tracked successfully"}
