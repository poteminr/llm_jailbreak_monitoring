from pydantic import BaseModel, Field
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean
from datetime import datetime

DATABASE_URL = "postgresql+asyncpg://username:password@db:5432/mydatabase"

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class Prompt(Base):
    __tablename__ = "prompts_"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False) 
    categorised_as = Column(String, nullable=False)
    generation = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    is_input_jailbreak = Column(Boolean, nullable=False, default=False)
    output_label = Column(String, nullable=True)
    is_unsafe = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_refused = Column(String, nullable=True)


class CategorisedAs(str, Enum):
    injection = "injection"
    unsafe = "unsafe"
    jailbreak = "jailbreak"
    toxic = "toxic"

class PromptInputModel(Base):
    __tablename__ = 'prompts'

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    categorised_as = Column(String, nullable=False)  # Adjust type and constraints as needed
    generation = Column(String)
    score = Column(Float)
    response_refused = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PromptInput(BaseModel):
    prompt: str
    categorised_as: str = Field(..., description="Possible values: injection, unsafe, jailbreak, toxic")
    generation: str = None
    score: float = None
    response_refused: bool = None
