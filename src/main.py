# src/main.py
from contextlib import asynccontextmanager
from typing import Annotated
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from src.model import load_model, load_encoder  # adjust if your path differs

ml_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup
    ml_models["ohe"] = load_encoder()
    ml_models["model"] = load_model()
    try:
        yield
    finally:
        # Free memory at shutdown
        ml_models.clear()

app = FastAPI(lifespan=lifespan)
bearer = HTTPBearer()

def get_username_for_token(token: str):
    return "teteca" if token == os.getenv("API_TOKEN", "abc123") else None

async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials
    username = get_username_for_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"username": username}

class Person(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    balance: int
    housing: str
    duration: int
    campaign: int

@app.get("/")
async def root():
    return "Model API is alive!"

@app.post("/predict")
async def predict(
    person: Annotated[Person, Body(examples=[{
        "age": 42, "job": "entrepreneur", "marital": "married",
        "education": "primary", "balance": 558, "housing": "yes",
        "duration": 186, "campaign": 2
    }])],
    user = Depends(validate_token),
):
    ohe = ml_models["ohe"]
    model = ml_models["model"]

    df = pd.DataFrame([person.model_dump() if hasattr(person, "model_dump") else person.dict()])
    X = ohe.transform(df)
    pred = model.predict(X)[0]
    return {"prediction": str(pred), "username": user["username"]}
