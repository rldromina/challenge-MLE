import fastapi
from pydantic import BaseModel, Field
from typing import List, Literal

from challenge.model import DelayModel
import pandas as pd

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

model = DelayModel()
app = fastapi.FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: Literal["N", "I"]
    MES: int = Field(ge=1, le=12)


class PredictRequest(BaseModel):
    flights: List[Flight]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(body: PredictRequest) -> dict:
    data = pd.DataFrame([flight.dict() for flight in body.flights])
    features = model.preprocess(data=data)
    predictions = model.predict(features=features)
    return {"predict": predictions}




