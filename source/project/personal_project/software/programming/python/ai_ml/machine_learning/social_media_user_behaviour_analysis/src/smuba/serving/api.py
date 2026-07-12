"""Optional FastAPI layer, for when a Gradio demo isn't enough (e.g. you
want to show a proper REST API in your portfolio, or call it from another
service). Loads whatever is currently in the MLflow 'Production' stage.
"""
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from smuba.config import load_config

_state = {"model": None, "data_cfg": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state["data_cfg"] = cfg["data"]
    mlflow.set_tracking_uri(cfg["pipeline"]["mlflow_tracking_uri"])
    model_name = cfg["pipeline"]["registered_model_name"]
    _state["model"] = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    yield


app = FastAPI(title="Social Media User Behaviour Predictor", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


@app.post("/predict")
def predict(payload: PredictionRequest):
    row = pd.DataFrame([payload.model_dump()])
    row = row[_state["data_cfg"]["selected_features"]]
    prediction = _state["model"].predict(row)
    return {"prediction": prediction.tolist()[0]}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}
