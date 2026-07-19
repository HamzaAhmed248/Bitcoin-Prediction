from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

try:
    from api.config import BASE_DIR
    from api.model_loader import load_model
    from api.preprocessing import FEATURE_COLUMNS, load_history, prepare_features
except ImportError:  # pragma: no cover - supports direct script execution
    from config import BASE_DIR
    from model_loader import load_model
    from preprocessing import FEATURE_COLUMNS, load_history, prepare_features

app = FastAPI(title="Bitcoin Prediction API")


class UserInput(BaseModel):
    Closing_P: float
    Highest_P: float
    Lowest_P: float
    Volume: float


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon() -> Response:
    return Response(content=b"", media_type="image/x-icon")


@app.get("/")
def serve_about_root() -> FileResponse:
    return FileResponse(str(BASE_DIR / "about.html"))


@app.get("/about")
def serve_about() -> FileResponse:
    return FileResponse(str(BASE_DIR / "about.html"))


@app.get("/prediction")
def serve_ui() -> FileResponse:
    return FileResponse(str(BASE_DIR / "ui.html"))


@app.get("/style.css")
def serve_css() -> FileResponse:
    return FileResponse(str(BASE_DIR / "style.css"))


@app.get("/nav.html")
def serve_nav() -> FileResponse:
    return FileResponse(str(BASE_DIR / "nav.html"))


@app.post("/predict")
def predict(data: UserInput) -> dict[str, Any]:
    try:
        history = load_history()
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        features_df = prepare_features(history, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Inference history is unavailable.") from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=400, detail="Unable to build prediction features.") from exc

    try:
        model = load_model()
        prediction = model.predict(features_df)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Model artifact could not be loaded.") from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return {"prediction": round(float(prediction), 2)}
