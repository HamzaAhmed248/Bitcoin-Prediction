from __future__ import annotations

from typing import Any

import joblib

try:
    from api.config import MODEL_PATH
except ImportError:  # pragma: no cover - supports direct script execution
    from config import MODEL_PATH


class ModelLoader:
    """Lazy model loader for inference."""

    def __init__(self, model_path: str | Any = MODEL_PATH) -> None:
        self.model_path = model_path
        self._model: Any | None = None

    def load(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model


model_loader = ModelLoader(MODEL_PATH)


def load_model() -> Any:
    return model_loader.load()
