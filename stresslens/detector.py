"""Shared image detector implementation."""

from dataclasses import dataclass
from typing import Dict

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


DEFAULT_MODEL_ID = "dima806/ai_vs_real_image_detection"


@dataclass
class DetectionResult:
    predicted_label: str
    ai_score: float
    scores: Dict[str, float]
    model_id: str
    score_semantics: str = (
        "Model softmax output score; not a calibrated probability "
        "or proof of image authenticity."
    )


class PrimaryDetector:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageClassification.from_pretrained(
                model_id
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{model_id}'. Check model access, "
                f"network connectivity, and installed dependencies. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        self.id2label = {
            int(index): str(label)
            for index, label in self.model.config.id2label.items()
        }

    @staticmethod
    def _is_ai_label(label: str) -> bool:
        normalized = (
            label.lower().replace("-", "_").replace(" ", "_")
        )
        ai_terms = (
            "fake",
            "ai",
            "generated",
            "synthetic",
            "artificial",
        )
        real_terms = (
            "human",
            "real",
            "authentic",
            "natural",
        )
        return (
            any(term in normalized for term in ai_terms)
            and not any(term in normalized for term in real_terms)
        )

    @staticmethod
    def _is_real_label(label: str) -> bool:
        normalized = (
            label.lower().replace("-", "_").replace(" ", "_")
        )
        ai_terms = (
            "fake",
            "ai",
            "generated",
            "synthetic",
            "artificial",
        )
        real_terms = (
            "human",
            "real",
            "authentic",
            "natural",
        )
        return (
            any(term in normalized for term in real_terms)
            and not any(term in normalized for term in ai_terms)
        )

    def predict(self, image: Image.Image) -> DetectionResult:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image.")

        image = ImageOps.exif_transpose(image).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        scores = {
            self.id2label.get(index, f"class_{index}"): float(
                probabilities[index].item()
            )
            for index in range(len(probabilities))
        }

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_label = self.id2label.get(
            predicted_index,
            f"class_{predicted_index}",
        )

        ai_scores = [
            score
            for label, score in scores.items()
            if self._is_ai_label(label)
        ]
        real_scores = [
            score
            for label, score in scores.items()
            if self._is_real_label(label)
        ]

        if ai_scores:
            ai_score = max(ai_scores)
        elif len(scores) == 2 and len(real_scores) == 1:
            ai_score = 1.0 - real_scores[0]
        else:
            raise ValueError(
                "Could not reliably identify the AI class from model "
                f"labels: {list(scores.keys())}"
            )

        return DetectionResult(
            predicted_label=predicted_label,
            ai_score=float(ai_score),
            scores=scores,
            model_id=self.model_id,
        )