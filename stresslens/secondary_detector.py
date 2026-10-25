
"""Secondary AI-generated-image detector for StressLens.

Uses a separate Hugging Face model repository. Scores are model outputs,
not calibrated probabilities.
"""

from typing import Dict

from .detector import (
    DetectionResult,
    PrimaryDetector,
)


SECONDARY_MODEL_ID = "capcheck/ai-human-generated-image-detection"


class SecondaryDetector(PrimaryDetector):
    """Run the secondary model using StressLens's shared result format."""

    def __init__(self, model_id: str = SECONDARY_MODEL_ID):
        super().__init__(model_id=model_id)

    def _find_ai_score(self, scores: Dict[str, float]) -> float:
        """Identify the AI-class score only when label semantics are clear."""
        normalized_scores = {
            self._normalise_label(label): float(score)
            for label, score in scores.items()
        }

        ai_terms = ("ai", "fake", "artificial", "generated", "synthetic")
        real_terms = ("real", "human", "authentic", "natural")

        ai_candidates = []
        real_candidates = []

        for label, score in normalized_scores.items():
            if any(term in label for term in ai_terms):
                ai_candidates.append(score)
            if any(term in label for term in real_terms):
                real_candidates.append(score)

        if len(ai_candidates) == 1:
            return ai_candidates[0]

        if not ai_candidates and len(normalized_scores) == 2:
            if len(real_candidates) == 1:
                return 1.0 - real_candidates[0]

        raise ValueError(
            "The secondary model's class labels do not unambiguously "
            "identify the AI-generated class. Inspect the model card "
            "and model configuration before interpreting its scores. "
            f"Available labels: {list(scores.keys())}"
        )

    def predict(self, image) -> DetectionResult:
        """Return the secondary model prediction in the shared result format."""
        result = super().predict(image)

        return DetectionResult(
            predicted_label=result.predicted_label,
            ai_score=result.ai_score,
            scores=result.scores,
            model_id=result.model_id,
            score_semantics=(
                "Secondary model softmax output for the AI-generated class; "
                "not a calibrated probability or proof of authenticity. "
                "Interpretation depends on the correctness of the model's "
                "class-label mapping."
            ),
        )
