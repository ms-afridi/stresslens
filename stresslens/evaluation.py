"""Dataset evaluation; classification metrics require ground-truth labels."""

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


AI_LABEL = "AI"
REAL_LABEL = "REAL"


def normalize_ground_truth(value):
    if value is None or pd.isna(value):
        return None

    label = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if label in {
        "ai",
        "fake",
        "generated",
        "ai_generated",
        "synthetic",
        "artificial",
        "1",
    }:
        return AI_LABEL

    if label in {
        "real",
        "human",
        "authentic",
        "natural",
        "0",
    }:
        return REAL_LABEL

    return None


def normalize_prediction(label):
    value = str(label).strip().lower()

    ai_match = any(
        term in value
        for term in (
            "fake",
            "ai",
            "generated",
            "synthetic",
            "artificial",
        )
    )

    real_match = any(
        term in value
        for term in (
            "real",
            "human",
            "authentic",
            "natural",
        )
    )

    if ai_match and not real_match:
        return AI_LABEL

    if real_match and not ai_match:
        return REAL_LABEL

    return None


def evaluate_dataset(
    detector,
    dataset,
    image_column="image_path",
    label_column="label",
    base_dir=None,
):
    required = {image_column, label_column}
    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    base_path = (
        Path(base_dir).expanduser()
        if base_dir
        else None
    )

    records = []

    for index, row in dataset.iterrows():
        raw_path = row.get(image_column)
        ground_truth = normalize_ground_truth(
            row.get(label_column)
        )

        record = {
            "row_index": index,
            "image_path": str(raw_path or ""),
            "ground_truth": ground_truth,
            "predicted_label": None,
            "predicted_label_raw": None,
            "ai_score": None,
            "status": "failed",
            "error": None,
        }

        if ground_truth is None:
            record["error"] = (
                "Missing or unsupported ground-truth label."
            )
            records.append(record)
            continue

        if raw_path is None or pd.isna(raw_path):
            record["error"] = "Missing image path."
            records.append(record)
            continue

        image_path = Path(str(raw_path)).expanduser()

        if not image_path.is_absolute() and base_path is not None:
            image_path = base_path / image_path

        if not image_path.is_file():
            record["error"] = (
                f"Image file not found: {image_path}"
            )
            records.append(record)
            continue

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")

            result = detector.predict(image)
            prediction = normalize_prediction(
                result.predicted_label
            )

            record.update({
                "predicted_label_raw": result.predicted_label,
                "predicted_label": prediction,
                "ai_score": float(result.ai_score),
                "status": (
                    "success" if prediction is not None else "failed"
                ),
                "error": (
                    None
                    if prediction is not None
                    else "Could not map the model label to AI or REAL."
                ),
            })

        except Exception as exc:
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        records.append(record)

    results_df = pd.DataFrame(records)

    valid = results_df[
        results_df["status"].eq("success")
        & results_df["ground_truth"].isin([AI_LABEL, REAL_LABEL])
        & results_df["predicted_label"].isin([AI_LABEL, REAL_LABEL])
    ].copy()

    if valid.empty:
        return {
            "status": "unavailable",
            "message": (
                "No samples had both valid ground-truth and "
                "interpretable prediction labels."
            ),
            "total_samples": len(dataset),
            "successful_samples": 0,
            "failed_samples": len(dataset),
            "metrics": {},
            "confusion_matrix": None,
            "confusion_matrix_labels": [REAL_LABEL, AI_LABEL],
            "classification_report": {},
            "results": results_df,
        }

    y_true = valid["ground_truth"].tolist()
    y_pred = valid["predicted_label"].tolist()

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_ai": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=AI_LABEL,
                zero_division=0,
            )
        ),
        "recall_ai": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=AI_LABEL,
                zero_division=0,
            )
        ),
        "f1_ai": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=AI_LABEL,
                zero_division=0,
            )
        ),
    }

    labels = [REAL_LABEL, AI_LABEL]

    return {
        "status": "success",
        "message": (
            "Metrics were calculated only from samples with "
            "valid ground-truth and predicted labels."
        ),
        "total_samples": len(dataset),
        "successful_samples": len(valid),
        "failed_samples": len(dataset) - len(valid),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "results": results_df,
        "limitations": [
            "Metrics depend on dataset composition and label quality.",
            "Small or unrepresentative datasets may not generalize.",
            "The AI score is not assumed to be calibrated.",
            "Failed or unmappable samples are excluded from metrics.",
        ],
    }