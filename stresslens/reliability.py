"""Descriptive summaries of image transformation sensitivity."""

import numpy as np


EXPERIMENTS = (
    "crop",
    "blur",
    "jpeg_compression",
    "resize",
)


def summarize_robustness(results):
    if not isinstance(results, dict):
        results = {}

    baseline = results.get("baseline") or {}

    try:
        baseline_score = float(baseline["ai_score"])
    except (KeyError, TypeError, ValueError):
        baseline_score = None

    summaries = {}
    all_changes = []
    all_label_changes = []

    for name in EXPERIMENTS:
        stage = results.get(name) or {}

        rows = (
            stage.get("results", [])
            if isinstance(stage, dict)
            else []
        )

        valid = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "success":
                continue

            score = row.get("ai_score")

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not np.isfinite(score)
            ):
                continue

            valid.append(row)

        changes = []

        for row in valid:
            change = row.get("score_change")

            if (
                isinstance(change, (int, float))
                and not isinstance(change, bool)
                and np.isfinite(change)
            ):
                changes.append(abs(float(change)))

            elif baseline_score is not None:
                changes.append(
                    abs(float(row["ai_score"]) - baseline_score)
                )

        label_flags = [
            row["label_changed"]
            for row in valid
            if isinstance(row.get("label_changed"), bool)
        ]

        mean_change = (
            float(np.mean(changes))
            if changes
            else None
        )

        if not valid:
            interpretation = "No successful scores were available."
        elif mean_change is None:
            interpretation = (
                "Score variation could not be calculated."
            )
        elif mean_change < 0.05:
            interpretation = (
                "Small score variation under these tested settings. "
                "This does not establish general reliability."
            )
        elif mean_change < 0.15:
            interpretation = (
                "Noticeable sensitivity under these tested settings. "
                "Interpret the detector output cautiously."
            )
        else:
            interpretation = (
                "Substantial sensitivity under some tested settings. "
                "Further validation is needed."
            )

        summaries[name] = {
            "status": "success" if valid else "unavailable",
            "successful_runs": len(valid),
            "total_runs": len(rows),
            "mean_absolute_score_change": mean_change,
            "max_absolute_score_change": (
                float(max(changes)) if changes else None
            ),
            "label_changes": (
                sum(label_flags) if label_flags else None
            ),
            "label_change_rate": (
                float(np.mean(label_flags))
                if label_flags
                else None
            ),
            "interpretation": interpretation,
        }

        all_changes.extend(changes)
        all_label_changes.extend(label_flags)

    overall_mean = (
        float(np.mean(all_changes))
        if all_changes
        else None
    )

    return {
        "baseline": baseline,
        "experiments": summaries,
        "overall": {
            "status": (
                "success" if all_changes else "unavailable"
            ),
            "mean_absolute_score_change": overall_mean,
            "label_change_count": (
                sum(all_label_changes)
                if all_label_changes
                else None
            ),
            "label_change_rate": (
                float(np.mean(all_label_changes))
                if all_label_changes
                else None
            ),
            "assessment": (
                "Descriptive measurements on this image only; "
                "not a calibrated reliability metric or a measure "
                "of general real-world accuracy."
            ),
        },
    }