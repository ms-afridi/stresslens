"""Occlusion-based perturbation explanations."""

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def run_occlusion_experiment(
    detector,
    image,
    baseline=None,
    grid_size=4,
    fill_value=127,
):
    """Mask image regions and measure changes in the AI-class score."""
    if not 2 <= grid_size <= 8:
        raise ValueError("grid_size must be between 2 and 8.")

    if not 0 <= fill_value <= 255:
        raise ValueError("fill_value must be between 0 and 255.")

    image = ImageOps.exif_transpose(image).convert("RGB")

    if baseline is None:
        baseline = detector.predict(image)

    width, height = image.size
    records = []

    for row in range(grid_size):
        for column in range(grid_size):
            left = round(column * width / grid_size)
            right = round((column + 1) * width / grid_size)
            top = round(row * height / grid_size)
            bottom = round((row + 1) * height / grid_size)

            region_name = f"row_{row + 1}_column_{column + 1}"

            masked = image.copy()
            draw = ImageDraw.Draw(masked)

            draw.rectangle(
                (
                    left,
                    top,
                    max(left, right - 1),
                    max(top, bottom - 1),
                ),
                fill=(fill_value, fill_value, fill_value),
            )

            try:
                result = detector.predict(masked)
                change = float(
                    result.ai_score - baseline.ai_score
                )

                records.append({
                    "row": row,
                    "column": column,
                    "region": region_name,
                    "bounds": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    },
                    "predicted_label": result.predicted_label,
                    "ai_score": float(result.ai_score),
                    "score_change": change,
                    "absolute_score_change": abs(change),
                    "label_changed": (
                        result.predicted_label
                        != baseline.predicted_label
                    ),
                    "status": "success",
                })

            except Exception as exc:
                records.append({
                    "row": row,
                    "column": column,
                    "region": region_name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    successful = [
        record
        for record in records
        if record["status"] == "success"
    ]

    most_influential = (
        max(
            successful,
            key=lambda record: record["absolute_score_change"],
        )
        if successful
        else None
    )

    return {
        "status": (
            "success"
            if len(successful) == len(records)
            else "partial"
            if successful
            else "failed"
        ),
        "baseline_ai_score": float(baseline.ai_score),
        "baseline_label": baseline.predicted_label,
        "grid_size": grid_size,
        "fill_value": fill_value,
        "results": records,
        "successful_regions": len(successful),
        "total_regions": len(records),
        "most_influential_region": most_influential,
        "interpretation_note": (
            "Regions with larger absolute score changes had greater "
            "measured influence under this masking procedure. This is "
            "a perturbation-based explanation, not a causal explanation. "
            "Results depend on the image and masking settings."
        ),
    }


def create_occlusion_heatmap(image, occlusion_results):
    """Create a heatmap from successful occlusion results."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    grid_size = int(occlusion_results.get("grid_size", 4))

    if not 2 <= grid_size <= 8:
        raise ValueError("Invalid grid_size.")

    values = np.zeros((grid_size, grid_size), dtype=float)
    valid = np.zeros((grid_size, grid_size), dtype=bool)

    for record in occlusion_results.get("results", []):
        if record.get("status") != "success":
            continue

        try:
            row = int(record["row"])
            column = int(record["column"])
            value = float(record["absolute_score_change"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (
            0 <= row < grid_size
            and 0 <= column < grid_size
            and np.isfinite(value)
        ):
            continue

        values[row, column] = value
        valid[row, column] = True

    if not valid.any():
        raise ValueError("No successful occlusion regions.")

    maximum = float(values[valid].max())

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size

    for row in range(grid_size):
        for column in range(grid_size):
            box = (
                round(column * width / grid_size),
                round(row * height / grid_size),
                round((column + 1) * width / grid_size),
                round((row + 1) * height / grid_size),
            )

            if valid[row, column]:
                intensity = (
                    values[row, column] / maximum
                    if maximum > 0
                    else 0
                )
                alpha = int(25 + 150 * intensity)

                draw.rectangle(
                    box,
                    fill=(255, 0, 0, alpha),
                    outline=(255, 255, 255, 180),
                    width=2,
                )
            else:
                draw.rectangle(
                    box,
                    outline=(255, 255, 255, 180),
                    width=2,
                )

    return Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    ).convert("RGB")