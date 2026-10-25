"""Image transformation experiments for StressLens."""

import io

from PIL import Image, ImageFilter, ImageOps


EXPERIMENTS = (
    "crop",
    "blur",
    "jpeg_compression",
    "resize",
)


def _record(detector, transformed, baseline, setting):
    try:
        result = detector.predict(transformed)

        return {
            "status": "success",
            "setting": setting,
            "predicted_label": result.predicted_label,
            "ai_score": float(result.ai_score),
            "score_change": float(
                result.ai_score - baseline.ai_score
            ),
            "label_changed": bool(
                result.predicted_label != baseline.predicted_label
            ),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "setting": setting,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _crop(image, fraction):
    width, height = image.size

    dx = int(width * fraction)
    dy = int(height * fraction)

    box = (
        dx,
        dy,
        max(dx + 1, width - dx),
        max(dy + 1, height - dy),
    )

    return image.crop(box).resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )


def _jpeg(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def _resize(image, scale):
    width, height = image.size

    smaller_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )

    smaller = image.resize(
        smaller_size,
        Image.Resampling.LANCZOS,
    )

    return smaller.resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )


def run_robustness_suite(detector, image, baseline=None):
    """Run all transformation experiments and return their raw results."""
    image = ImageOps.exif_transpose(image).convert("RGB")

    if baseline is None:
        baseline = detector.predict(image)

    specifications = {
        "crop": [
            ("crop_5_percent", lambda: _crop(image, 0.05)),
            ("crop_10_percent", lambda: _crop(image, 0.10)),
            ("crop_15_percent", lambda: _crop(image, 0.15)),
        ],
        "blur": [
            (
                "radius_0.5",
                lambda: image.filter(ImageFilter.GaussianBlur(0.5)),
            ),
            (
                "radius_1.0",
                lambda: image.filter(ImageFilter.GaussianBlur(1.0)),
            ),
            (
                "radius_2.0",
                lambda: image.filter(ImageFilter.GaussianBlur(2.0)),
            ),
        ],
        "jpeg_compression": [
            (
                f"quality_{quality}",
                lambda quality=quality: _jpeg(image, quality),
            )
            for quality in (90, 70, 50)
        ],
        "resize": [
            (
                f"scale_{scale}",
                lambda scale=scale: _resize(image, scale),
            )
            for scale in (0.75, 0.50, 0.25)
        ],
    }

    output = {
        "baseline": {
            "predicted_label": baseline.predicted_label,
            "ai_score": float(baseline.ai_score),
            "model_id": baseline.model_id,
        }
    }

    for experiment_name, runs in specifications.items():
        rows = []

        for setting, transform in runs:
            try:
                transformed = transform()
                rows.append(
                    _record(
                        detector,
                        transformed,
                        baseline,
                        setting,
                    )
                )
            except Exception as exc:
                rows.append({
                    "status": "failed",
                    "setting": setting,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        succeeded = any(
            row.get("status") == "success"
            for row in rows
        )

        output[experiment_name] = {
            "status": "success" if succeeded else "failed",
            "results": rows,
        }

        if not succeeded:
            output[experiment_name]["error"] = (
                "No transformation run succeeded."
            )

    return output