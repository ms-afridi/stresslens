"""Optional evidence-grounded research commentary using Groq."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = """
You are a cautious research assistant interpreting an image-detector
experiment. Use only the evidence supplied by StressLens.

Do not invent results, infer ground truth, or change detector predictions.
Do not describe model scores as calibrated probabilities. If two detectors
disagree, state that neither is established as correct. Discuss only the
transformations and explainability results actually provided. State that
findings are limited to the tested image and settings.
""".strip()


def _result_dict(result):
    if result is None:
        return None

    if isinstance(result, dict):
        return result

    return {
        key: getattr(result, key, None)
        for key in (
            "predicted_label",
            "ai_score",
            "scores",
            "model_id",
            "score_semantics",
        )
    }


def generate_research_commentary(
    detection_result=None,
    robustness_summary=None,
    occlusion_results=None,
    extra_context=None,
    max_tokens=1800,
    temperature=0.2,
    secondary_detection_result=None,
    secondary_robustness_summary=None,
    secondary_occlusion_results=None,
):
    api_key = (
        os.getenv("GROQ_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    if not api_key:
        return {
            "status": "unavailable",
            "commentary": None,
            "model": model,
            "error": (
                "GROQ_API_KEY is missing from the project-root .env file. "
                "The rest of StressLens can run without LLM commentary."
            ),
        }

    evidence = {
        "primary_detector": _result_dict(detection_result),
        "primary_robustness": robustness_summary,
        "primary_occlusion": occlusion_results,
        "secondary_detector": _result_dict(
            secondary_detection_result
        ),
        "secondary_robustness": secondary_robustness_summary,
        "secondary_occlusion": secondary_occlusion_results,
        "additional_context": extra_context or {},
    }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Write a concise research interpretation from "
                        "the following JSON evidence. Discuss detector "
                        "agreement or disagreement, robustness results, "
                        "occlusion findings, missing evidence, and "
                        "limitations. Do not invent missing values.\n\n"
                        + json.dumps(
                            evidence,
                            ensure_ascii=False,
                            default=str,
                            allow_nan=False,
                        )
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        commentary = response.choices[0].message.content

        if not commentary or not commentary.strip():
            return {
                "status": "failed",
                "commentary": None,
                "model": model,
                "error": "The LLM returned empty commentary.",
            }

        return {
            "status": "success",
            "commentary": commentary.strip(),
            "model": model,
            "error": None,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "commentary": None,
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
        }