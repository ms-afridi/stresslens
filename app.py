"""StressLens: AI-image detection, robustness, explainability, and reporting."""

from __future__ import annotations

import inspect
import streamlit as st
from PIL import Image, ImageOps

from stresslens.detector import PrimaryDetector
from stresslens.secondary_detector import SecondaryDetector
from stresslens.robustness import run_robustness_suite
from stresslens.reliability import summarize_robustness
from stresslens.explain import (
    run_occlusion_experiment,
    create_occlusion_heatmap,
)
from stresslens.llm_note import generate_research_commentary
from stresslens.pdf_report import generate_pdf_report


st.set_page_config(
    page_title="StressLens",
    page_icon="🔬",
    layout="wide",
)

EXPERIMENTS = ("crop", "blur", "jpeg_compression", "resize")


@st.cache_resource(show_spinner=False)
def get_primary_detector():
    return PrimaryDetector()


@st.cache_resource(show_spinner=False)
def get_secondary_detector():
    return SecondaryDetector()


def call_compatible(func, **kwargs):
    """Pass only arguments accepted by the current function signature."""
    signature = inspect.signature(func)

    accepts_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_extra_kwargs:
        return func(**kwargs)

    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
        and signature.parameters[key].kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    return func(**supported)


def format_score(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "Unavailable"


def show_detection(title, result):
    st.subheader(title)

    if result is None:
        st.warning("This detector did not produce a result.")
        return

    left, right = st.columns(2)
    left.metric("Predicted label", str(result.predicted_label))
    right.metric("AI-class score", format_score(result.ai_score))

    st.caption(f"Model: {result.model_id}")
    st.caption(result.score_semantics)

    if result.scores:
        st.dataframe(
            [
                {"Class": label, "Model score": score}
                for label, score in result.scores.items()
            ],
            hide_index=True,
            use_container_width=True,
        )


def show_robustness(title, raw_results, summary):
    st.subheader(title)

    experiments = (summary or {}).get("experiments", {})

    for name in EXPERIMENTS:
        stage = (raw_results or {}).get(name) or {}
        item = experiments.get(name) or {}

        with st.expander(name.replace("_", " ").title()):
            if not stage and not item:
                st.warning("No evidence was recorded for this experiment.")
                continue

            st.write(
                item.get(
                    "interpretation",
                    "No interpretation is available.",
                )
            )

            a, b, c = st.columns(3)
            a.metric("Successful runs", item.get("successful_runs", 0))
            b.metric(
                "Mean absolute score change",
                format_score(item.get("mean_absolute_score_change")),
            )
            c.metric(
                "Label-change rate",
                format_score(item.get("label_change_rate")),
            )

            if stage.get("error"):
                st.warning(str(stage["error"]))

            rows = stage.get("results", [])

            if rows:
                st.dataframe(
                    rows,
                    hide_index=True,
                    use_container_width=True,
                )
            elif not stage.get("error"):
                st.info("No successful runs were recorded.")


def run_analysis(image):
    """Run each analysis stage independently so one failure doesn't stop all."""
    errors = []

    result = {
        "image": image.copy(),
        "primary": None,
        "secondary": None,
        "primary_raw": None,
        "secondary_raw": None,
        "primary_summary": None,
        "secondary_summary": None,
        "primary_occlusion": None,
        "secondary_occlusion": None,
        "commentary": None,
        "pdf": None,
        "errors": errors,
    }

    primary = get_primary_detector()
    secondary = get_secondary_detector()

    try:
        result["primary"] = primary.predict(image)
    except Exception as exc:
        errors.append(
            f"Primary detector: {type(exc).__name__}: {exc}"
        )

    try:
        result["secondary"] = secondary.predict(image)
    except Exception as exc:
        errors.append(
            f"Secondary detector: {type(exc).__name__}: {exc}"
        )

    if result["primary"] is not None:
        try:
            result["primary_raw"] = run_robustness_suite(
                primary,
                image,
                baseline=result["primary"],
            )
            result["primary_summary"] = summarize_robustness(
                result["primary_raw"]
            )
        except Exception as exc:
            errors.append(
                f"Primary robustness: {type(exc).__name__}: {exc}"
            )

        try:
            result["primary_occlusion"] = run_occlusion_experiment(
                primary,
                image,
                baseline=result["primary"],
            )
        except Exception as exc:
            errors.append(
                f"Primary occlusion: {type(exc).__name__}: {exc}"
            )

    if result["secondary"] is not None:
        try:
            result["secondary_raw"] = run_robustness_suite(
                secondary,
                image,
                baseline=result["secondary"],
            )
            result["secondary_summary"] = summarize_robustness(
                result["secondary_raw"]
            )
        except Exception as exc:
            errors.append(
                f"Secondary robustness: {type(exc).__name__}: {exc}"
            )

        try:
            result["secondary_occlusion"] = run_occlusion_experiment(
                secondary,
                image,
                baseline=result["secondary"],
            )
        except Exception as exc:
            errors.append(
                f"Secondary occlusion: {type(exc).__name__}: {exc}"
            )

    try:
        result["commentary"] = call_compatible(
            generate_research_commentary,
            detection_result=result["primary"],
            robustness_summary=result["primary_summary"],
            occlusion_results=result["primary_occlusion"],
            secondary_detection_result=result["secondary"],
            secondary_robustness_summary=result["secondary_summary"],
            secondary_occlusion_results=result["secondary_occlusion"],
            extra_context={
                "image_dimensions": {
                    "width": image.width,
                    "height": image.height,
                },
                "analysis_scope": (
                    "One uploaded image and the transformations "
                    "successfully completed."
                ),
            },
        )
    except Exception as exc:
        result["commentary"] = {
            "status": "failed",
            "commentary": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        errors.append(f"Research commentary: {type(exc).__name__}: {exc}")

    try:
        result["pdf"] = call_compatible(
            generate_pdf_report,
            detection_result=result["primary"],
            robustness_summary=result["primary_summary"],
            occlusion_results=result["primary_occlusion"],
            llm_commentary=result["commentary"],
            image=image.copy(),
            project_title="StressLens Image Reliability Report",
            secondary_detection_result=result["secondary"],
            secondary_robustness_summary=result["secondary_summary"],
            secondary_occlusion_results=result["secondary_occlusion"],
        )
    except Exception as exc:
        errors.append(f"PDF generation: {type(exc).__name__}: {exc}")

    return result


st.title("🔬 StressLens")
st.markdown(
    "**AI-generated image detection · Robustness testing · Explainability**"
)
st.write(
    "Compare two image detectors, test their sensitivity to image "
    "transformations, inspect occlusion-based influence, and create a PDF "
    "research report."
)

st.info(
    "Research prototype. Model scores are not calibrated probabilities "
    "and do not prove image authenticity."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
)

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {type(exc).__name__}: {exc}")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with right:
        st.write(f"**Dimensions:** {image.width} × {image.height}")
        st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
        st.write(
            "**Experiments:** crop, blur, JPEG compression, resizing, "
            "and occlusion"
        )

    if st.button(
        "Run complete analysis",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Running detectors, robustness tests, explainability, "
            "commentary, and PDF generation..."
        ):
            st.session_state["stresslens_result"] = run_analysis(image)

if "stresslens_result" in st.session_state:
    result = st.session_state["stresslens_result"]

    st.divider()
    st.header("1. Detector results")

    col1, col2 = st.columns(2)

    with col1:
        show_detection("Primary detector", result["primary"])

    with col2:
        show_detection("Secondary detector", result["secondary"])

    primary = result["primary"]
    secondary = result["secondary"]

    if primary is not None and secondary is not None:
        if primary.predicted_label.lower() == secondary.predicted_label.lower():
            st.success("The detector labels agree for this image.")
        else:
            st.warning(
                "The detector labels disagree. Neither output should "
                "be treated as ground truth."
            )

    st.header("2. Robustness experiments")

    col1, col2 = st.columns(2)

    with col1:
        show_robustness(
            "Primary detector",
            result["primary_raw"],
            result["primary_summary"],
        )

    with col2:
        show_robustness(
            "Secondary detector",
            result["secondary_raw"],
            result["secondary_summary"],
        )

    st.header("3. Occlusion explainability")

    col1, col2 = st.columns(2)

    for column, title, key in (
        (col1, "Primary detector", "primary_occlusion"),
        (col2, "Secondary detector", "secondary_occlusion"),
    ):
        with column:
            st.subheader(title)
            occlusion = result[key]

            if occlusion and occlusion.get("status") in ("success", "partial"):
                try:
                    heatmap = create_occlusion_heatmap(
                        result["image"],
                        occlusion,
                    )
                    st.image(
                        heatmap,
                        caption="Occlusion influence map",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(f"Heatmap unavailable: {exc}")

                region = occlusion.get("most_influential_region")

                if region:
                    st.write(
                        f"Most influential tested region: "
                        f"**{region.get('region', 'unknown')}**"
                    )
                    st.write(
                        "Observed score change: "
                        f"**{format_score(region.get('score_change'))}**"
                    )

                st.caption(
                    occlusion.get(
                        "interpretation_note",
                        "Perturbation-based explanation; not causal.",
                    )
                )
            else:
                st.info("Occlusion results are unavailable.")

    st.header("4. Research interpretation")

    commentary = result.get("commentary") or {}

    if commentary.get("commentary"):
        st.markdown(commentary["commentary"])
    else:
        st.warning(
            "LLM commentary unavailable: "
            + str(commentary.get("error", "No commentary was returned."))
        )

    st.header("5. Download report")

    if result.get("pdf"):
        st.download_button(
            "Download StressLens PDF report",
            data=result["pdf"],
            file_name="stresslens_research_report.pdf",
            mime="application/pdf",
            type="primary",
        )
    else:
        st.warning("PDF generation did not return a report.")

    if result["errors"]:
        with st.expander("Technical warnings"):
            for error in result["errors"]:
                st.warning(error)

    st.caption(
        "These tests characterize model behaviour on this image under "
        "selected transformations. They do not establish general "
        "real-world accuracy."
    )