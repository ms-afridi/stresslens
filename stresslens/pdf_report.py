
"""PDF report generation for StressLens."""

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get(obj, key, default=None):
    """Read a value from either a dictionary or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _safe_text(value, default="Unavailable"):
    """Convert values into safe, readable PDF text."""
    if value is None:
        return default

    if isinstance(value, (dict, list, tuple)):
        value = str(value)

    value = str(value).strip()

    return value if value else default


def _paragraph(value, style):
    """Create a ReportLab Paragraph with safely escaped text."""
    text = escape(_safe_text(value))
    text = text.replace("\n", "<br/>")
    return Paragraph(text, style)


def _number(value):
    """Format numeric values without crashing on missing data."""
    if value is None:
        return "Unavailable"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _percentage(value):
    """Format a rate as a percentage when it is a valid number."""
    if value is None:
        return "Unavailable"

    try:
        number = float(value)

        # The app stores rates between 0 and 1.
        if 0 <= number <= 1:
            return f"{number * 100:.2f}%"

        return f"{number:.2f}%"

    except (TypeError, ValueError, OverflowError):
        return str(value)


def _table_style():
    """Consistent styling for all report tables."""
    return TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#263B53"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white,
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.4,
            colors.HexColor("#C5CDD6"),
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP",
        ),
        (
            "LEFTPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "RIGHTPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            6,
        ),
        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            6,
        ),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [
                colors.white,
                colors.HexColor("#F5F7FA"),
            ],
        ),
    ])


def _make_table_cell(value, styles):
    """Make a wrapping table cell."""
    return _paragraph(value, styles["table_cell"])


def _make_table_header(value, styles):
    """Make a wrapping header cell."""
    return _paragraph(value, styles["table_header"])


# ---------------------------------------------------------------------
# Detector comparison table
# ---------------------------------------------------------------------

def _add_detector_table(story, primary, secondary, styles):
    """Add detector comparison table with wrapped cell content."""

    headers = [
        _make_table_header("Field", styles),
        _make_table_header("Primary detector", styles),
        _make_table_header("Secondary detector", styles),
    ]

    data = [headers]

    fields = (
        ("Model", "model_id"),
        ("Predicted label", "predicted_label"),
        ("AI-class score", "ai_score"),
        ("Score semantics", "score_semantics"),
    )

    for label, key in fields:
        first = _get(primary, key)
        second = _get(secondary, key)

        if key == "ai_score":
            first = _number(first)
            second = _number(second)

        data.append([
            _make_table_cell(label, styles),
            _make_table_cell(first, styles),
            _make_table_cell(second, styles),
        ])

    # A4 width is 21 cm. With 1.7 cm margins on both sides,
    # the available content width is 17.6 cm.
    table = Table(
        data,
        colWidths=[
            3.0 * cm,
            7.3 * cm,
            7.3 * cm,
        ],
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(_table_style())

    story.extend([
        table,
        Spacer(1, 12),
    ])


# ---------------------------------------------------------------------
# Robustness sections
# ---------------------------------------------------------------------

def _add_robustness_section(story, title, summary, styles):
    story.append(
        Paragraph(
            escape(title),
            styles["Heading1"],
        )
    )

    if not summary:
        story.append(
            _paragraph(
                "No robustness summary was available.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))
        return

    experiments = summary.get("experiments", {})

    if not experiments:
        story.append(
            _paragraph(
                "No experiment results were recorded.",
                styles["BodyText"],
            )
        )
    else:
        data = [[
            _make_table_header("Experiment", styles),
            _make_table_header("Successful runs", styles),
            _make_table_header(
                "Mean absolute score change",
                styles,
            ),
            _make_table_header("Label-change rate", styles),
        ]]

        for name, item in experiments.items():
            item = item or {}

            data.append([
                _make_table_cell(name, styles),
                _make_table_cell(
                    item.get("successful_runs", "Unavailable"),
                    styles,
                ),
                _make_table_cell(
                    _number(
                        item.get("mean_absolute_score_change")
                    ),
                    styles,
                ),
                _make_table_cell(
                    _percentage(item.get("label_change_rate")),
                    styles,
                ),
            ])

        table = Table(
            data,
            colWidths=[
                4.0 * cm,
                2.8 * cm,
                6.0 * cm,
                4.8 * cm,
            ],
            repeatRows=1,
            hAlign="LEFT",
        )

        table.setStyle(_table_style())

        story.extend([
            table,
            Spacer(1, 10),
        ])

    overall = summary.get("overall", {})

    if isinstance(overall, dict):
        assessment = overall.get("assessment")

        if assessment:
            story.append(
                _paragraph(
                    assessment,
                    styles["BodyText"],
                )
            )

        scope_note = overall.get("scope_note")

        if scope_note:
            story.append(
                _paragraph(
                    scope_note,
                    styles["SmallNote"],
                )
            )

    story.append(Spacer(1, 8))


# ---------------------------------------------------------------------
# Occlusion / explainability sections
# ---------------------------------------------------------------------

def _add_occlusion_section(story, title, results, styles):
    story.append(
        Paragraph(
            escape(title),
            styles["Heading1"],
        )
    )

    if not results:
        story.append(
            _paragraph(
                "No occlusion results were available.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))
        return

    region = results.get("most_influential_region") or {}

    story.append(
        _paragraph(
            f"Status: {results.get('status', 'Unavailable')}",
            styles["BodyText"],
        )
    )

    story.append(
        _paragraph(
            "Successful regions: "
            f"{results.get('successful_regions', 0)} / "
            f"{results.get('total_regions', 0)}",
            styles["BodyText"],
        )
    )

    story.append(
        _paragraph(
            "Most influential tested region: "
            f"{region.get('region', 'Unavailable')}",
            styles["BodyText"],
        )
    )

    story.append(
        _paragraph(
            "Observed score change: "
            f"{_number(region.get('score_change'))}",
            styles["BodyText"],
        )
    )

    story.append(
        _paragraph(
            results.get(
                "interpretation_note",
                (
                    "Occlusion is a perturbation-based explanation. "
                    "It does not establish causal reasoning."
                ),
            ),
            styles["SmallNote"],
        )
    )

    story.append(Spacer(1, 10))


# ---------------------------------------------------------------------
# Main PDF generator
# ---------------------------------------------------------------------

def generate_pdf_report(
    detection_result=None,
    robustness_summary=None,
    occlusion_results=None,
    llm_commentary=None,
    evaluation_results=None,
    image=None,
    project_title="StressLens Research Report",
    researcher="Not specified",
    secondary_detection_result=None,
    secondary_robustness_summary=None,
    secondary_occlusion_results=None,
):
    """
    Build and return the complete StressLens PDF report as bytes.

    Supports both primary and secondary detectors while preserving
    the existing app.py function interface.
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=project_title,
        author=researcher,
    )

    styles = getSampleStyleSheet()

    # Wrapping styles for table content.
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#444444"),
            alignment=TA_LEFT,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="DetectorTableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            alignment=TA_LEFT,
            wordWrap="CJK",
            splitLongWords=True,
        )
    )

    styles.add(
        ParagraphStyle(
            name="DetectorTableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=colors.white,
            alignment=TA_LEFT,
            wordWrap="CJK",
            splitLongWords=True,
        )
    )

    # These aliases let all table functions share the same styles.
    styles.add(
        ParagraphStyle(
            name="table_cell",
            parent=styles["DetectorTableCell"],
        )
    )

    styles.add(
        ParagraphStyle(
            name="table_header",
            parent=styles["DetectorTableHeader"],
        )
    )

    story = [
        Paragraph(
            escape(_safe_text(project_title)),
            styles["Title"],
        ),
        Paragraph(
            "AI image detector behaviour and robustness report",
            styles["Heading2"],
        ),
        _paragraph(
            f"Researcher: {researcher}",
            styles["BodyText"],
        ),
        Spacer(1, 10),
    ]

    # -------------------------------------------------------------
    # Uploaded image
    # -------------------------------------------------------------

    if image is not None:
        try:
            # Copy the image so the source object is not modified.
            report_image = image.copy().convert("RGB")
            report_image.thumbnail((900, 900))

            image_buffer = BytesIO()

            report_image.save(
                image_buffer,
                format="JPEG",
                quality=85,
            )

            image_buffer.seek(0)

            display_width = 8 * cm
            display_height = (
                display_width
                * report_image.height
                / report_image.width
            )

            story.extend([
                ReportLabImage(
                    image_buffer,
                    width=display_width,
                    height=display_height,
                ),
                Spacer(1, 10),
            ])

        except Exception:
            story.append(
                _paragraph(
                    "The uploaded image could not be embedded.",
                    styles["SmallNote"],
                )
            )

    # -------------------------------------------------------------
    # Detector results
    # -------------------------------------------------------------

    story.append(
        Paragraph(
            "Detector results",
            styles["Heading1"],
        )
    )

    _add_detector_table(
        story,
        detection_result,
        secondary_detection_result,
        styles,
    )

    # -------------------------------------------------------------
    # Robustness experiments
    # -------------------------------------------------------------

    _add_robustness_section(
        story,
        "Primary detector robustness",
        robustness_summary,
        styles,
    )

    _add_robustness_section(
        story,
        "Secondary detector robustness",
        secondary_robustness_summary,
        styles,
    )

    # -------------------------------------------------------------
    # Explainability
    # -------------------------------------------------------------

    _add_occlusion_section(
        story,
        "Primary occlusion explanation",
        occlusion_results,
        styles,
    )

    _add_occlusion_section(
        story,
        "Secondary occlusion explanation",
        secondary_occlusion_results,
        styles,
    )

    # -------------------------------------------------------------
    # Research interpretation / LLM commentary
    # -------------------------------------------------------------

    story.append(
        Paragraph(
            "Research interpretation",
            styles["Heading1"],
        )
    )

    if isinstance(llm_commentary, dict):
        commentary = (
            llm_commentary.get("commentary")
            or (
                "Commentary unavailable: "
                + str(
                    llm_commentary.get(
                        "error",
                        "No details supplied.",
                    )
                )
            )
        )
    else:
        commentary = str(
            llm_commentary
            or "No LLM commentary was supplied."
        )

    for paragraph in commentary.split("\n"):
        if paragraph.strip():
            story.append(
                _paragraph(
                    paragraph,
                    styles["BodyText"],
                )
            )

    # -------------------------------------------------------------
    # Optional evaluation results
    # -------------------------------------------------------------

    if evaluation_results:
        story.append(
            Paragraph(
                "Dataset evaluation",
                styles["Heading1"],
            )
        )

        for key, value in evaluation_results.items():
            # Avoid dumping a potentially large confusion matrix as
            # an unreadable line of text.
            if key in ("confusion_matrix",):
                continue

            story.append(
                _paragraph(
                    f"{key.replace('_', ' ').title()}: "
                    f"{_safe_text(value)}",
                    styles["BodyText"],
                )
            )

    # -------------------------------------------------------------
    # Limitations
    # -------------------------------------------------------------

    story.extend([
        Spacer(1, 12),
        Paragraph(
            "Research limitations",
            styles["Heading1"],
        ),
        _paragraph(
            (
                "Detector scores are model outputs, not necessarily "
                "calibrated probabilities. These experiments describe "
                "behaviour on one image under selected transformations "
                "and do not establish ground truth or general "
                "real-world accuracy. Occlusion results are "
                "perturbation-based and do not prove causal reasoning."
            ),
            styles["SmallNote"],
        ),
    ])

    document.build(story)

    buffer.seek(0)
    return buffer.getvalue()
