# StressLens
### Exploring Synthetic-Image Detection, Model Reliability, and Explainable AI

**StressLens** is a research-oriented prototype for exploring the reliability of AI-generated image detection systems. It combines two pretrained image classifiers with image-transformation tests, model-comparison analysis, and human-readable reporting to investigate an important question:

> **Can we trust an AI-generated image detector when different models disagree or their predictions change under different conditions?**

Rather than treating a detector's prediction as definitive proof, StressLens explores how model outputs, prediction scores, robustness tests, and explanatory reports can help users understand the limitations of synthetic-media detection.

**Project status:** Early-stage research prototype. Detection accuracy, calibration, and generalizability require systematic evaluation.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Objectives](#key-objectives)
- [How StressLens Works](#how-stresslens-works)
- [Detection Models](#detection-models)
- [Robustness and Reliability Analysis](#robustness-and-reliability-analysis)
- [Observed Research Problem](#observed-research-problem)
- [Research Gaps and Limitations](#research-gaps-and-limitations)
- [Future Research Roadmap](#future-research-roadmap)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running StressLens](#running-stresslens)
- [Responsible Use](#responsible-use)
- [Research Direction](#research-direction)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Motivation

AI-generated images are becoming increasingly realistic, making it more difficult to distinguish synthetic content from authentic photographs. Such content can be misused in misinformation, impersonation, fraud, and other forms of social engineering.

Although machine-learning models can help identify synthetic media, their outputs are not infallible. Performance may vary across image generators, datasets, compression levels, image transformations, and real-world deployment conditions.

Three challenges motivate StressLens:

1. **Model disagreement:** Different detectors can produce conflicting predictions for the same image.
2. **Robustness:** Image transformations can affect a detector's prediction scores or classification.
3. **Explainability and uncertainty:** A prediction score alone does not necessarily tell a user whether a result is reliable or why a model reached its decision.

StressLens provides a practical starting point for examining these challenges through a reproducible analysis workflow.

## Key Objectives

- Compare predictions from two pretrained synthetic-image detection models.
- Examine differences between their predicted labels and reported scores.
- Test prediction stability under image transformations.
- Investigate localized sensitivity using an occlusion-based analysis.
- Generate structured reliability reports that make results easier to inspect.
- Identify limitations that motivate future work in AI Security, robustness evaluation, and Explainable AI (XAI).

## How StressLens Works

The prototype follows a pipeline from image input to comparative analysis and reporting.

1. **Image input:** The user supplies an image for analysis.
2. **Primary detection:** The image is passed to the primary pretrained classifier.
3. **Secondary detection:** The same image is evaluated using an independent pretrained classifier.
4. **Comparative analysis:** The system examines predicted labels and model-reported scores.
5. **Robustness testing:** The image is transformed through operations such as cropping, blurring, JPEG compression, resizing, and occlusion.
6. **Reliability analysis:** Changes in predictions and scores are summarized to help assess model stability.
7. **Reporting:** The system produces a structured report containing model outputs, test observations, and limitations.

The workflow is designed to encourage critical interpretation of detector outputs rather than blind reliance on a single classification.

## Detection Models

StressLens currently uses two pretrained Hugging Face image-classification models.

### 1. Primary detector

**Model:** [`dima806/ai_vs_real_image_detection`](https://huggingface.co/dima806/ai_vs_real_image_detection)

This model classifies images into real and fake categories. StressLens uses its output as one source of evidence when assessing whether an image may be AI-generated.

The model's prediction should not be interpreted as ground truth. Its performance depends on the images and generation methods encountered during training and evaluation, and may not generalize to newer image-generation systems.

### 2. Secondary detector

**Model:** [`capcheck/ai-human-generated-image-detection`](https://huggingface.co/capcheck/ai-human-generated-image-detection)

This model provides a second classification perspective, distinguishing human-generated images from AI-generated images.

Its output is compared with the primary detector to identify agreement or disagreement. However, two model predictions are not equivalent to two independent confirmations of the image's authenticity.

### Why use two models?

Using two detectors makes disagreement visible. For example, one model may classify an image as fake while the other classifies it as human-generated.

This is an important diagnostic signal, but disagreement alone does not establish which model is correct. Both models may be wrong, and their errors may be correlated because of overlapping training data, architectures, or learned visual features.

A future version should evaluate both models on a common, independently labelled benchmark and explicitly validate their class-label mappings and score semantics.

## Robustness and Reliability Analysis

StressLens includes image-transformation tests designed to examine how stable detector outputs remain when an input image is modified.

| Test | Purpose |
|---|---|
| Cropping | Examines sensitivity to changes in visible image regions and framing. |
| Blurring | Tests sensitivity to reduced image detail. |
| JPEG compression | Examines the effect of lossy compression artefacts. |
| Resizing | Tests the effect of changes in image dimensions and resampling. |
| Occlusion | Investigates whether predictions change when selected image regions are hidden. |

### What is measured?

The analysis can compare:

- Predicted labels before and after transformation.
- Changes in model-reported scores.
- Label changes, where observed.
- Differences between the primary and secondary models.
- Regional score changes during occlusion analysis.

For a score \(s\), the absolute score change can be expressed as:

\[
\Delta s = |s_{\text{transformed}} - s_{\text{original}}|
\]

A small score change indicates stability for that particular transformation and input. It does **not** prove that the original prediction is correct.

Likewise, a large score change may indicate sensitivity to the transformation, but further experiments are required to determine its significance.

### An important distinction

**Robustness is not the same as accuracy.** A model can consistently produce the wrong answer. A model can also produce a correct answer on one image but fail on another image generated by an unfamiliar system.

StressLens therefore treats transformation testing as one component of reliability analysis, not as a substitute for ground-truth evaluation.

## Observed Research Problem

During an exploratory test of an image identified by the project owner as AI-generated, the two detectors produced conflicting outputs:

- The primary detector predicted **FAKE**, with a reported AI-class score of approximately **0.7784**.
- The secondary detector predicted **human**, with a reported AI-class score of approximately **0.0105**.

These values are observations from an individual test, not estimates of either model's overall accuracy.

The disagreement raises several research questions:

1. Are the models interpreting their output labels and class scores correctly?
2. Do the models generalize to the image generator used to create the test image?
3. Are their score values comparable, and are those scores calibrated probabilities?
4. Do similar disagreements occur across a larger and more diverse test dataset?
5. Can uncertainty estimates help distinguish reliable predictions from ambiguous cases?

The correct response to disagreement is not to arbitrarily select one model as correct. Instead, the system should expose the disagreement, validate the underlying model outputs, and communicate uncertainty.

## Research Gaps and Limitations

### 1. Limited evaluation data

A small collection of example images cannot establish general detection performance.

**Research opportunity:** Evaluate on a larger, balanced dataset containing authentic photographs and AI-generated images from multiple sources, with clear provenance and independently verified labels.

### 2. Generalization to new generators

Detectors may perform differently on image generators or generation methods not represented in their training data.

**Research opportunity:** Use cross-generator evaluation, where models are tested on generators excluded from training or development datasets.

### 3. Model disagreement and score semantics

Conflicting predictions may arise from genuine model differences, domain shift, preprocessing, or incorrect interpretation of model outputs.

**Research opportunity:** Verify class-index mappings, inspect raw logits and class probabilities, standardize preprocessing, and evaluate disagreement against known ground truth.

### 4. Uncertainty quantification

A model's reported confidence score is not automatically a calibrated measure of the probability that its prediction is correct.

**Research opportunity:** Investigate calibration methods and uncertainty-aware decision rules. Evaluate calibration with appropriate metrics, such as expected calibration error and Brier score, alongside accuracy and discrimination metrics.

### 5. Limited explainability

An explanation based only on changes in prediction scores does not necessarily reveal the features or reasoning responsible for a decision.

**Research opportunity:** Compare explanation methods, investigate whether highlighted regions faithfully reflect model behaviour, and measure explanation stability across transformations.

### 6. Narrow modality coverage

The current prototype focuses on image classification. It does not establish the authenticity of audio, video, text, or multimodal content.

**Research opportunity:** Extend evaluation to video, audio, and cross-modal consistency, while carefully validating each modality-specific detector.

### 7. Security against adaptive attacks

Ordinary image transformations are not equivalent to adversarial attacks. A malicious actor may deliberately modify content to evade a detector.

**Research opportunity:** Develop controlled threat models and evaluate both adaptive attacks and defensive strategies under clearly defined assumptions.

## Future Research Roadmap

The following directions represent proposed work, not capabilities already validated by the current prototype.

### Phase 1 — Validate the existing pipeline

- Inspect raw model outputs and class-index mappings.
- Verify the conversion from model outputs to reported AI-class scores.
- Standardize and document preprocessing.
- Add tests for score extraction and label interpretation.
- Record model versions and experimental settings.

### Phase 2 — Build a reproducible benchmark

- Collect a diverse, appropriately licensed dataset of authentic and AI-generated images.
- Record provenance, generation methods, and transformations where known.
- Use independent data splits to reduce leakage.
- Evaluate accuracy, precision, recall, F1 score, ROC-AUC, and relevant calibration metrics.
- Report confidence intervals and performance across different generator families.

### Phase 3 — Improve reliability and uncertainty handling

- Establish rules for reporting model agreement and disagreement.
- Evaluate whether calibrated scores improve decision-making.
- Investigate abstention or an “inconclusive” outcome when evidence is insufficient.
- Measure performance under distribution shift and common image transformations.
- Test whether robustness results generalize across datasets.

### Phase 4 — Develop faithful explanations

- Compare occlusion-based analysis with established explanation techniques.
- Evaluate whether explanations reflect the actual behaviour of the detector.
- Measure explanation stability under input transformations.
- Clearly distinguish observed model behaviour from hypotheses about why a model made a prediction.

### Phase 5 — Extend toward AI Security

- Investigate adaptive evasion and adversarial robustness under controlled threat models.
- Explore ensembles and evidence aggregation without assuming that combining models automatically improves reliability.
- Consider multimodal detection and cross-modal consistency checking.
- Explore LLM-assisted reporting that translates technical findings into understandable explanations without inventing evidence.
- Evaluate explanation faithfulness, uncertainty communication, and usefulness for human decision-makers.

## Project Structure

The current repository is organized as follows:

```text
stresslens/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── config.toml
└── stresslens/
    ├── __init__.py
    ├── detector.py
    ├── secondary_detector.py
    ├── evaluation.py
    ├── explain.py
    ├── reliability.py
    ├── robustness.py
    ├── llm_note.py
    └── pdf_report.py
```

The application entry point is `app.py`. The `stresslens/` package contains the detector integrations, evaluation and robustness logic, explanation utilities, and report-generation components.

The `Image for model test/` directory in the repository contains an example image used during exploratory testing. It should not be treated as a representative benchmark dataset.

## Installation

### Prerequisites

- Python compatible with the dependencies in `requirements.txt`
- Git
- Internet access to download pretrained models from Hugging Face on first use

### 1. Clone the repository

```bash
git clone https://github.com/ms-afridi/stresslens.git
cd stresslens
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running StressLens

Start the Streamlit application from the project root:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the application.

The first model-loading run may take longer because pretrained model files may need to be downloaded. Runtime and memory requirements depend on the selected models and hosting environment.

## Responsible Use

StressLens is an experimental research prototype, not a forensic authentication system.

- A detector's prediction is not proof that an image is authentic or synthetic.
- Model scores should not be interpreted as calibrated probabilities unless calibration has been evaluated.
- Disagreement between models does not independently establish which prediction is correct.
- Robustness under a limited set of transformations does not guarantee resistance to adversarial attacks.
- Results should not be used as the sole basis for high-stakes decisions involving people, organizations, or public claims.
- Images should be handled with appropriate attention to privacy, consent, copyright, and data retention.

Users and researchers should consult the model cards and licensing conditions of the underlying models before redistribution or deployment.

## Research Direction

StressLens provides a starting point for exploring a broader research question:

> **How can synthetic-media detection systems become more reliable, robust, and explainable when facing model disagreement, distribution shift, and adversarial manipulation?**

The long-term direction is to move beyond isolated predictions toward evidence-based reliability assessment, transparent uncertainty communication, faithful explanations, and systematic evaluation.

This direction connects the prototype to research themes in AI Security, adversarial machine learning, synthetic-media detection, and Explainable AI.

## Acknowledgements

StressLens uses pretrained models published on Hugging Face:

- [`dima806/ai_vs_real_image_detection`](https://huggingface.co/dima806/ai_vs_real_image_detection)
- [`capcheck/ai-human-generated-image-detection`](https://huggingface.co/capcheck/ai-human-generated-image-detection)

Please consult the respective model cards for training details, intended use, limitations, and licensing.

## License

See [`LICENSE`](LICENSE) for the license applicable to this repository. The licensing conditions of the underlying pretrained models and third-party dependencies may differ and should be checked separately.
