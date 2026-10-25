# StressLens
### Investigating Synthetic-Image Detection, Model Failure, Robustness, and Explainable AI

StressLens is an experimental research prototype designed to investigate the reliability of AI-generated image detection systems. It combines two pretrained image classifiers with image-transformation tests, comparative model analysis, and structured reporting to explore how detection systems behave when their predictions may be unreliable.

The project focuses on a central research question:

> **How can we identify, evaluate, and explain systematic failures in AI-generated image detection systems, particularly when models produce misleading or contradictory predictions?**

StressLens is not presented as a fully reliable deepfake detection solution. Instead, it provides a practical foundation for investigating model disagreement, robustness, uncertainty, and explainability in synthetic-media detection.

**Project status:** Early-stage research prototype. The observed detector failure patterns are preliminary and require systematic evaluation on labelled datasets.

---

## Table of Contents

- [Motivation](#motivation)
- [Research Objectives](#research-objectives)
- [How StressLens Works](#how-stresslens-works)
- [Detection Models](#detection-models)
- [Observed Model Failure Pattern](#observed-model-failure-pattern)
- [Robustness Testing](#robustness-testing)
- [Research Gaps and Limitations](#research-gaps-and-limitations)
- [Future Research and Proposed Solutions](#future-research-and-proposed-solutions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Responsible Use](#responsible-use)
- [Research Significance](#research-significance)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Motivation

AI image-generation systems can now produce highly realistic images. Although these technologies have many beneficial applications, synthetic media can also be misused for misinformation, impersonation, fraud, and social engineering.

Automated image detectors offer one way to examine potentially synthetic content. However, a detector's prediction is not necessarily correct, and its performance may deteriorate when it encounters unfamiliar image generators, different image-processing pipelines, or data outside its training distribution.

During exploratory testing of StressLens, a concerning pattern was observed: the primary detector appeared to classify images believed to be real as fake, while the secondary detector appeared to classify an AI-generated test image as human-generated.

This raised a broader question: **What is the value of an AI detector if its predictions appear systematically misleading, and how can such failures be detected before users rely on its outputs?**

StressLens was developed as an initial platform for exploring this problem.

## Research Objectives

The current project aims to:

- Compare predictions from two pretrained AI-image detection models.
- Investigate apparent systematic misclassification and disagreement between models.
- Examine changes in model predictions and scores under common image transformations.
- Explore occlusion-based testing to identify regions that may influence model outputs.
- Present detector results and limitations in structured, human-readable reports.
- Identify methodological improvements for evaluating reliability, robustness, uncertainty, and explainability.

## How StressLens Works

StressLens follows a comparative image-analysis workflow.

1. **Image input:** A user provides an image for analysis.
2. **Primary detection:** The image is passed to the primary pretrained classifier.
3. **Secondary detection:** The same image is evaluated using a second pretrained classifier.
4. **Prediction comparison:** The application compares predicted labels and reported scores.
5. **Transformation testing:** Modified versions of the image are evaluated using operations such as cropping, blurring, JPEG compression, and resizing.
6. **Occlusion analysis:** Selected image regions are hidden and the resulting score changes are examined.
7. **Reliability reporting:** The application organizes the observations into a structured report intended to make model behaviour easier to inspect.

This workflow helps expose differences between models and sensitivity to input changes. It does not independently establish whether an image is authentic.

## Detection Models

StressLens currently integrates two pretrained image-classification models published on Hugging Face.

### 1. Primary detector

**Model:** [`dima806/ai_vs_real_image_detection`](https://huggingface.co/dima806/ai_vs_real_image_detection)

This model classifies images using real and fake categories. StressLens uses its output as one source of evidence when investigating whether an image may be AI-generated.

During exploratory testing, the primary detector appeared to predict **FAKE even for images believed to be real**. If confirmed across a representative evaluation dataset, this behaviour would indicate a significant reliability problem.

Potential explanations to investigate include:

- Poor generalization to the images being tested.
- Dataset bias or a mismatch between training and evaluation distributions.
- Differences in image preprocessing.
- Incorrect class-index interpretation or score extraction in the application.
- Limitations associated with the image-generation methods represented in the test data.

These are hypotheses, not established causes. The model's raw outputs and label mappings must be checked before drawing conclusions.

### 2. Secondary detector

**Model:** [`capcheck/ai-human-generated-image-detection`](https://huggingface.co/capcheck/ai-human-generated-image-detection)

This model distinguishes human-generated images from AI-generated images. StressLens uses it as a second source of predictions for comparison with the primary detector.

In exploratory testing, the secondary detector appeared to predict **human-generated even for an image identified as AI-generated**.

If systematic testing confirms this pattern, possible explanations include poor generalization, training-data limitations, preprocessing differences, or incorrect interpretation of the model's output labels and scores.

The current observations do not establish which explanation is correct.

### Why compare two models?

The purpose of using two detectors is to expose differences in their behaviour, not to assume that agreement proves authenticity.

Two models can disagree because they learn different visual patterns or respond differently to distribution shifts. They can also make correlated errors. A majority vote or a simple average of their scores would therefore not automatically produce a more reliable result.

A meaningful comparison requires validated class mappings, consistent evaluation data, and ground-truth labels.

---

## Observed Model Failure Pattern

The current prototype has raised a specific concern about the reliability of both detection models.

| Component | Preliminary observation | Why it matters |
|---|---|---|
| Primary detector | Appeared to classify images believed to be real as fake. | May produce false alarms if the behaviour persists. |
| Secondary detector | Appeared to classify an AI-generated test image as human-generated. | May fail to flag synthetic content if the behaviour persists. |
| Model comparison | The detectors produced conflicting predictions on an AI-generated test image. | Neither prediction should automatically be treated as definitive. |
| Reliability analysis | Score changes under image transformations were examined. | Stability alone cannot establish correctness. |

For one exploratory AI-generated image, the report recorded:

- **Primary detector:** `FAKE`, with a reported AI-class score of approximately `0.7784`.
- **Secondary detector:** `human`, with a reported AI-class score of approximately `0.0105`.

These are observations from an individual test, not estimates of overall model performance.

The pattern that the detectors appear to favour opposite classes is a hypothesis based on exploratory testing. It must be tested on a diverse, independently labelled dataset containing both authentic and AI-generated images before describing either model as consistently biased or systematically failing.

### Why this matters

A detector that produces misleading predictions can create two kinds of risk:

- **False positives:** authentic images may be incorrectly labelled as AI-generated.
- **False negatives:** AI-generated images may be incorrectly labelled as authentic.

Both errors matter in security-sensitive settings. False positives can damage trust or reputation, while false negatives can allow synthetic media to go undetected.

The key research problem is therefore not merely to obtain a prediction, but to establish when that prediction is dependable and how its limitations should be communicated.

---

## Robustness Testing

StressLens includes image-transformation experiments to examine how detector outputs respond to changes in the input image.

| Transformation | Purpose |
|---|---|
| Cropping | Examines sensitivity to image framing and visible content. |
| Blurring | Tests sensitivity to reduced image detail. |
| JPEG compression | Examines the effects of lossy compression. |
| Resizing | Investigates the effects of image scaling and resampling. |
| Occlusion | Examines how hiding selected regions affects model scores. |

### Score-change analysis

For a model score \(s\), the absolute change between the original and transformed image can be expressed as:

\[
\Delta s = |s_{\mathrm{transformed}} - s_{\mathrm{original}}|
\]

A small change suggests that the score was stable under that particular transformation. A large change suggests sensitivity to the modification.

However, **a stable prediction can still be wrong**. Robustness testing must therefore be interpreted alongside ground-truth accuracy, calibration, and evaluation across different datasets.

Common image transformations also do not cover every possible adversarial attack. Deliberately designed evasion methods require separate threat models and controlled experiments.

---

## Research Gaps and Limitations

### 1. No comprehensive benchmark evaluation yet

Exploratory examples cannot establish model accuracy, false-positive rates, false-negative rates, or generalization.

**Research gap:** The detectors need to be evaluated on a sufficiently large and diverse dataset with verified labels.

### 2. Possible systematic misclassification

The observed predictions raise the possibility that one or both models may perform poorly on the tested image distribution.

**Research gap:** It remains unclear whether the behaviour results from model limitations, distribution shift, preprocessing, label mapping, or score extraction.

### 3. Unverified score calibration

A model-reported score is not automatically a calibrated probability of authenticity or synthetic origin.

**Research gap:** Score semantics must be verified, and calibration should be measured against known labels before probabilities are presented as reliable uncertainty estimates.

### 4. Limited generalization testing

AI-generated images can differ considerably across generators, prompts, editing methods, and post-processing pipelines.

**Research gap:** Cross-generator and out-of-distribution evaluations are needed to determine how performance changes on unfamiliar content.

### 5. Limited explainability

A score change after occlusion can reveal sensitivity to a region, but it does not necessarily explain the complete reasoning behind a prediction.

**Research gap:** Explanation methods need to be assessed for faithfulness, stability, and usefulness rather than judged solely by visual appearance.

### 6. Limited adversarial evaluation

Cropping, blur, resizing, and compression are useful robustness tests, but they do not constitute a comprehensive adversarial security evaluation.

**Research gap:** Future work should test defined attack scenarios and defensive strategies under explicit threat models.

### 7. Image-only scope

The current prototype focuses on images and does not establish the reliability of audio, video, or multimodal detection.

**Research gap:** Multimodal methods could be investigated as a future extension, with independent evaluation of each modality and cross-modal consistency.

---

## Future Research and Proposed Solutions

The following roadmap describes possible research directions. These are proposed improvements, not capabilities already validated by StressLens.

### Phase 1: Verify model outputs and implementation

Before changing or replacing the models, the first step is to establish that the application interprets their outputs correctly.

Planned work:

- Inspect each model's configuration and class-index mapping.
- Compare raw logits, predicted class indices, and post-processed scores.
- Verify that the application maps `REAL`, `FAKE`, `human`, and `AI-generated` labels consistently.
- Confirm image preprocessing against each model's intended input requirements.
- Add unit tests for label mapping and score extraction.
- Record model versions and evaluation settings.

**Goal:** Distinguish implementation errors from genuine model-performance limitations.

### Phase 2: Evaluate on labelled data

Build an evaluation dataset containing authentic photographs and AI-generated images from multiple sources.

Planned work:

- Define inclusion criteria and verify ground-truth labels.
- Separate development data from held-out evaluation data.
- Avoid duplicates and prevent data leakage where possible.
- Evaluate each model independently before comparing their outputs.
- Report accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and confidence intervals.
- Examine false positives and false negatives separately.

**Goal:** Determine whether the suspected failure pattern persists and quantify its severity.

### Phase 3: Investigate distribution shift

Test the models across image sources and generation methods that differ from the data used during development.

Planned work:

- Evaluate across multiple image-generation systems.
- Compare original images with transformed versions.
- Examine performance on unfamiliar generators and post-processing pipelines.
- Identify conditions associated with sharp performance degradation.

**Goal:** Understand when and why the detectors fail to generalize.

### Phase 4: Improve uncertainty and decision-making

If evaluation shows that the detectors are unreliable in certain conditions, the application should communicate that limitation rather than present a definitive classification.

Planned work:

- Evaluate probability calibration using suitable metrics.
- Investigate uncertainty-aware thresholds.
- Introduce an inconclusive or abstention outcome when evidence is insufficient.
- Report disagreement explicitly.
- Compare decision rules against ground-truth evaluation results.

**Goal:** Reduce the risk that users mistake a model output for proof of authenticity.

### Phase 5: Investigate explanations and adversarial robustness

Planned work:

- Compare occlusion analysis with established feature-attribution methods.
- Evaluate explanation faithfulness and stability.
- Test controlled adversarial and evasion scenarios.
- Explore defensive methods and model ensembles.
- Investigate whether explanations help users make better-informed decisions.

**Goal:** Move from reporting predictions to evaluating whether the system's behaviour can be understood and trusted under defined conditions.

### Phase 6: Explore multimodal security

A later version could investigate synthetic audio, video, and text, including consistency between modalities.

Potential directions include foundation-model-based approaches, cross-modal consistency checking, and LLM-assisted explanations. These extensions would require new models, evaluation datasets, and security testing.

**Goal:** Explore how evidence from multiple modalities might contribute to synthetic-media analysis without assuming that multimodal systems are automatically more reliable.

---

## Project Structure

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

- `app.py` — Streamlit application entry point.
- `detector.py` — primary detector integration.
- `secondary_detector.py` — secondary detector integration.
- `evaluation.py` — evaluation utilities.
- `robustness.py` — image-transformation testing.
- `reliability.py` — reliability-analysis utilities.
- `explain.py` — explanation-related functionality.
- `llm_note.py` — LLM-based reporting functionality, subject to its configured dependencies.
- `pdf_report.py` — PDF report generation.

The `Image for model test/` directory contains an example image used during exploratory testing. It is not a representative benchmark dataset, and the image alone cannot establish model accuracy.

---

## Installation

### Prerequisites

- Python version compatible with the packages in `requirements.txt`
- Git
- Internet access to download pretrained models when required

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

## Run the Application

Start StressLens from the repository root:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

The initial run may take longer while pretrained models are downloaded. Resource requirements and execution time depend on the models, dependencies, and hosting environment.

---

## Responsible Use

StressLens is an experimental research prototype. It is not a forensic authentication service and should not be used as the sole basis for high-stakes decisions.

- A prediction does not prove that an image is authentic or AI-generated.
- A model's reported score should not be described as a calibrated probability unless calibration has been evaluated.
- Agreement between two models does not guarantee correctness.
- Disagreement should be reported transparently rather than resolved by arbitrarily choosing one detector.
- Robustness under a limited set of transformations does not guarantee resistance to adversarial attacks.
- Images should be handled with appropriate attention to privacy, consent, copyright, and data retention.
- Users should consult the underlying models' documentation and licensing conditions before deployment or redistribution.

## Research Significance

StressLens began as a practical exploration of synthetic-image detection but has raised a broader reliability question: **what should an AI security system do when its detectors appear to produce systematically misleading predictions?**

The project creates a foundation for investigating model validation, distribution shift, robustness, uncertainty quantification, and explainable AI. Its next stage is to move from exploratory observations to systematic, reproducible evaluation.

The long-term aim is to investigate how synthetic-media detection systems can communicate not only their predictions, but also the limitations and uncertainty associated with those predictions.

## Acknowledgements

StressLens uses pretrained models published on Hugging Face:

- [`dima806/ai_vs_real_image_detection`](https://huggingface.co/dima806/ai_vs_real_image_detection)
- [`capcheck/ai-human-generated-image-detection`](https://huggingface.co/capcheck/ai-human-generated-image-detection)

Please consult each model card for information about training data, intended use, limitations, and licensing.

## License

See [`LICENSE`](LICENSE) for the license applicable to this repository. The underlying models and third-party dependencies may have separate licensing conditions.
