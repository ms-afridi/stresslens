
# StressLens

**Investigating the Reliability of AI-Generated Image Detectors**

StressLens is an experimental Streamlit research application for exploring
AI-generated-image detector outputs, transformation sensitivity, visual
attribution, and evaluation against labelled images.

## Research objectives

- Inspect detector outputs without treating scores as calibrated probabilities.
- Measure prediction sensitivity to predefined image transformations.
- Explore gradient-based attribution and region-occlusion experiments.
- Evaluate binary classification predictions against labelled datasets.
- Document methodological limitations and reproducibility requirements.

## Current status

StressLens is a research prototype. Model loading, output-label mappings,
preprocessing, ONNX artifact configuration, and explanation compatibility
must be verified before interpreting results or using the application
for research claims.

## Project structure

```text
stresslens/
├── app.py
├── pages/
├── stresslens/
├── tests/
├── sample_images/
├── requirements.txt
├── README.md
├── LICENSE
└── .streamlit/
    └── config.toml
```

## Requirements

- Python 3.11 recommended
- pip
- Internet access for downloading model artifacts when required
- Sufficient memory for the selected model

## Local installation

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can use the
environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Launch the application

From the project root:

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal.

## Research modules

- Image Analysis: inspect primary-detector outputs.
- Robustness Lab: examine sensitivity to controlled image transformations.
- Explainability Lab: explore attribution maps and occlusion experiments.
- Dataset Evaluation: evaluate predictions against ground-truth labels.

## Dataset evaluation format

The initial evaluation page expects a CSV containing:

| image_path | label |
|---|---:|
| sample_images/photo1.jpg | 0 |
| sample_images/generated1.png | 1 |

Labels:
- `0`: real image
- `1`: AI-generated image

Image paths must resolve on the machine running the application. A cloud
deployment requires the images to be made available to that environment.

## Responsible interpretation

- Model scores are not necessarily calibrated probabilities.
- Detector agreement does not establish image authenticity.
- Transformation sensitivity on one image does not establish general robustness.
- Attribution maps do not automatically demonstrate explanation faithfulness.
- Evaluation metrics apply to the evaluated dataset and protocol.
- Thresholds should be selected using validation data and evaluated on held-out data.

## Deployment

The intended deployment target is Streamlit Community Cloud or another
compatible Python hosting service. Confirm that model artifacts can be
downloaded, dependencies install correctly, and memory and processing
requirements fit the selected host before publishing a live demo.

## Security and privacy

Do not commit API keys, passwords, access tokens, private datasets, or
unnecessary personal information. Avoid uploading sensitive images.
Review third-party model licences and usage conditions before deployment.

## Limitations

This project is an experimental research prototype, not a forensic
authentication service. It should not be the sole basis for consequential
decisions about individuals.

## License

See the LICENSE file. Third-party models and dependencies may have their
own separate licences and usage conditions.
