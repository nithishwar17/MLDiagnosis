# 🩺 MLDiagnosis: Chest X-Ray AI Platform

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

### 🔴 [Live Demo: Click Here to Try MLDiagnosis](https://mldiagnosis-2448.streamlit.app)

An enterprise-grade, end-to-end Machine Learning platform that classifies chest X-rays (Standard & DICOM) as NORMAL or PNEUMONIA. Built for scale, reliability, and clinical explainability.

## ✨ Enterprise Features
* **Model Ensembling:** Averages predictions from a Custom CNN and a Transfer Learning ResNet18 model for highly robust consensus predictions.
* **Native DICOM Support & HIPAA Privacy:** Automatically parses raw hospital `.dcm` files and aggressively scrubs/anonymizes Protected Health Information (PHI) before inference.
* **Explainable AI (XAI):** Generates Grad-CAM heatmaps overlaying the original X-ray to highlight the exact pulmonary regions triggering the diagnosis.
* **Confidence Calibration:** Uses Temperature Scaling on the validation set to convert raw neural network logits into reliable, clinically-safe probabilities.
* **Automated Clinical Reports:** Generates downloadable PDF reports containing patient demographics, severity estimations, AI recommendations, and heatmap visuals.
* **Doctor Dashboard & SQLite History:** A local database logs every scan, feeding into an interactive Plotly analytics dashboard for hospital statistics.
* **MLOps & CI/CD:** Fully tracked experiments using MLflow, complete with a GitHub Actions pipeline (`ci_cd.yml`) for automated `pytest` validation.

## 📊 Model Performance (Test Set)
| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | AUC-ROC |
|-------|----------|-----------|----------------------|----------|---------|
| **ResNet18** | **0.972** | **0.971** | **0.992** | **0.981** | **0.996** |
| Baseline CNN | 0.905 | 0.892 | 0.989 | 0.938 | 0.976 |

*(Note: Pneumonia detection prioritizes high Recall/Sensitivity to completely minimize false negatives).*

## 🚀 Quickstart (Local Run)
1. Install requirements: `pip install -r requirements.txt`
2. Start the FastAPI Backend: `python -m uvicorn app.api:app --host 0.0.0.0 --port 8000`
3. Start the Streamlit Frontend: `python -m streamlit run app/streamlit_app.py`

## ☁️ Deploying for Free (Hugging Face Spaces)
This project is pre-configured to deploy on Hugging Face Spaces for free (with 16GB of RAM).
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new **Docker** space.
2. Upload this entire repository.
3. Hugging Face will automatically detect the `Dockerfile.hf` and `run_hf.sh` scripts, spinning up both the API and Web App securely on port `7860`.

## 🛑 Limitations & Clinical Disclaimer
> **⚠️ NOT FOR MEDICAL DIAGNOSIS**
> This application is a portfolio project demonstrating Machine Learning architectures. It is strictly not approved by the FDA or any medical body. Real-world deployment requires rigorous cross-hardware validation (different X-ray machines), demographic bias auditing, and continuous radiologist-in-the-loop oversight.
