# MLDiagnosis Architecture & Data Flow

This document outlines the core components and logic of the MLDiagnosis platform as currently implemented.

## 1. Authentication Flow
* **Implementation:** Uses a raw OAuth 2.0 flow via standard HTTP requests (`requests` library) to Google's OpenID Connect endpoints, triggered by a `google_credentials.json` file. 
* **State Management:** Once authenticated, user details are stored in Streamlit's in-memory `st.session_state`.
* **Production Safety:** This is a **dev/portfolio implementation**. While it successfully authenticates a user against Google, it does not securely validate JWT signatures, handle token rotation, or manage secure HTTP-only cookies, which would be required for a HIPAA-compliant production environment.

## 2. Ensemble Logic (Consensus Confidence)
* **Implementation:** The backend utilizes an unweighted arithmetic average of the probabilities from both models.
* **Location:** `app/api.py` (inside the `/predict` endpoint).
* **Math:** 
  1. `resnet_prob` = calibrated sigmoid output of ResNet18.
  2. `baseline_prob` = sigmoid output of Baseline CNN.
  3. `ensemble_prob = (resnet_prob + baseline_prob) / 2.0`.
  4. Final Prediction relies on `ensemble_prob > 0.5`.

## 3. AI Recommendations 
* **Implementation:** Strictly **Rule-Based / Template-Driven**.
* **Risk Profile:** **Zero risk of hallucination**. No Large Language Models (LLMs) are used to generate medical text.
* **Logic:** Found in `app/streamlit_app.py`. Recommendations are hardcoded strings triggered by confidence thresholds:
  * `> 90%` Confidence -> "Severe Pneumonia (High Confidence)" + Urgent consult template.
  * `> 75%` Confidence -> "Moderate Pneumonia" + Standard consult template.
  * `< 75%` Confidence -> "Mild / Ambiguous" + Human review template.
  * `NORMAL` -> "No immediate pneumonia signs detected."

## 4. Database and Storage (History & PDF)
* **Patient History Database:** Uses a **real**, persistent local SQLite database (`app/healthcare_app.db`). All records and feedback are permanently written to disk using Python's `sqlite3` library (`app/database.py`).
* **PDF Reports:** Uses the `fpdf` library to dynamically compile text and images into a **real** `.pdf` file saved locally to the `reports/` directory (`app/pdf_generator.py`). These are not mocked; they are fully functional local persistence mechanisms.
