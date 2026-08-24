import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
from app.pdf_generator import generate_pdf

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MLDiagnosis | Chest X-Ray AI", page_icon="🩺", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Clinical Theme Header */
    .main-header {
        font-size: 2.8rem;
        color: #0A5C36;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4A5568;
        margin-bottom: 2rem;
    }
    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: #FFF3CD;
        color: #856404;
        padding: 12px;
        border-radius: 5px;
        border-left: 5px solid #FFEBA8;
        margin-bottom: 25px;
        font-size: 0.95rem;
    }
    /* Custom Cards */
    .result-card-normal {
        background-color: #D4EDDA;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #28A745;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card-pneumonia {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #DC3545;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Adjust container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER & DISCLAIMER ---
st.markdown('<div class="disclaimer-banner"><strong>⚠️ DISCLAIMER:</strong> This tool is for educational/portfolio purposes only and is not a certified diagnostic device. Do not use for clinical decision making.</div>', unsafe_allow_html=True)

st.markdown('<div class="main-header">🩺 MLDiagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced AI Chest X-Ray Analysis Platform</div>', unsafe_allow_html=True)
st.markdown("---")

# --- LAYOUT ---
col_left, col_spacing, col_right = st.columns([4, 1, 6])

with col_left:
    st.subheader("📋 Patient Information")
    with st.container():
        patient_name = st.text_input("Patient Name", placeholder="e.g., Jane Doe")
        col_age, col_gen = st.columns(2)
        with col_age:
            patient_age = st.number_input("Age", min_value=1, max_value=120, value=30)
        with col_gen:
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("Reported Symptoms", placeholder="e.g., Persistent cough, fever...")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📤 Medical Imaging")
    uploaded_file = st.file_uploader("Upload Chest X-Ray (JPEG/PNG/DICOM)", type=["jpg", "jpeg", "png", "dcm"])

with col_right:
    if uploaded_file is not None and patient_name:
        st.subheader("📊 Diagnostic Analysis")
        
        # For displaying in streamlit, if it's DICOM, convert it
        if uploaded_file.name.lower().endswith('.dcm'):
            from src.dicom_utils import process_and_anonymize_dicom
            image, _ = process_and_anonymize_dicom(uploaded_file.getvalue())
            st.info("🔒 DICOM file automatically anonymized for privacy before processing.")
        else:
            image = Image.open(uploaded_file)
        
        with st.spinner("🧠 Analyzing image across ensemble models..."):
            uploaded_file.seek(0)
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    low_conf = result["low_confidence"]
                    heatmap_b64 = result["heatmap_base64"]
                    
                    # Calculate Severity & Recommendations
                    severity = "N/A"
                    recs = ["No immediate pneumonia signs detected.", "Routine follow-up if symptoms persist."]
                    
                    if prediction == "PNEUMONIA":
                        if confidence > 0.90:
                            severity = "Severe Pneumonia (High Confidence)"
                            recs = ["URGENT: Consult pulmonologist immediately.", "Start antibiotic therapy review.", "Monitor oxygen levels."]
                        elif confidence > 0.75:
                            severity = "Moderate Pneumonia"
                            recs = ["Consult pulmonologist.", "Consider further CT scan."]
                        else:
                            severity = "Mild / Ambiguous"
                            recs = ["Human review required.", "Wait and observe or run secondary tests."]

                    # --- RESULT CARD ---
                    if prediction == "PNEUMONIA":
                        st.markdown(f"""
                        <div class="result-card-pneumonia">
                            <h3 style="margin:0; color:#721C24;">🚨 Prediction: PNEUMONIA</h3>
                            <p style="margin:0; margin-top:5px;"><strong>Severity:</strong> {severity}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card-normal">
                            <h3 style="margin:0; color:#155724;">✅ Prediction: NORMAL</h3>
                            <p style="margin:0; margin-top:5px;">No significant opacities detected.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # --- CONFIDENCE BAR ---
                    st.write(f"**Consensus Confidence Score:** {confidence:.2%}")
                    st.progress(float(confidence))
                    
                    if low_conf:
                        st.warning("⚠️ **Low consensus confidence** — models are unsure or disagreeing. Human review highly recommended.")
                    
                    # --- VISUAL COMPARISON ---
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("👁️ Visual Explanation (Grad-CAM)")
                    
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                    
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.image(image, use_container_width=True, caption="Original X-Ray")
                    with img_col2:
                        st.image(heatmap_image, use_container_width=True, caption="Grad-CAM Overlay (Red = High Influence)")

                    # --- RECOMMENDATIONS & METRICS ---
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("🩺 View Clinical Recommendations", expanded=True):
                        for r in recs:
                            st.write(f"- {r}")
                    
                    with st.expander("⚙️ System Metrics"):
                        st.write(f"- **Model Version:** MLDiagnosis Ensemble v2.0 (ResNet18 + Custom CNN)")
                        if "inference_time_sec" in result:
                            st.write(f"- **Inference Time:** {result['inference_time_sec']} seconds")

                    # --- PDF GENERATION ---
                    st.markdown("---")
                    if st.button("📄 Generate & Download Clinical PDF Report", use_container_width=True):
                        # Save to temp paths for PDF
                        os.makedirs("reports", exist_ok=True)
                        orig_path = "reports/temp_orig.jpg"
                        heat_path = "reports/temp_heat.jpg"
                        image.save(orig_path)
                        heatmap_image.save(heat_path)
                        
                        # PDF
                        patient_data = {"doctor": "Local User", "name": patient_name, 
                                        "age": patient_age, "gender": patient_gender, "symptoms": symptoms}
                        pdf_file = generate_pdf(patient_data, prediction, confidence, severity, recs, orig_path, heat_path)
                        
                        with open(pdf_file, "rb") as pdf:
                            st.download_button("⬇️ Click Here to Download PDF", data=pdf, file_name=os.path.basename(pdf_file), mime="application/pdf", use_container_width=True)
                            
            except Exception as e:
                st.error(f"API Error: {str(e)}")
    elif uploaded_file is None:
        st.info("👈 Please enter patient details and upload an image to begin.")
    elif not patient_name:
        st.warning("👈 Please enter the patient's name.")
