#!/bin/bash
# Hugging Face Spaces exposes a single port (7860 by default).
# This script starts BOTH the FastAPI backend and the Streamlit frontend in the same container.

echo "Starting FastAPI Backend..."
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 &

echo "Waiting for Backend to initialize..."
sleep 5

echo "Starting Streamlit Frontend..."
# Streamlit connects to localhost:8000 natively inside this container
python -m streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
