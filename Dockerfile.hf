FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and DICOM processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make the run script executable
RUN chmod +x run_hf.sh

# Hugging Face Spaces strictly routes traffic to port 7860
EXPOSE 7860

# Run both API and Streamlit via the bash script
CMD ["./run_hf.sh"]
