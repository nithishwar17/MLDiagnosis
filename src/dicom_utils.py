import pydicom
import numpy as np
from PIL import Image
import io

def process_and_anonymize_dicom(file_bytes):
    """
    Reads a raw DICOM (.dcm) file, completely anonymizes the patient's Protected Health 
    Information (PHI) for HIPAA compliance, and extracts the pixel array into a standard PIL Image.
    """
    # Load DICOM from bytes
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    
    # --- DATA PRIVACY: ANONYMIZE PHI (HIPAA Compliance) ---
    phi_tags = [
        'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex', 
        'PatientAge', 'PatientWeight', 'PatientAddress', 'InstitutionName'
    ]
    
    for tag in phi_tags:
        if tag in ds:
            ds.data_element(tag).value = "ANONYMIZED"
            
    # --- EXTRACT IMAGE ---
    pixel_array = ds.pixel_array
    
    # Normalize the pixel array to 0-255 for standard image processing
    pixel_array = pixel_array - np.min(pixel_array)
    if np.max(pixel_array) != 0:
        pixel_array = pixel_array / np.max(pixel_array)
    pixel_array = (pixel_array * 255).astype(np.uint8)
    
    # Convert to PIL Image (RGB for the ResNet model)
    image = Image.fromarray(pixel_array)
    if len(image.getbands()) == 1:
        image = image.convert('RGB')
        
    return image, ds
