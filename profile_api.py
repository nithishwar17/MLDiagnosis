import time
import requests
import os

print("Testing inference time directly via API...")
file_path = "data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg" # Using a known sample
if not os.path.exists(file_path):
    print("Cannot find test image.")
else:
    with open(file_path, "rb") as f:
        files = {"file": f}
        start = time.time()
        res = requests.post("http://localhost:8000/predict", files=files)
        end = time.time()
        
    print(f"HTTP Request Time: {end - start:.3f} sec")
    if res.status_code == 200:
        print(f"Internal Inference Time (from API JSON): {res.json().get('inference_time_sec')} sec")
    else:
        print(f"API Error: {res.text}")
