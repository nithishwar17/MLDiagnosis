import os
import subprocess
import zipfile
import shutil

def download_dataset():
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    data_dir = "data"
    zip_path = os.path.join(data_dir, "chest-xray-pneumonia.zip")
    extract_dir = os.path.join(data_dir, "chest_xray")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(extract_dir):
        print(f"Dataset already exists at {extract_dir}. Skipping download.")
        return

    print(f"Downloading dataset {dataset_name} using Kaggle CLI...")
    try:
        # Run kaggle datasets download command
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", data_dir],
            check=True
        )
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Ensure kaggle is installed (pip install kaggle) and your kaggle.json is in ~/.kaggle/")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset. Ensure your kaggle.json is correctly configured. {e}")
        return
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    # The extraction usually creates a `chest_xray` folder inside data/
    # Sometimes it extracts directly or creates another nested structure.
    # Let's clean up the zip file.
    os.remove(zip_path)
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_dataset()
