import os
from PIL import Image
import numpy as np

def create_dummy_dataset(base_dir="data/chest_xray", num_images_per_class=20):
    for split in ["train"]: # just put all in train, get_data_loaders will split it
        for cls in ["NORMAL", "PNEUMONIA"]:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            for i in range(num_images_per_class):
                # Create a random noise image
                img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(dir_path, f"dummy_{i}.jpeg"))

if __name__ == "__main__":
    print("Creating dummy dataset...")
    create_dummy_dataset()
    print("Dummy dataset created.")
