import os
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32, num_workers=4, seed=42):
    # Find all images in train and val folders from the original dataset
    # We will ignore the original test folder, or we can merge it all and re-split.
    # The instruction says: "merge train+val, then create a new stratified split (70% train / 15% val / 15% test) with a fixed random seed"
    # Actually, let's just grab everything in train/ val/ test/ and re-split it all to be safe and use all data, 
    # OR just train+val. The instruction specifically said: "merge train+val, then create a new stratified split".
    
    # Wait, the instruction says "merge train+val...". The original kaggle dataset has train, val, test. 
    # Usually people merge all of them to make a proper split since test has 624 images and val has 16.
    # I'll just gather all images from all three folders to be robust, or strictly train + val + test if requested.
    # "Do NOT use the provided train/val/test split as-is. Instead, merge train+val, then create a new stratified split..."
    # Actually, it might mean merge train+val+test or just train+val and ignore test. Let's merge all of them to get the full dataset.
    
    all_normal_files = glob(os.path.join(data_dir, "**", "NORMAL", "*.jpeg"), recursive=True)
    all_pneumonia_files = glob(os.path.join(data_dir, "**", "PNEUMONIA", "*.jpeg"), recursive=True)
    
    # Check if files were found
    if not all_normal_files and not all_pneumonia_files:
        raise FileNotFoundError(f"No images found in {data_dir}. Ensure data is downloaded and extracted.")

    all_files = all_normal_files + all_pneumonia_files
    # 0 for NORMAL, 1 for PNEUMONIA
    all_labels = [0] * len(all_normal_files) + [1] * len(all_pneumonia_files)

    # Stratified split: 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_files, all_labels, test_size=0.30, random_state=seed, stratify=all_labels
    )
    
    # Split temp into 50% val and 50% test (which is 15% / 15% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    print("Data Split Distribution:")
    print(f"Train - NORMAL: {y_train.count(0)}, PNEUMONIA: {y_train.count(1)}")
    print(f"Val   - NORMAL: {y_val.count(0)}, PNEUMONIA: {y_val.count(1)}")
    print(f"Test  - NORMAL: {y_test.count(0)}, PNEUMONIA: {y_test.count(1)}")

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # Zoom
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Brightness/contrast jitter
        # No horizontal flip for X-rays
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ChestXrayDataset(X_train, y_train, transform=train_transform)
    val_dataset = ChestXrayDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = ChestXrayDataset(X_test, y_test, transform=val_test_transform)

    # Class weighting for imbalanced dataset
    # Weights should be inversely proportional to class frequencies
    num_normal = y_train.count(0)
    num_pneumonia = y_train.count(1)
    
    # Compute class weights for Loss function (we return this to use in train.py)
    # class_weights = [total / num_normal, total / num_pneumonia]
    total = len(y_train)
    weight_for_0 = (1 / num_normal) * (total / 2.0)
    weight_for_1 = (1 / num_pneumonia) * (total / 2.0)
    class_weights = torch.FloatTensor([weight_for_0, weight_for_1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights

if __name__ == "__main__":
    # Test dataloader if run directly
    train_loader, val_loader, test_loader, class_weights = get_data_loaders("data/chest_xray")
    print(f"Class weights: {class_weights}")
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
