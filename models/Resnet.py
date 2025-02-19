import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# Parameters
IMAGE_SIZE = 256           # Resize images to 256x256
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 1            # For regression (single numeric value)

# Custom dataset class (same as before)
class AerialDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns:
               - 'filename': the filename of the image
               - 'group': the numeric label (e.g., landing score)
            root_dir (str): Path to the folder containing images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['filename']))
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)
        
        label = float(row['group'])
        # Create label tensor as float32 with shape [1]
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return image, label_tensor

# Instantiate a pretrained ResNet18 and modify the final layer for regression
model = models.resnet18(pretrained=True)
# Replace the final fully-connected layer (default is nn.Linear(512, 1000))
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load CSV and split data
data = pd.read_csv('../dataset/aerial/data.csv')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Define transforms (resize, to tensor, normalize)
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = AerialDataset(train_data, root_dir='../dataset/aerial/images', transform=transform)
val_dataset   = AerialDataset(val_data,   root_dir='../dataset/aerial/images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define a helper function for predictions on random samples
def predict_random_samples(model, dataset, num_samples=6):
    # Function to reverse normalization for visualization
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        tensor = tensor * std + mean
        return tensor

    model.eval()
    sample_indices = random.sample(range(len(dataset)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*3, 3))
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            image, label = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            prediction = model(image_input)
            pred_value = prediction.item()
            
            # Denormalize for display
            img_disp = denormalize(image).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axes[i].imshow(img_disp)
            axes[i].axis('off')
            axes[i].set_title(f"Pred: {pred_value:.2f}")
    plt.suptitle("Random Validation Samples with Predictions")
    plt.show()

# Check if a pretrained model file exists
model_file = "resnet_model.pth"
if os.path.exists(model_file):
    print(f"Found {model_file}. Loading model and performing predictions only.")
    model.load_state_dict(torch.load(model_file, map_location=device))
    predict_random_samples(model, val_dataset)
else:
    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)  # labels: [batch_size, 1]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Plot Loss vs. Epoch graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the trained model's state dictionary
    torch.save(model.state_dict(), model_file)
    print(f"Model saved as {model_file}")

    # After training, perform predictions on random samples
    predict_random_samples(model, val_dataset)
