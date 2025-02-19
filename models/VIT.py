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

# Vision Transformer parameters
PATCH_SIZE = 16         # Size of each image patch
EMBED_DIM = 64          # Dimensions of the patch embeddings
NUM_HEADS = 4           # Number of attention heads
NUM_LAYERS = 8          # Number of Transformer blocks
MLP_DIM = 128           # Hidden size in the MLP (feed-forward) part of the transformer
NUM_CLASSES = 1         # Output size (for regression, a single numeric value)
IMAGE_SIZE = 256        # Images are resized to 256x256
DROP_RATE = 0.1         # Dropout rate

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

# Custom dataset class
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

# Vision Transformer components in PyTorch
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, h/p, w/p)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x = x + self.pos_embed
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop_rate=DROP_RATE):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate=DROP_RATE):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_dim, out_features=embed_dim, drop_rate=drop_rate)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # skip connection
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output  # skip connection
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        num_classes=NUM_CLASSES,
        drop_rate=DROP_RATE
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling over patches
        x = self.head(x)
        return x

# Initialize model and send to device
model = VisionTransformer(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_layers=NUM_LAYERS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    num_classes=NUM_CLASSES,
    drop_rate=DROP_RATE
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load CSV and split data
data = pd.read_csv('../dataset/aerial/data.csv')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Define transforms (resize, to tensor, normalize)
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = AerialDataset(train_data, root_dir='../dataset/aerial/images', transform=transform)
val_dataset   = AerialDataset(val_data,   root_dir='../dataset/aerial/images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define a helper function for predictions
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
model_file = "vit_model_load.pth"
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
            labels = labels.to(device)  # labels already shape [batch_size, 1] and float32

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
