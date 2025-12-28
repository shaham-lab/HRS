import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ImageEmbedder(nn.Module):
    """
    A CNN-based module for embedding grayscale images.
    It resizes input images to 28x28, applies 2 convolutional layers,
    followed by pooling and two fully connected layers to produce a 20-dim embedding.
    """

    def __init__(self):
        super(ImageEmbedder, self).__init__()
        # Define CNN layers for embedding
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flatten to 128 dimensions
        self.fc2 = nn.Linear(128, 20)
        # Define the image transform (convert to tensor and normalize)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize the image to 28x28 if needed
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
        ])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second conv + pooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output from (batch_size, 64, 7, 7) to (batch_size, 64*7*7)
        x = F.relu(self.fc1(x))  # Embedding output (128 features)
        # Embed to 20 features
        x = F.relu(self.fc2(x))
        return x
