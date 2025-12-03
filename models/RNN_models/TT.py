import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the 1D CNN model for pixel-wise classification
class PixelCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(PixelCNN, self).__init__()
        self.conv1d = nn.Conv1d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1d(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Example usage
input_length = 224 * 224  # Length of the input pixel (number of spectral bands)
num_classes = 10  # Number of classes for pixel-wise classification
num_channel = 200
# Create an instance of the PixelCNN model
model = PixelCNN(num_channel, num_classes)

# Create a random input tensor
batch_size = 32
input_tensor = torch.randn(batch_size, num_channel, input_length )

# Forward pass
output = model(input_tensor)
print(output.shape)  # Print the shape of the output tensor