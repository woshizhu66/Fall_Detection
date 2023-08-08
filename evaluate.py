import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from tcn import TCN

input_size = 9
output_size = 1
num_channels = (64,) * 3 + (128,) * 2
learning_rate = 0.001
kernel_size = 2
dropout = 0.2
num_epochs = 10

# Instantiate the TCN model
model = TCN(input_size, output_size, num_channels, kernel_size, dropout)

# Load the model
model_path = 'model_0.0621.pth'  # Specify the path to the saved model
model.load_state_dict(torch.load(model_path))
# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test loop
model.eval()  # Set the model in evaluation mode
test_loss = 0
predictions = []
actuals = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        pred = pred.squeeze(1)

        # Optionally compute test loss
        t_loss = F.binary_cross_entropy_with_logits(pred, y.float())
        test_loss += t_loss.item()

        # Convert predicted probabilities to binary outputs
        binary_outputs = (torch.sigmoid(pred) > 0.5).cpu().numpy()

        # Append the model predictions and true labels to their respective lists
        predictions.extend(binary_outputs)
        actuals.extend(y.cpu().numpy())
        print(binary_outputs)
        print(y.cpu().numpy())

accuracy = (np.array(predictions) == np.array(actuals)).mean()
print(f'Test Accuracy: {accuracy}')