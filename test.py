import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class MNIST_FullyConnected(nn.Module):
    # A fully-connected neural network model designed for the MNIST dataset. This model is not an optimizer
    # itself but can be optimized using different optimization algorithms.
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_FullyConnected, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def forward(self, x):
        # Compute a prediction using the given input data.
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MNIST_FullyConnected(28 * 28, 128, 10).to(DEVICE)
model.load_state_dict(torch.load('model.pth'))
model.eval()

import numpy as np
from PIL import Image
image = Image.open("image.png")
image = image.convert('L')
image = image.resize((28, 28))
image = 255 - np.array(image)
image = image / 255.0
input_image = torch.tensor(image, dtype=torch.float32)
input_image = torch.reshape(input_image, (-1, 28 * 28))


# Pass the input tensor through the model to obtain predictions:
with torch.no_grad():
    input_image = input_image.to(DEVICE)
    output = model(input_image)
    _, predicted_label = torch.max(output, 1)
    print("Predicted Label:", predicted_label.sum().item())
