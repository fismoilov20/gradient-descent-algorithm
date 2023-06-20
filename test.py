import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def preprocess_input(image):
    # Preprocess the input image to fit the MNIST dataset.
    image = image.convert("L")  # Convert the image to grayscale
    image = image.resize((28, 28))  # Resize the image to 28x28 pixels
    image = torchvision.transforms.ToTensor()(image)  # Convert the image to a PyTorch tensor
    image = 1 - image  # Invert the pixel values (inverting the background to white and the drawing to black)
    return image

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


from PIL import Image
image = Image.open("image.png")

transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.1307,), (0.3081,)),])

input_tensor = preprocess_input(image)  # Preprocess your image
input_normalized = transform(input_tensor)
input_image = torch.Tensor(input_normalized).view(-1, 28 * 28).to(DEVICE)   # Convert to tensor and add batch dimension

# Pass the input tensor through the model to obtain predictions:
with torch.no_grad():
    output = model(input_image)

# Get the predicted label:
predicted_label = torch.argmax(output, dim=1).item()
print(f"Predicted label: {predicted_label}")
