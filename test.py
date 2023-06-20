import torch
import torchvision
from algorithm import MNIST_FullyConnected

def preprocess_input(image):
    # Preprocess the input image to fit the MNIST dataset.
    image = image.convert("L")  # Convert the image to grayscale
    image = image.resize((28, 28))  # Resize the image to 28x28 pixels
    image = torchvision.transforms.ToTensor()(image)  # Convert the image to a PyTorch tensor
    image = 1 - image  # Invert the pixel values (inverting the background to white and the drawing to black)
    return image


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
