"""Evaluate the trained Fashion-MNIST classifier without retraining.

Loads the saved weights (fashion_model_v2.pth) and reports test accuracy
overall and for each clothing class.

Run:  pip install torch numpy
      python evaluate.py
"""

import torch

from data_loader import get_fashion_mnist_data
from model import FashionClassifier

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

_, _, test_x, test_y = get_fashion_mnist_data("data")

model = FashionClassifier()
model.load_state_dict(torch.load("fashion_model_v2.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    predicted = model(test_x).argmax(dim=1)

accuracy = (predicted == test_y).float().mean().item()
print(f"Test accuracy: {accuracy:.1%}\n")

print("Accuracy by class:")
for i, name in enumerate(CLASSES):
    mask = test_y == i
    class_acc = (predicted[mask] == i).float().mean().item()
    print(f"  {name:<12} {class_acc:.1%}")
