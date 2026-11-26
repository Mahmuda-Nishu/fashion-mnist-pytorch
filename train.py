"""Train the Fashion-MNIST classifier and save its weights.

Run:  pip install torch numpy
      python train.py
"""

import torch
import torch.nn.functional as F
from torch import optim

from data_loader import get_fashion_mnist_data
from model import FashionClassifier

train_x, train_y, test_x, test_y = get_fashion_mnist_data("data")
print("Train:", train_x.shape, train_y.shape)
print("Test :", test_x.shape, test_y.shape)

model = FashionClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

batch_size = 64
epochs = 50

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(train_x.size(0))
    epoch_loss, correct, total = 0.0, 0, 0

    for i in range(0, train_x.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        x_batch, y_batch = train_x[indices], train_y[indices]

        preds = model(x_batch)
        loss = F.cross_entropy(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (preds.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    scheduler.step()
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss:.4f} | Train Acc: {correct / total:.4f}")

torch.save(model.state_dict(), "fashion_model_v2.pth")

model.eval()
with torch.no_grad():
    accuracy = (model(test_x).argmax(dim=1) == test_y).float().mean().item()
print(f"Final test accuracy: {accuracy:.4f}")
