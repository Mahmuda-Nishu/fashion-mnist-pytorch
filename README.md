# Fashion-MNIST Clothing Classifier (PyTorch)

**A neural network built from scratch in PyTorch: 90.5% test accuracy**

## Goal
Classify 28×28 grayscale images of clothing into 10 categories (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot).

## Data
- [Fashion-MNIST (Zalando Research)](https://github.com/zalandoresearch/fashion-mnist): 60,000 training images and 10,000 test images
- Loaded with a **custom data loader** (`data_loader.py`) that reads the raw compressed `idx` files directly, with no prebuilt dataset loader

## Model
A fully connected network with regularization:

```
784 → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(10)
```

- Optimizer: Adam (learning rate 0.001) with a step learning-rate scheduler
- Loss: cross-entropy · Batch size: 64 · Epochs: 50

## Results
| Metric | Value |
|---|---|
| Training accuracy | ~94.7% |
| **Test accuracy** | **90.5%** |

**Accuracy by class (test set):**

| Class | Accuracy | Class | Accuracy |
|---|---|---|---|
| Trouser | 98.0% | Dress | 91.7% |
| Sneaker | 97.9% | T-shirt/top | 85.8% |
| Bag | 97.3% | Pullover | 84.7% |
| Sandal | 97.0% | Coat | 84.1% |
| Ankle boot | 95.5% | **Shirt** | **73.0%** |

**What this shows:** items with distinct shapes (trousers, shoes, bags) are classified almost perfectly. The model struggles most with **shirts**, which look similar to T-shirts, pullovers and coats at low resolution. A convolutional neural network (CNN), which learns local shapes and textures, would be the next step to improve these classes.

Batch normalization and dropout kept the gap between training and test accuracy small, so the model generalizes instead of memorizing the training set.

## Files
- `data_loader.py`: custom loader for the raw Fashion-MNIST files
- `model.py`: model architecture
- `train.py`: trains the model and saves the weights
- `evaluate.py`: loads the saved weights and reports test accuracy, without retraining
- `fashion_model_v2.pth`: trained weights
- `assignment_notebook.ipynb`: full training process with outputs
- `data/`: the four Fashion-MNIST files

## Tools
Python · PyTorch · NumPy

## How to run
```bash
pip install torch numpy
python evaluate.py   # evaluate the trained model (fast)
python train.py      # retrain from scratch (about 50 epochs)
```

---
*Built for ECON 8310 Business Forecasting, University of Nebraska Omaha.*
