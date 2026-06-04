# 🔤 SortTransformer

A from-scratch implementation of the Transformer (Encoder-Decoder) architecture, specifically trained to perform character-level sorting. Given an input word like `transformer`, the model predicts the alphabetically sorted sequence `aeefmnnorrrt`.

![Gradio Demo](https://img.shields.io/badge/Demo-Gradio-orange)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

## 🚀 Overview

This repository contains a production-ready Transformer backbone built using PyTorch. While the primary task is character sorting, the underlying architecture is a general-purpose seq2seq model capable of handling various translation or mapping tasks.

### Key Features
- **Modern Transformer Architecture**: Implements Pre-Layer Normalization for stability, GELU activations, and Weight Tying between target embeddings and the output projection head.
- **Dual Inference Modes**: Support for both **Greedy Decoding** and **Beam Search** (with adjustable beam size and length penalty).
- **Complete Training Pipeline**: Includes label smoothing, a warmup-based learning rate scheduler (inverse square root), gradient clipping, and checkpoint management.
- **Interactive UI**: A built-in Gradio web interface for real-time testing and visualization.
- **Deployment Ready**: Includes a `Dockerfile` optimized for containerized environments and Hugging Face Spaces.

---

## 🏗️ Architecture

The model follows the original *Attention is All You Need* design with several modern enhancements:

- **Backbone**: N-layer Encoder and N-layer Decoder.
- **Attention**: Scaled Dot-Product Multi-Head Attention.
- **Positional Encoding**: Sinusoidal positional embeddings.
- **Stability**: Pre-LN (LayerNorm before sub-layers) to allow for deeper stacks and more stable gradients.
- **Efficiency**: Weight tying between the target embedding layer and the final linear projection layer to reduce parameter count and improve generalization.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Sort-Transformer-Docker.git
   cd Sort-Transformer-Docker
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Interactive Web UI
Launch the Gradio demo to test the model in your browser:
```bash
python app.py
```
By default, the app runs on `http://0.0.0.0:7860`.

### Python API
You can use the model programmatically for inference:
```python
import torch
from model import Transformer
from inference import beam_search

# Load your model and vocab (see app.py for full loading logic)
# ...

src = torch.tensor([[...]]) # Encoded input
output_ids = beam_search(model, src, sos_idx=1, eos_idx=2, beam_size=4)
```

---

## 🏋️ Training

The training script `train.py` provides a robust loop with validation and checkpointing.

To train the model on your own data:
1. Define your dataset in `dataset.py`.
2. Run the training loop:
   ```python
   from train import train
   # Initialize model and dataloaders
   train(model, train_loader, val_loader, ...)
   ```
The trainer automatically handles:
- **Teacher Forcing** during training.
- **Label Smoothing** to prevent overconfidence.
- **Learning Rate Warmup** for stable convergence.

---

## 🐳 Docker Deployment

To build and run the application using Docker:

```bash
docker build -t sort-transformer .
docker run -p 7860:7860 sort-transformer
```

---

## 📂 Project Structure

- `model.py`: Core Transformer architecture.
- `train.py`: Training logic and loss functions.
- `dataset.py`: Vocabulary management and data loading.
- `inference.py`: Greedy and Beam Search decoding algorithms.
- `app.py`: Gradio interface.
- `sort_model.pt`: Pre-trained model weights and vocabulary.

---
