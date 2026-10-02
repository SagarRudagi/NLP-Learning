"""
Architecture note:
This is a simple feed-forward neural network (FFN) block used inside transformer architectures.
It represents the core MLP-style component that applies non-linear transformation to each token 
independently after self-attention.

Author note:
This file is meant for learning and understanding the architecture of transformer feed-forward layers.
It is intentionally minimal and educational rather than production-optimized.

Why this network is used:
The FFN allows the model to project token representations into a richer feature space,
then project them back. This helps capture non-linear patterns and interactions that attention alone
may not fully model. In transformers, this block is a key ingredient for expressivity and learning.

Activation functions:
Different activation functions can be used here depending on the architecture and task, such as:
- ReLU: simple and efficient but can suffer from dead neurons.
- GELU: commonly used in transformer models because it performs well in practice.
- SiLU / Swish: smooth, non-monotonic activation often used in modern neural networks.
- Tanh: smooth, zero-centered, but often less effective than modern choices in large transformers.

The choice of activation affects how information flows through the network and impacts learning behavior,
optimization stability and model performance.
"""

import torch

ffn = torch.nn.Sequential(
    torch.nn.Linear(512, 2048),
    torch.nn.GELU(),
    torch.nn.Linear(2048, 512)
)
