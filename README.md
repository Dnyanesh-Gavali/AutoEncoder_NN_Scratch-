# Autoencoder using only numpy :  `Built From Scratch`

This was built inorder to learn all the actual working of neural network by implementing forward pass, backpropagation, dense layers and ReLU

## Overview
This network is designed for dimensionality reduction and image compression, trained on the `scikit-learn` 8x8 handwritten digits dataset.

> [!IMPORTANT]
> *   **Encoder:** Maps a flattened 64-pixel image array into a compressed 32-neuron latent space.
> *   **Decoder:** Reconstructs the 32-dimensional bottleneck vector back into a 64-pixel image.
> *   **Loss Function:** Mean Squared Error (MSE).
> *   **Optimization:** Custom Gradient Descent.

## Key Features Implemented Manually
1.  **State Management:** Custom classes cache forward-pass data required for derivative calculations.
2.  **Calculus Engine:** Manual computation of $dW$ and $db$ gradients for weight updates.
3.  **Overfitting Resolution:** Scaled the latent space bottleneck to correctly map spatial relationships without relying on pixel-averaging heuristics.

## How to Run
Ensure you have NumPy and Scikit-Learn installed:
```bash
pip install numpy scikit-learn

Then RUN

python autoencoder.py