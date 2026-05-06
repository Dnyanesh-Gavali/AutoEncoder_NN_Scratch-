import numpy as np

class DenseLayer:
    def __init__(self, ip_size, op_size):

        # Generates a matrix of shape (input_size, output_size) 
        # filled with random numbers from a standard normal distribution
        # the values are then scaled to very small number by multiplying 0.01
        self.weights = np.random.randn(ip_size, op_size) * 0.01

        # Creates a row vector of zeros with shape (1, output_size).
        self.biases = np.zeros((1, op_size))

        self.inputs = 0


    def forward(self, inputs):
        self.inputs = inputs

        # The core forward math: Z = XW + b
        return np.dot(inputs, self.weights) + self.biases


    #We want:
    # How does loss change w.r.t weights →  ∂W / ∂L
    # How does loss change w.r.t bias → ∂b / ∂L
    # How does loss change w.r.t inputs →  ∂X / ∂L

    #Each layer:
    # Figures out how much it messed up (gradients)
    # Fixes itself (updates weights)
    # Passes blame backward (returns dInputs)
    def backward(self, dZ, learning_rate):
        # dZ is the gradient of the error coming from the layer ahead of this one

        # 1. Calculate the gradient for the weights: dW = X^T * dZ
        # ^T means transpose (.T)
        dW = np.dot(self.inputs.T, dZ)

        # 2. Calculate the gradient for the biases: db = sum of dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        # 3. Calculate the error to pass backward to the previous layer: dX = dZ * W^T
        dInputs = np.dot(dZ, self.weights.T)

        # Update the layer's parameters (Gradient Descent)
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        
        return dInputs
    

# Relu Activation 
# The network needs non-linearity to learn complex patterns. 
# We will use ReLU: f(x) = max(0, x).
# The derivative of ReLU is incredibly simple: it's 1 if the input was greater than 0, and 0 otherwise.

class ReLU:
    def __init__(self):
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, dValues, learning_rate=None):
        # learning_rate isn't used here since activation has no learnable weights, 
        # but we keep it in the signature for consistency
        
        dInputs = dValues.copy()
        dInputs[self.inputs <= 0] = 0
        return dInputs
    
#  The network needs to know how wrong its reconstructed observation x ^ (x-cap)
#  is compared to the original data x.
class MSELoss:
    def forward(self, y_pred, y_true):
        # y_pred is reconstructed image, y_true is the original image
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred, y_true):
        samples = y_true.shape[0] * y_true.shape[1]
        return 2 * (y_pred - y_true) / samples
    
class Autoencoder:
    def __init__(self):
        # Encoder: 64 pixels down to 16
        self.encoder_layer = DenseLayer(64, 32)
        self.encoder_activation = ReLU()
        
        # Decoder: 16 pixels back to 64
        self.decoder_layer = DenseLayer(32, 64)
        self.decoder_activation = ReLU()
        
        self.loss_fn = MSELoss()

    def forward(self, x):
        # Forward pass through the network
        z = self.encoder_layer.forward(x)
        z = self.encoder_activation.forward(z)
        
        x_hat = self.decoder_layer.forward(z)
        x_hat = self.decoder_activation.forward(x_hat)

        return x_hat
    
    def train_step(self, x, learning_rate):
        # 1. Forward Pass
        x_hat = self.forward(x)
        
        # 2. Calculate Loss
        loss = self.loss_fn.forward(x_hat, x)

        # 3. BackPropagation
        dLoss = self.loss_fn.backward(x_hat, x)
        # Pass the gradients backward through the network in reverse order
        dDecoderAct = self.decoder_activation.backward(dLoss)
        dDecoderLayer = self.decoder_layer.backward(dDecoderAct, learning_rate)
        
        dEncoderAct = self.encoder_activation.backward(dDecoderLayer)
        dEncoderLayer = self.encoder_layer.backward(dEncoderAct, learning_rate)
        
        return loss


#Printing image on console
def print_Image(flat_array, title):
    print(f"\n-----{title}-----")
    grid = flat_array.reshape(8, 8)
    for row in grid:
        pixels = ["██" if val > 0.3 else "  " for val in row]
        print("".join(pixels)) 

from sklearn.datasets import load_digits
print("Loading real handwritten digits...")
digits = load_digits()

train_data = digits.images[:100].reshape(100, 64) / 16

model = Autoencoder()
epochs = 40000
lr = 0.01

print(f"Training on {train_data.shape[0]} images. This might take a few seconds...")

for epoch in range(epochs):
    loss = model.train_step(train_data, lr)
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.5f}")


print("\nTraining complete. Testing Reconstruction...")

# Test it on the first image in dataset (which is a '0')
test_image = train_data[0:1]
reconstructed_image = model.forward(test_image)

print_Image(test_image[0], "Original Image (Digit '0')")
print_Image(reconstructed_image[0], "Network's Reconstruction")