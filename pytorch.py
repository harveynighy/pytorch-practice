import torch
import numpy as np 
#f = w * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0

# model prediction
def forward(x):
    return w * x

# loss MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient 
# MSE = 1/N * (w * X - y)**2
# dj/dw = 1/N 2x (w*x -y)

def gradient(x, y, y_predicted):
    return np.dot(2 * X, y_predicted - y).mean()

print(f'prediction before training: f(5) = {forward(5):.3f}')