import numpy as np
import tensorflow as tf
from wgan import WGAN

epochs = 5000
batch_size = 64
noise_dim = 128
g_iter = 1
d_iter = 5

model = WGAN(noise_dim)
model.train(epochs, batch_size, g_iter, d_iter, 2000)


