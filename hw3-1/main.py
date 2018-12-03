import numpy as np
import tensorflow as tf
from gan import GAN
from wgan import WGAN
from dcgan import DCGAN
import os
import argparse

parser = argparse.ArgumentParser(description='tensorflow implementation of WGAN')
parser.add_argument('-mode', type=str, default='train', help='train or infer mode')
parser.add_argument('-curr_epoch', type=int, default='0', help='loads model trained for curr_epoch epochs, 0 for untrained')
parser.add_argument('-epoch', type=int, help='epochs to train')
args = parser.parse_args()
mode = getattr(args, 'mode')
curr_epoch = getattr(args, 'curr_epoch')
epochs = getattr(args, 'epoch')

batch_size = 128
noise_dim = 128
g_iter = 2
d_iter = 1

# change curr_epoch, 0 for untrained
# curr_epoch = 100
restore_model_dir = './model/model_'+str(curr_epoch)+'/'
print(restore_model_dir)

model = DCGAN(noise_dim)

if mode == 'train':
	model.train(curr_epoch, epochs, batch_size, g_iter, d_iter, restore_model_dir)
if mode == 'infer':
	model.infer(restore_model_dir)