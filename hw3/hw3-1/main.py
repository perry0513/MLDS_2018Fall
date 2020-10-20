import numpy as np
import tensorflow as tf
from gan import GAN
from wgan import WGAN
from dcgan import DCGAN
import os
import argparse

parser = argparse.ArgumentParser(description='tensorflow implementation of WGAN')
parser.add_argument('-type', type=str, help='\'gan\', \'dcgan\' or \'wgan\'')
parser.add_argument('-mode', type=str, default='train', help='train or infer mode (default: train)')
parser.add_argument('-model_dir', type=str, default='', help='relative path to trained model directory (default: not trained)')
parser.add_argument('-save_model', type=str, default='', help='name of saved model, will be saved in \'./model/\' (default: not saved)')
parser.add_argument('-epoch', type=int, default=30, help='epochs to train (default: 30)')
args = parser.parse_args()
mode = getattr(args, 'mode')
type = getattr(args, 'type')
restore_model_dir = getattr(args, 'model_dir')
save_model_name = getattr(args, 'save_model')
epochs = getattr(args, 'epoch')

batch_size = 128
noise_dim = 128
g_iter = 1
d_iter = 1

# change curr_epoch, 0 for untrained
# curr_epoch = 100
if restore_model_dir != '' and not os.path.exists(restore_model_dir):
	print('Error: Model \'{}\' does not exist'.format(restore_model_dir))
	os._exit(0)
	
print(restore_model_dir)

if type == 'gan':
	model = GAN(noise_dim)
elif type == 'dcgan':
	model = DCGAN(noise_dim)
else:
	model = WGAN(noise_dim)

if mode == 'train':
	model.train(epochs, batch_size, g_iter, d_iter, restore_model_dir, save_model_name)
if mode == 'infer':
	model.infer(restore_model_dir)

