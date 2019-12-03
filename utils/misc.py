import os
import gym
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from models.vae import VAE
from models.mdrnn import MDRNNCell

IMG_DIM = 64

def rgb2gray(image):
	gray = np.dot(image, [0.299, 0.587, 0.114]).astype(np.float32)
	# norm = gray / 128.0 - 1.0
	return gray

def resize(image, dim=IMG_DIM):
	img = cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_CUBIC)
	return img

def show_image(img, filename="test.png", save=True):
	if save: plt.imsave(filename, img)
	plt.imshow(img, cmap=plt.get_cmap('gray'))
	plt.show()

def load_other(gpu=True):
	vae_file, rnn_file = [os.path.join("./logs", m, 'best.tar') for m in ['vae', 'mdrnn']]
	assert os.path.exists(vae_file) and os.path.exists(rnn_file), "Either vae or mdrnn is untrained."
	vae_state, rnn_state = [torch.load(fname, map_location=torch.device("cpu")) for fname in (vae_file, rnn_file)]
	vae = VAE(gpu=gpu)
	mdrnn = MDRNNCell(gpu=gpu)
	vae.load_state_dict(vae_state['state_dict'])
	mdrnn.load_state_dict({k.replace('rnn','lstm').replace('_linear','').replace('_l0',''): v for k, v in rnn_state['state_dict'].items()})
	for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
		print("Loaded {} at epoch {} with test loss {}".format(m, s['epoch'], s['precision']))
	return vae, mdrnn

def make_video(imgs, dim, filename):
	video = cv2.VideoWriter(filename, 0, 60, dim)
	for img in imgs:
		video.write(img.astype(np.uint8))
	video.release()

def to_env(env, action):
	action_normal = (1+action)/2
	action_range = env.action_space.high - env.action_space.low
	env_action = env.action_space.low + np.multiply(action_normal, action_range)
	return env_action

def from_env(env, env_action):
	action_range = env.action_space.high - env.action_space.low
	action_normal = np.divide(env_action - env.action_space.low, action_range)
	action = 2*action_normal - 1
	return action