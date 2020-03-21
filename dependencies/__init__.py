import os
import gym
import cv2
import numpy as np
import vizdoom as vzd
# from ..utils.misc import resize

IMG_DIM = 96					# The height and width to scale the environment image to

def resize(image, dim=IMG_DIM):
	img = cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_CUBIC)
	return img

configs = sorted([s.replace(".cfg","") for s in sorted(os.listdir("./dependencies/ViZDoom/scenarios/")) if s.endswith(".cfg")])

class VizDoomEnv():
	def __init__(self, env_name, resize=IMG_DIM, transpose=[1,2,0], render=True):
		self.transpose = transpose
		self.env = vzd.DoomGame()
		self.env.load_config(os.path.abspath(f"./dependencies/ViZDoom/scenarios/{env_name}.cfg"))
		self.action_space = gym.spaces.Discrete(self.env.get_available_buttons_size())
		self.size = [self.env.get_screen_channels()] + ([resize, resize] if resize else [self.env.get_screen_height(), self.env.get_screen_width()])
		self.sizet = [self.size[x] for x in transpose] if self.transpose else self.size
		self.observation_space = gym.spaces.Box(0,255,shape=self.sizet)
		self.env.set_window_visible(render)
		self.env.init()

	def reset(self):
		self.time = 0
		self.done = False
		self.env.new_episode()
		state = self.env.get_state().screen_buffer
		if self.transpose: state = np.transpose(state, self.transpose)
		return resize(state.astype(np.uint8))

	def step(self, action):
		self.time += 1
		action_oh = self.one_hot(action)
		reward = self.env.make_action(action_oh)
		self.done = self.env.is_episode_finished() or self.done
		state = np.zeros(self.size) if self.done else self.env.get_state().screen_buffer
		if self.transpose: state = np.transpose(state, self.transpose)
		return resize(state.astype(np.uint8)), reward, self.done, None

	def render(self):
		pass

	def one_hot(self, action):
		action_oh = [0]*self.action_space.n
		action_oh[int(action)] = 1
		return action_oh

	def close(self):
		self.env.close()

	def __del__(self):
		self.close()
