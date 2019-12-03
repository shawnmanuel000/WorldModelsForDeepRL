import os
import gym
import cv2
import torch
import random
import argparse
import numpy as np
import utils.misc as misc
from models.controller import ControlAgent
from models.rand import RandomAgent
from data.loaders import ROOT

parser = argparse.ArgumentParser(description='PPO Trainer')
parser.add_argument('--nsamples', type=int, default=0, help='how many worker servers to connect to')
parser.add_argument('--iternum', type=int, default=0, help='which port to listen on (as a worker server)')
parser.add_argument('--datadir', type=str, default=ROOT, help='which port to listen on (as a worker server)')
args = parser.parse_args()

class RolloutCollector():
	def __init__(self, save_path):
		self.reset_rollout()
		self.save_path = save_path

	def reset_rollout(self):
		self.a_rollout = []
		self.s_rollout = []
		self.r_rollout = []
		self.d_rollout = []

	def step(self, env_action, next_state, reward, done, number=None):
		self.a_rollout.append(env_action)
		self.s_rollout.append(misc.resize(next_state))
		self.r_rollout.append(reward)
		self.d_rollout.append(done)
		if done: self.save_rollout(number=number)

	def save_rollout(self, number=None):
		if len(self.a_rollout) + len(self.s_rollout) + len(self.r_rollout) + len(self.d_rollout) == 0: return
		if number is None: number = len([n for n in os.listdir(self.save_path)])
		a = np.array(self.a_rollout)
		s = np.array(self.s_rollout)
		r = np.array(self.r_rollout)
		d = np.array(self.d_rollout)
		np.savez(os.path.join(self.save_path, f"rollout_{number}"), actions=a, states=s, rewards=r, dones=d)
		self.reset_rollout()

def sample(runs, iternum, root=ROOT, number=None):
	dirname = f"iter{iternum}/"
	env = gym.make("CarRacing-v0")
	env.env.verbose = 0
	os.makedirs(os.path.dirname(os.path.join(root, dirname)), exist_ok=True)
	rollout = RolloutCollector(os.path.join(root, dirname))
	agent = RandomAgent(env.action_space.shape) if iternum <= 0 else ControlAgent(env.action_space.shape, gpu=False, load=f"iter{iternum-1}/")
	for ep in range(runs):
		state = env.reset()
		total_reward = 0
		done = False
		while not done:
			env_action = agent.get_env_action(env, state)[0]
			state, reward, done, _ = env.step(env_action)
			rollout.step(env_action, state, reward, done, number)
			agent.train(state, env_action, state, reward, done)
			total_reward += reward
		print(f"Ep: {ep}, Reward: {total_reward}")
	env.close()

def check_samples(iternum, root=ROOT):
	dirname = f"iter{iternum}/"
	if os.path.exists(os.path.join(root, dirname)):
		files = [os.path.join(root, dirname, n) for n in sorted(os.listdir(os.path.join(root, dirname)), key=lambda x: str(len(x))+x)]
		for i,f in enumerate(files):
			with np.load(f) as data:
				size = data["states"].shape[0]
			if size != 1000:
				print(f"{f}: {size}")
				sample(1, iternum, number=i)
				check_samples(iternum)

if __name__ == "__main__":
	if args.nsamples > 0:
		sample(args.nsamples, args.iternum, args.datadir)
	check_samples(args.iternum, args.datadir)