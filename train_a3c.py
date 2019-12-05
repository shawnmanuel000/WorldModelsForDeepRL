import os
import gym
import torch
import argparse
import numpy as np
from collections import deque
from models.ppo import PPOAgent
from models.ddpg import DDPGAgent
from models.rand import RandomAgent
from utils.envs import EnsembleEnv, EnvManager, EnvWorker, WorldModel, ImgStack

parser = argparse.ArgumentParser(description='PPO Trainer')
parser.add_argument('--workerports', type=int, default=[16], nargs="+", help='how many worker servers to connect to')
parser.add_argument('--selfport', type=int, default=None, help='which port to listen on (as a worker server)')
parser.add_argument('--iternum', type=int, default=-1, help='whether to load last saved model')
parser.add_argument('--runs', type=int, default=1, help='how many times to run the simulation')
parser.add_argument('--model', type=str, default="ppo", help='whether to load last saved model')
parser.add_argument('--trial', action="store_true", help='whether to use image observations')
args = parser.parse_args()

ENV_NAME = "CarRacing-v0"

class WorldACAgent(RandomAgent):
	def __init__(self, action_size, num_envs, acagent, statemodel=WorldModel, load="", gpu=True):
		super().__init__(action_size)
		self.world_model = statemodel(action_size, num_envs, load=load, gpu=gpu)
		self.acagent = acagent(self.world_model.state_size, action_size, load="", gpu=gpu)

	def get_env_action(self, env, state, eps=None):
		state, latent = self.world_model.get_state(state)
		env_action, action = self.acagent.get_env_action(env, state, eps)
		self.world_model.step(latent, env_action)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		next_state = self.world_model.get_state(next_state)[0]
		self.acagent.train(state, action, next_state, reward, done)

	def reset(self, num_envs):
		self.world_model.reset(num_envs, restore=num_envs>1)
		return self

	def save_model(self, dirname="pytorch", name="best"):
		self.acagent.network.save_model(dirname, name)

	def load(self, dirname="pytorch", name="best"):
		self.world_model.load_model(dirname, name)
		self.acagent.network.load_model(dirname, name)
		return self

def rollout(env, agent, render=False):
	state = env.reset()
	total_reward = 0
	done = False
	with torch.no_grad():
		while not done:
			if render: env.render()
			env_action = agent.get_env_action(env, state, 0.1)[0]
			state, reward, done, _ = env.step(env_action.reshape(-1))
			total_reward += reward
	return total_reward

def run(model, statemodel, runs=1, load_dir="", ports=16, restarts=0):
	model_name = "ppo" if model == PPOAgent else "ddpg" if model == DDPGAgent else "tmp"
	run_num = len([n for n in os.listdir(f"logs/{model_name}/")])
	num_envs = len(ports) if type(ports) == list else min(ports, 16)
	envs = EnvManager(ENV_NAME, ports) if type(ports) == list else EnsembleEnv(ENV_NAME, ports)
	agent = WorldACAgent(envs.action_size, num_envs, model, statemodel, load=load_dir)
	total_rewards = deque(maxlen=100)
	states = envs.reset()
	for ep in range(runs):
		agent.reset(num_envs)
		total_reward = 0
		for _ in range(envs.env.spec.max_episode_steps):
			env_actions, actions, states = agent.get_env_action(envs.env, states)
			next_states, rewards, dones, _ = envs.step(env_actions, render=(ep%10==0))
			agent.train(states, actions, next_states, rewards, dones)
			total_reward += np.mean(rewards)
			states = next_states
		test_reward = np.mean([rollout(envs.env, agent.reset(1)) for _ in range(5)])
		total_rewards.append(test_reward)
		agent.save_model(load_dir, "checkpoint")
		if total_rewards[-1] >= max(total_rewards): agent.save_model(load_dir)
		if ep == runs//(restarts+1): agent.acagent.network.init_weights(agent.acagent.network.actor_local)
		with open(f"logs/{model_name}/logs_{run_num}.txt", "a+") as f:
			if ep==0: f.write(f"Agent: {model_name}, Model: {statemodel}, State: {agent.world_model.state_size}, Dir: {load_dir}\n")
			f.write(f"Ep: {ep}, Reward: {total_reward:.4f}, Test: {test_reward:.4f}, Avg: {np.mean(total_rewards):.4f} ({agent.acagent.eps:.4f})\n")
		print(f"Ep: {ep}, Reward: {total_reward:.4f}, Test: {test_reward:.4f}, Avg: {np.mean(total_rewards):.4f} ({agent.acagent.eps:.4f})")
	envs.close()

def trial(model, steps=40000, ports=16):
	env_name = "Pendulum-v0"
	envs = EnvManager(ENV_NAME, ports) if type(ports) == list else EnsembleEnv(ENV_NAME, ports)
	agent = model(envs.state_size, envs.action_size, decay=0.99)
	env = gym.make(env_name)
	state = envs.reset()
	test_rewards = []
	for s in range(steps):
		env_action, action = agent.get_env_action(env, state)
		next_state, reward, done, _ = envs.step(env_action)
		agent.train(state, action, next_state, reward, done)
		state = next_state
		if s % env.spec.max_episode_steps == 0:
			test_reward = np.mean([rollout(env, agent) for _ in range(10)])
			test_rewards.append(test_reward)
			print(f"Ep: {s//env.spec.max_episode_steps}, Rewards: {test_reward}, Avg: {np.mean(test_rewards)}")
			if test_reward > -200: break
	env.close()
	envs.close()

if __name__ == "__main__":
	dirname = "pytorch" if args.iternum < 0 else f"iter{args.iternum}/"
	state = ImgStack if args.iternum < 0 else WorldModel
	model = PPOAgent if args.model == "ppo" else DDPGAgent
	if args.trial:
		trial(model, ports=args.workerports)
	elif args.selfport is not None:
		EnvWorker(args.selfport, ENV_NAME).start()
	else:
		if len(args.workerports) == 1: args.workerports = args.workerports[0]
		run(model, state, args.runs, dirname, args.workerports)