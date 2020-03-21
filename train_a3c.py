import os
import gym
import torch
import argparse
import numpy as np
from envs import VizDoomEnv
from utils.rand import RandomAgent
from utils.misc import Logger, rollout
from utils.envs import EnsembleEnv, EnvManager, EnvWorker, GymEnv
from utils.wrappers import WorldACAgent
from utils.multiprocess import set_rank_size
from models.singleagent.ppo import PPOAgent
from models.singleagent.ddpg import DDPGAgent, EPS_MIN

TRIAL_AT = 1000
SAVE_AT = 1

# env_name = "CartPole-v0"
# env_name = "Pendulum-v0"
# env_name = "basic"
# env_name = "my_way_home"
# env_name = "health_gathering"
# env_name = "predict_position"
# env_name = "defend_the_center"
# env_name = "defend_the_line"
# env_name = "take_cover"
env_names = ["defend_the_line", "take_cover", "CarRacing-v0"]
env_name = env_names[0]
models = {"ppo":PPOAgent, "ddpg":DDPGAgent}

def make_env():
	if "-v" in env_name:
		env = GymEnv(gym.make(env_name))
		env.unwrapped.verbose = 0
	else:
		env = VizDoomEnv(env_name)
	return env

def train(make_env, model, ports, steps, checkpoint=None, save_best=False, log=True, render=False):
	envs = (EnvManager if len(ports)>0 else EnsembleEnv)(make_env, ports)
	agent = WorldACAgent(envs.state_size, envs.action_size, model, envs.num_envs, load=checkpoint, gpu=True, worldmodel=True) 
	logger = Logger(model, checkpoint, num_envs=envs.num_envs, state_size=agent.state_size, action_size=envs.action_size, action_space=envs.env.action_space, envs=type(envs), statemodel=agent.state_model)
	states = envs.reset(train=True)
	total_rewards = []
	for s in range(steps+1):
		env_actions, actions, states = agent.get_env_action(envs.env, states)
		next_states, rewards, dones, _ = envs.step(env_actions, train=True)
		agent.train(states, actions, next_states, rewards, dones)
		states = next_states
		if s%TRIAL_AT==0:
			rollouts = rollout(envs, agent, render=render)
			total_rewards.append(np.round(np.mean(rollouts, axis=-1), 3))
			if checkpoint and len(total_rewards)%SAVE_AT==0: agent.save_model(checkpoint)
			if checkpoint and save_best and np.all(total_rewards[-1] >= np.max(total_rewards, axis=-1)): agent.save_model(checkpoint, "best")
			if log: logger.log(f"Step: {s:7d}, Reward: {total_rewards[-1]} [{np.std(rollouts):4.3f}], Avg: {np.mean(total_rewards, axis=0)} ({agent.acagent.eps:.4f})")
	envs.close()

def trial(make_env, model, checkpoint=None, log=False):
	envs = EnsembleEnv(make_env, 1)
	agent = WorldACAgent(envs.state_size, envs.action_size, model, envs.num_envs, load="", train=False, gpu=False, worldmodel=True).load(checkpoint)
	print(f"Reward: {rollout(envs, agent, eps=EPS_MIN, render=True)}")
	envs.close()

def parse_args(envs, models):
	parser = argparse.ArgumentParser(description="A3C Trainer")
	parser.add_argument("--model", type=str, default="ppo", choices=models, help="Which RL algorithm to use. Allowed values are:\n"+', '.join(models), metavar="model")
	parser.add_argument("--iternum", type=int, default=-1, choices=[-1,0,1], help="Whether to train using World Model to load (0 or 1) or raw images (-1)")
	parser.add_argument("--env_name", type=str, default=env_name, choices=envs, help="Name of the environment to use. Allowed values are:\n"+', '.join(envs), metavar="env_name")
	parser.add_argument("--tcp_ports", type=int, default=[], nargs="+", help="The list of worker ports to connect to")
	parser.add_argument("--tcp_rank", type=int, default=0, help="Which port to listen on (as a worker server)")
	parser.add_argument("--render", action="store_true", help="Whether to render an environment rollout")
	parser.add_argument("--trial", action="store_true", help="Whether to show a trial run training on the Pendulum-v0 environment")
	parser.add_argument("--steps", type=int, default=100000, help="Number of steps to train the agent")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args(env_names, models.keys())
	checkpoint = f"{env_name}/pytorch" if args.iternum < 0 else f"{env_name}/iter{args.iternum}/"
	rank, size = set_rank_size(args.tcp_rank, args.tcp_ports)
	model = models[args.model]
	if rank>0:
		EnvWorker(make_env=make_env).start()
	elif args.trial:
		trial(make_env=make_env, model=model, checkpoint=checkpoint)
	else:
		train(make_env=make_env, model=model, ports=list(range(1,size)), steps=args.steps, checkpoint=checkpoint, render=args.render)