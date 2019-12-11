import cma
import gym
import torch
import asyncio
import argparse
import numpy as np
import socket as Socket
from torchvision import transforms
from models.vae import VAE, LATENT_SIZE
from models.mdrnn import MDRNNCell, HIDDEN_SIZE, ACTION_SIZE
from models.controller import Controller, ControlAgent
from utils.multiprocess import Manager, Worker
from utils.misc import to_env, IMG_DIM, Logger

parser = argparse.ArgumentParser(description='Controller Trainer')
parser.add_argument('--workerports', type=int, default=None, nargs="+", help='how many worker servers to connect to')
parser.add_argument('--selfport', type=int, default=None, help='which port to listen on (as a worker server)')
parser.add_argument('--iternum', type=int, default=0, help='which port to listen on (as a worker server)')
parser.add_argument('--test', action="store_true", help='whether to just show a test rollout')
args = parser.parse_args()

ENV_NAME = "CarRacing-v0"

class ControllerWorker(Worker):
	def start(self, load_dirname, gpu=True, iterations=1):
		env = gym.make(ENV_NAME)
		env.env.verbose = 0
		agent = ControlAgent(env.action_space.shape, gpu=gpu, load=load_dirname)
		episode = 0
		while True:
			episode += 1
			data = self.conn.recv(20000)
			params = np.frombuffer(data, dtype=np.float64)
			if params.shape[0] != 867: break
			score = np.mean([rollout(env, agent.set_params(params), render=False) for _ in range(iterations)])
			self.conn.sendall(f"{score}".encode())
			print(f"Ep: {episode}, Score: {score:.4f}")
		env.close()

class ControllerManager(Manager):
	def start(self, save_dirname, popsize, epochs=1000):
		def get_scores(params):
			scores = []
			for i in range(0, len(params), self.num_clients):
				self.send_params(params[i:i+self.num_clients])
				scores.extend(self.await_results(converter=float))
			return scores
		popsize = (max(popsize//self.num_clients, 1))*self.num_clients
		train(save_dirname, get_scores, epochs, popsize=popsize)

def rollout(env, agent, render=False):
	state = env.reset()
	total_reward = 0
	done = False
	with torch.no_grad():
		while not done:
			if render: env.render()
			env_action = agent.get_env_action(env, state)[0]
			state, reward, done, _ = env.step(env_action.reshape(-1))
			total_reward += reward
	return total_reward

def train(save_dirname, get_scores, epochs=250, popsize=4, restarts=1):
	logger = Logger(Controller, save_dirname, popsize=popsize, restarts=restarts)
	controller = Controller(gpu=False, load=False)
	best_solution = (controller.get_params(), -np.inf)
	for run in range(restarts):
		start_epochs = epochs//restarts
		es = cma.CMAEvolutionStrategy(best_solution[0], 0.1, {"popsize": popsize})
		while not es.stop() and start_epochs > 0:
			start_epochs -= 1
			params = es.ask()
			scores = get_scores(params)
			es.tell(params, [-s for s in scores])
			best_index = np.argmax(scores)
			best_params = (params[best_index], scores[best_index])
			if best_params[1] > best_solution[1]:
				controller.set_params(best_params[0]).save_model(save_dirname)
				best_solution = best_params
			logger.log(f"Ep: {run}-{start_epochs}, Best score: {best_params[1]:3.4f}, Min: {np.min(scores):.4f}, Avg: {np.mean(scores):.4f}")

def run(load_dirname, test=False, gpu=True, iterations=1):
	env = gym.make(ENV_NAME)
	env.env.verbose = 0
	agent = ControlAgent(env.action_space.shape, gpu=gpu, load=load_dirname)
	get_scores = lambda params: [np.mean([rollout(env, agent.set_params(p)) for _ in range(iterations)]) for p in params]
	train(load_dirname, get_scores, 1000) if not test else rollout(env, agent, True)
	env.close()

if __name__ == "__main__":
	dirname = "pytorch" if args.iternum < 0 else f"iter{args.iternum}/"
	if args.selfport is not None:
		ControllerWorker(args.selfport).start(dirname, gpu=True, iterations=5)
	elif args.workerports is not None:
		ControllerManager(args.workerports).start(dirname, epochs=250, popsize=16)
	else:
		run(dirname, args.test, gpu=False)