import cma
import gym
import pickle
import argparse
import numpy as np
import socket as Socket
from envs import make_env, env_name
from torchvision import transforms
from models.worldmodel.vae import VAE, LATENT_SIZE
from models.worldmodel.mdrnn import MDRNNCell, HIDDEN_SIZE, ACTION_SIZE
from models.worldmodel.controller import Controller, ControlAgent
from utils.multiprocess import Manager, Worker
from utils.misc import rollout, IMG_DIM, Logger

parser = argparse.ArgumentParser(description="Controller Trainer")
parser.add_argument("--workerports", type=int, default=None, nargs="+", help="The list of worker ports to connect to")
parser.add_argument("--selfport", type=int, default=None, help="Which port to listen on (as a worker server)")
parser.add_argument("--iternum", type=int, default=0, help="Which iteration of trained World Model to load")
args = parser.parse_args()

class ControllerWorker(Worker):
	def start(self, load_dirname, gpu=True, iterations=1):
		env = make_env()
		action_size = [env.action_space.n] if hasattr(env.action_space, 'n') else env.action_space.shape
		agent = ControlAgent(action_size=action_size, gpu=gpu, load=load_dirname)
		episode = 0
		while True:
			data = pickle.loads(self.conn.recv(100000))
			if not "cmd" in data or data["cmd"] != "ROLLOUT": break
			score = np.mean([rollout(env, agent.set_params(data["item"]), render=False) for _ in range(iterations)])
			self.conn.sendall(pickle.dumps(score))
			print(f"Ep: {episode}, Score: {score:.4f}")
			episode += 1
		env.close()

class ControllerManager(Manager):
	def start(self, save_dirname, popsize, epochs=1000):
		def get_scores(params):
			scores = []
			for i in range(0, len(params), self.num_clients):
				self.send_params([pickle.dumps({"cmd": "ROLLOUT", "item": params[i+j]}) for j in range(self.num_clients)])
				scores.extend(self.await_results(converter=pickle.loads))
			return scores
		popsize = (max(popsize//self.num_clients, 1))*self.num_clients
		train(save_dirname, get_scores, epochs, popsize=popsize)

def train(save_dirname, get_scores, epochs=250, popsize=4, restarts=1):
	env = make_env()
	action_size = [env.action_space.n] if hasattr(env.action_space, 'n') else env.action_space.shape
	logger = Logger(Controller, save_dirname, popsize=popsize, restarts=restarts)
	controller = Controller(action_size=action_size, gpu=False, load=False)
	best_solution = (controller.get_params(), -np.inf)
	total_scores = []
	for run in range(restarts):
		start_epochs = epochs//restarts
		es = cma.CMAEvolutionStrategy(best_solution[0], 0.1, {"popsize": popsize})
		while not es.stop() and start_epochs > 0:
			start_epochs -= 1
			params = es.ask()
			scores = get_scores(params)
			total_scores.append(np.mean(scores))
			es.tell(params, [-s for s in scores])
			best_index = np.argmax(scores)
			best_params = (params[best_index], scores[best_index])
			if best_params[1] > best_solution[1]:
				controller.set_params(best_params[0]).save_model(save_dirname)
				best_solution = best_params
			logger.log(f"Ep: {run}-{start_epochs}, Best score: {best_params[1]:3.4f}, Min: {np.min(scores):.4f}, Avg: {total_scores[-1]:.4f}, Rolling: {np.mean(total_scores):.4f}")

def run(load_dirname, gpu=True, iterations=1):
	env = make_env()
	action_size = [env.action_space.n] if hasattr(env.action_space, 'n') else env.action_space.shape
	agent = ControlAgent(action_size, gpu=gpu, load=load_dirname)
	get_scores = lambda params: [np.mean([rollout(env, agent.set_params(p)) for _ in range(iterations)]) for p in params]
	train(load_dirname, get_scores, 1000)
	env.close()

if __name__ == "__main__":
	dirname = f"{env_name}/pytorch" if args.iternum < 0 else f"{env_name}/iter{args.iternum}/"
	if args.selfport is not None:
		ControllerWorker(args.selfport).start(dirname, gpu=True, iterations=1)
	elif args.workerports is not None:
		ControllerManager(args.workerports).start(dirname, epochs=500, popsize=64)
	else:
		run(dirname, gpu=False)