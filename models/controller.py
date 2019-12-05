import os
import torch
import numpy as np
from utils.envs import WorldModel
from models.rand import RandomAgent
from models.vae import LATENT_SIZE
from models.mdrnn import HIDDEN_SIZE, ACTION_SIZE

class ControlActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.linear = torch.nn.Linear(state_size[-1], *action_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) == torch.nn.Linear else None)

	def forward(self, state):
		action = self.linear(state)
		return action

class Controller():
	def __init__(self, state_size=[LATENT_SIZE+HIDDEN_SIZE], action_size=[ACTION_SIZE], gpu=True, load=""):
		super().__init__()
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
		self.actor_local = ControlActor(state_size, action_size).to(self.device)
		if load: self.load_model(load)

	def get_action(self, state):
		with torch.no_grad():
			action = self.actor_local(state.to(self.device)).clamp(-1, 1)
			return action.cpu().numpy()

	def get_params(self):
		params = [p.view(-1) for p in self.actor_local.parameters()]
		params = torch.cat(params, dim=0)
		return params.cpu().detach().numpy()

	def set_params(self, params):
		numels = [p.numel() for p in self.actor_local.parameters()]
		starts = np.cumsum([0] + numels)
		params = [params[s:starts[i+1]] for i,s in enumerate(starts[:-1])]
		for p,d in zip(self.actor_local.parameters(), params):
			p.data.copy_(torch.Tensor(d).view(p.size()))
		return self

	def save_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.actor_local.state_dict(), filepath)
		
	def load_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		if os.path.exists(filepath):
			self.actor_local.load_state_dict(torch.load(filepath))
		return self

class ControlAgent(RandomAgent):
	def __init__(self, action_size, gpu=True, load=""):
		super().__init__(action_size)
		self.world_model = WorldModel(action_size, num_envs=1, load=load, gpu=gpu)
		self.controller = Controller(self.world_model.state_size, action_size, gpu=gpu, load=load)

	def get_action(self, state, eps=None):
		state, self.latent = self.world_model.get_state(state, numpy=False)
		action = self.controller.get_action(state)
		return action

	def get_env_action(self, env, state, eps=None):
		env_action, action = super().get_env_action(env, state, eps)
		self.world_model.step(self.latent, env_action)
		return env_action, action

	def set_params(self, params):
		self.controller.set_params(params)
		return self

	def save_model(self, dirname="pytorch", name="best"):
		self.controller.save_model(dirname, name)

	def load_model(self, dirname="pytorch", name="best"):
		self.world_model.load_model(dirname, name)
		self.controller.load_model(dirname, name)
		return self

def get_checkpoint_path(dirname="pytorch", name="best"):
	return f"./saved_models/ctrl/{dirname}/{name}.pth"