import os
import math
import torch
import random
import numpy as np
from models.rand import RandomAgent, PrioritizedReplayBuffer, ReplayBuffer, ACAgent
from utils.network import PTACNetwork, Conv, TARGET_UPDATE_RATE

LEARN_RATE = 0.0001           # Sets how much we want to update the network weights at each training step
REPLAY_BATCH_SIZE = 32        # How many experience tuples to sample from the buffer for each train step
EPS_MIN = 0.1                 # The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.990             # The rate at which eps decays from EPS_MAX to EPS_MIN
INPUT_LAYER = 512
ACTOR_HIDDEN = 256
CRITIC_HIDDEN = 1024

class DDPGActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.layer3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, *action_size)
		self.action_sig = torch.nn.Linear(ACTOR_HIDDEN, *action_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		state = self.layer1(state).relu() 
		state = self.layer2(state).relu() 
		state = self.layer3(state).relu() 
		action_mu = self.action_mu(state)
		action_sig = self.action_sig(state).exp()
		epsilon = torch.randn_like(action_sig)
		action = action_mu + epsilon.mul(action_sig)
		return action.tanh()
	
class DDPGCritic(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.net_state = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.net_action = torch.nn.Linear(*action_size, INPUT_LAYER)
		self.net_layer1 = torch.nn.Linear(2*INPUT_LAYER, CRITIC_HIDDEN)
		self.net_layer2 = torch.nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
		self.q_value = torch.nn.Linear(CRITIC_HIDDEN, 1)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, action):
		state = self.net_state(state).relu()
		net_action = self.net_action(action).relu()
		net_layer = torch.cat([state, net_action], dim=-1)
		net_layer = self.net_layer1(net_layer).relu()
		net_layer = self.net_layer2(net_layer).relu()
		q_value = self.q_value(net_layer)
		return q_value

class DDPGNetwork(PTACNetwork):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, gpu=True, load=None): 
		super().__init__(state_size, action_size, DDPGActor, DDPGCritic, lr=lr, gpu=gpu, load=load)

	def get_action(self, state, use_target=False, numpy=True):
		with torch.no_grad():
			action = self.actor_local(state) if not use_target else self.actor_target(state)
			return action.cpu().numpy() if numpy else action

	def get_q_value(self, state, action, use_target=False, numpy=True):
		with torch.no_grad():
			q_value = self.critic_local(state, action) if not use_target else self.critic_target(state, action)
			return q_value.cpu().numpy() if numpy else q_value
	
	def optimize(self, states, actions, q_targets, importances=1):
		q_values = self.critic_local(states, actions)
		critic_error = q_values - q_targets.detach()
		critic_loss = importances * critic_error.pow(2)
		self.step(self.critic_optimizer, critic_loss.mean())

		q_actions = self.critic_local(states, self.actor_local(states))
		actor_loss = -(q_actions - q_values.detach())
		self.step(self.actor_optimizer, actor_loss.mean())
		
		self.soft_copy(self.actor_local, self.actor_target)
		self.soft_copy(self.critic_local, self.critic_target)
		return critic_error.cpu().detach().numpy().squeeze(-1)
	
	def save_model(self, dirname="pytorch", name="best"):
		super().save_model("ddpg", dirname, name)
		
	def load_model(self, dirname="pytorch", name="best"):
		super().load_model("ddpg", dirname, name)

class DDPGAgent(ACAgent):
	def __init__(self, state_size, action_size, decay=EPS_DECAY, lr=LEARN_RATE, gpu=True, load=None):
		super().__init__(state_size, action_size, decay=decay, gpu=gpu, load=load)
		self.network = DDPGNetwork(state_size, action_size, lr=lr, gpu=gpu, load=load)

	def get_action(self, state, eps=None, e_greedy=False):
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state, eps)
		if e_greedy and random.random() < eps: return action_random
		action_greedy = self.network.get_action(self.to_tensor(state))
		action = action_greedy if e_greedy else np.clip((1-eps)*action_greedy + eps*action_random, -1, 1)
		return action
		
	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, reward, done))
		if len(self.buffer) == self.update_freq:
			states, actions, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			next_state = self.to_tensor(next_state)
			next_action = self.network.get_action(next_state, use_target=True, numpy=False)
			values = self.network.get_q_value(states, actions, use_target=True, numpy=False)
			next_value = self.network.get_q_value(next_state, next_action, use_target=True, numpy=False)
			targets, advantages = self.compute_gae(next_value, rewards.unsqueeze(-1), dones.unsqueeze(-1), values)
			states, actions, targets, advantages = [x.view(x.size(0)*x.size(1), *x.size()[2:]).cpu().numpy() for x in (states, actions, targets, advantages)]
			self.replay_buffer.extend(zip(states, actions, targets, advantages))	
		if len(self.replay_buffer) > 0:
			(states, actions, targets, advantages), indices, importance = self.replay_buffer.sample(REPLAY_BATCH_SIZE, dtype=self.to_tensor)
			errors = self.network.optimize(states, actions, targets, importance**(1-self.eps))
			self.replay_buffer.update_priorities(indices, errors)
			if done[0]: self.eps = max(self.eps * self.decay, EPS_MIN)
