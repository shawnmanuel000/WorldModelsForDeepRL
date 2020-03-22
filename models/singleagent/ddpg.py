import os
import math
import torch
import random
import numpy as np
from utils.rand import RandomAgent, PrioritizedReplayBuffer, ReplayBuffer
from utils.network import PTACNetwork, PTACAgent, Conv, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, LEARN_RATE, TARGET_UPDATE_RATE, NUM_STEPS, gsoftmax, one_hot

EPS_MIN = 0.020              	# The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.98             	# The rate at which eps decays from EPS_MAX to EPS_MIN
REPLAY_BATCH_SIZE = 128        	# How many experience tuples to sample from the buffer for each train step

class DDPGActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.discrete = type(action_size) != tuple
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.layer3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.action_sig = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, sample=True):
		state = self.layer1(state).relu() 
		state = self.layer2(state).relu() 
		state = self.layer3(state).relu() 
		action_mu = self.action_mu(state)
		action_sig = self.action_sig(state).exp()
		epsilon = torch.randn_like(action_sig)
		action = action_mu + epsilon.mul(action_sig) if sample else action_mu
		return action.tanh() if not self.discrete else gsoftmax(action)
	
class DDPGCritic(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.net_state = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.net_action = torch.nn.Linear(action_size[-1], INPUT_LAYER)
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
	def __init__(self, state_size, action_size, actor=DDPGActor, critic=DDPGCritic, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load=None): 
		super().__init__(state_size, action_size, actor=actor, critic=critic, lr=lr, tau=tau, gpu=gpu, load=load, name="ddpg")

	def get_action(self, state, use_target=False, grad=False, numpy=True, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			actor = self.actor_local if not use_target else self.actor_target
			return actor(state, sample).cpu().numpy() if numpy else actor(state, sample)

	def get_q_value(self, state, action, use_target=False, grad=False, numpy=True):
		with torch.enable_grad() if grad else torch.no_grad():
			critic = self.critic_local if not use_target else self.critic_target
			return critic(state, action).cpu().numpy() if numpy else critic(state, action)
	
	def optimize(self, states, actions, q_targets, importances=1):
		if self.actor_local.discrete: actions = one_hot(actions)
		q_values = self.critic_local(states, actions)
		critic_error = q_values - q_targets.detach()
		critic_loss = importances.to(self.device) * critic_error.pow(2)
		self.step(self.critic_optimizer, critic_loss.mean())

		actor_action = self.actor_local(states)
		q_actions = self.critic_local(states, actor_action)
		actor_loss = -(q_actions - q_values.detach())
		self.step(self.actor_optimizer, actor_loss.mean())
		
		self.soft_copy(self.actor_local, self.actor_target)
		self.soft_copy(self.critic_local, self.critic_target)
		return critic_error.cpu().detach().numpy().squeeze(-1)
		
class DDPGAgent(PTACAgent):
	def __init__(self, state_size, action_size, decay=EPS_DECAY, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load=None):
		super().__init__(state_size, action_size, DDPGNetwork, decay=decay, lr=lr, tau=tau, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True):
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state, eps)
		if self.discrete and random.random() < eps: return action_random
		action_greedy = self.network.get_action(self.to_tensor(state), sample=sample)
		action = np.clip((1-eps)*action_greedy + eps*action_random, -1, 1)
		return action
		
	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, reward, done))
		if np.any(done[0]) or len(self.buffer) >= self.update_freq:
			states, actions, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			states = torch.cat([states, self.to_tensor(next_state).unsqueeze(0)], dim=0)
			actions = torch.cat([actions, self.network.get_action(states[-1], use_target=True, numpy=False).unsqueeze(0)], dim=0)
			values = self.network.get_q_value(states, actions, use_target=True, numpy=False)
			targets = self.compute_gae(values[-1], rewards.unsqueeze(-1), dones.unsqueeze(-1), values[:-1])[0]
			states, actions, targets = [x.view(x.size(0)*x.size(1), *x.size()[2:]).cpu().numpy() for x in (states[:-1], actions[:-1], targets)]
			self.replay_buffer.extend(list(zip(states, actions, targets)), shuffle=False)	
		if len(self.replay_buffer) > REPLAY_BATCH_SIZE:
			(states, actions, targets), indices, importances = self.replay_buffer.sample(REPLAY_BATCH_SIZE, dtype=self.to_tensor)
			errors = self.network.optimize(states, actions, targets, importances**(1-self.eps))
			self.replay_buffer.update_priorities(indices, errors)
			if np.any(done[0]): self.eps = max(self.eps * self.decay, EPS_MIN)
