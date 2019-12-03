import gym
import torch
import pickle
import argparse
import numpy as np
from models.rand import ReplayBuffer, ACAgent
from utils.network import PTACNetwork, Conv

LEARN_RATE = 0.0001
BATCH_SIZE = 5
PPO_EPOCHS = 4
EPS_MIN = 0.1                 # The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.995             # The rate at which eps decays from EPS_MAX to EPS_MIN
ENTROPY_WEIGHT = 0.005
INPUT_LAYER = 512
ACTOR_HIDDEN = 256
CRITIC_HIDDEN = 1024

class PPOActor(torch.nn.Module):
	def __init__(self, state_size, action_size, sig=0.0):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.layer3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, *action_size)
		self.action_sig = torch.nn.Parameter(torch.zeros(*action_size))
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		
	def forward(self, state, action=None):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		action_mu = self.action_mu(state)
		action_sig = self.action_sig.exp().expand_as(action_mu)
		dist = torch.distributions.Normal(action_mu, action_sig)
		action = dist.sample() if action is None else action
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		return action, log_prob, entropy

class PPOCritic(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, CRITIC_HIDDEN)
		self.layer3 = torch.nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
		self.value = torch.nn.Linear(CRITIC_HIDDEN, 1)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu() + state
		value = self.value(state)
		return value

class PPONetwork(PTACNetwork):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, gpu=True, load=None):
		super().__init__(state_size, action_size, PPOActor, PPOCritic, lr=lr, gpu=gpu, load=load)

	def get_action_probs(self, state, action_in=None, grad=True):
		with torch.enable_grad() if grad else torch.no_grad():
			action, log_prob, entropy = self.actor_local(state.to(self.device), action_in)
			return action if action_in is None else entropy.mean(), log_prob

	def get_value(self, state, grad=True):
		with torch.enable_grad() if grad else torch.no_grad():
			value = self.critic_local(state.to(self.device))
			return value

	def optimize(self, states, actions, old_log_probs, targets, advantages, clip_param=0.1):
		values = self.get_value(states)
		critic_loss = 0.5*(targets - values).pow(2).mean()
		self.step(self.critic_optimizer, critic_loss)

		entropy, new_log_probs = self.get_action_probs(states, actions)
		ratio = (new_log_probs - old_log_probs).exp()
		ratio_clipped = torch.clamp(ratio, 1.0-clip_param, 1.0+clip_param)
		actor_loss = -(torch.min(ratio*advantages, ratio_clipped*advantages).mean() + ENTROPY_WEIGHT*entropy)
		self.step(self.actor_optimizer, actor_loss)

	def save_model(self, dirname="pytorch", name="best"):
		super().save_model("ppo", dirname, name)
		
	def load_model(self, dirname="pytorch", name="best"):
		super().load_model("ppo", dirname, name)

class PPOAgent(ACAgent):
	def __init__(self, state_size, action_size, decay=EPS_DECAY, gpu=True, load=None):
		super().__init__(state_size, action_size, decay=decay, gpu=gpu, load=load)
		self.network = PPONetwork(state_size, action_size, gpu=gpu, load=load)
		self.ppo_epochs = PPO_EPOCHS
		self.ppo_batch = BATCH_SIZE

	def get_action(self, state, eps=None):
		state = self.to_tensor(state)
		self.action = self.network.get_action_probs(state, grad=False)[0].cpu().numpy()
		return np.tanh(self.action)

	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, self.action, reward, done))
		if len(self.buffer) == self.update_freq:
			states, actions, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()
			next_state = self.to_tensor(next_state)
			values = self.network.get_value(states, grad=False)
			next_value = self.network.get_value(next_state, grad=False)
			targets, advantages = self.compute_gae(next_value, rewards.unsqueeze(-1), dones.unsqueeze(-1), values)
			log_probs = self.network.get_action_probs(states, actions, grad=False)[1]
			states, actions, log_probs, targets, advantages = [x.view(x.size(0)*x.size(1), *x.size()[2:]) for x in (states, actions, log_probs, targets, advantages)]
			self.replay_buffer.clear().extend(zip(states, actions, log_probs, targets, advantages))
			for _ in range(self.ppo_epochs*states.size(0)//self.ppo_batch):
				states, actions, log_probs, targets, advantages = self.replay_buffer.sample(self.ppo_batch, dtype=torch.stack)[0]
				self.network.optimize(states, actions, log_probs, targets, advantages, clip_param=0.1*self.eps)
			self.eps = max(self.eps * self.decay, EPS_MIN)