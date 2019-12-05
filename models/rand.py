import math
import torch
import random
import numpy as np
from collections import deque
from operator import itemgetter
from utils.network import PTACNetwork, PTActor, PTCritic

DISCOUNT_RATE = 0.97
NUM_STEPS = 20
EPS_MAX = 1.0                 # The starting proportion of random to greedy actions to take
EPS_MIN = 0.1                 # The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.995             # The rate at which eps decays from EPS_MAX to EPS_MIN
ADVANTAGE_DECAY = 0.95
MAX_BUFFER_SIZE = 100000       # Sets the maximum length of the replay buffer

class BrownianNoise:
	def __init__(self, size, dt=0.02):
		self.size = size
		self.dt = dt
		self.reset()

	def reset(self):
		self.action = np.clip(np.random.randn(*self.size), -1, 1)
		self.daction_dt = np.random.randn(*self.size)

	def sample(self, scale=1):
		self.daction_dt = np.random.randn(*self.size)
		self.action = np.clip(self.action + math.sqrt(self.dt) * self.daction_dt, -1, 1)
		return self.action * scale

class RandomAgent():
	def __init__(self, action_size):
		self.noise_process = BrownianNoise(action_size)

	def get_action(self, state, eps=None):
		action = self.noise_process.sample()
		return action

	def get_env_action(self, env, state=None, eps=None):
		action = self.get_action(state, eps)
		action_normal = (1+action)/2
		action_range = env.action_space.high - env.action_space.low
		env_action = env.action_space.low + np.multiply(action_normal, action_range)
		return env_action, action

	def train(self, state, action, next_state, reward, done):
		if done: self.noise_process.reset()

class ACAgent(RandomAgent):
	def __init__(self, state_size, action_size, update_freq=NUM_STEPS, eps=EPS_MAX, decay=EPS_DECAY, gpu=True, load=None):
		super().__init__(action_size)
		self.network = PTACNetwork(state_size, action_size, PTActor, PTCritic, gpu=gpu, load=load)
		self.to_tensor = lambda x: torch.from_numpy(np.array(x)).float().to(self.network.device)
		self.replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
		self.update_freq = update_freq
		self.buffer = []
		self.decay = decay
		self.eps = eps

	def get_action(self, state, eps=None, e_greedy=False):
		action_random = super().get_action(state)
		return action_random

	def compute_gae(self, last_value, rewards, dones, values, gamma=DISCOUNT_RATE, tau=ADVANTAGE_DECAY):
		with torch.no_grad():
			gae = 0
			targets = torch.zeros_like(values, device=values.device)
			values = torch.cat([values, last_value.unsqueeze(0)])
			for step in reversed(range(len(rewards))):
				delta = rewards[step] + gamma * values[step + 1] * (1-dones[step]) - values[step]
				gae = delta + gamma * tau * (1-dones[step]) * gae
				targets[step] = gae + values[step]
			advantages = targets - values[:-1]
			return targets, advantages
		
	def train(self, state, action, next_state, reward, done):
		pass

class ReplayBuffer():
	def __init__(self, maxlen=None):
		self.buffer = deque(maxlen=maxlen)
		
	def add(self, experience):
		self.buffer.append(experience)
		return self

	def extend(self, experiences):
		for exp in experiences:
			self.add(exp)
		return self

	def clear(self):
		self.buffer.clear()
		return self
		
	def sample(self, batch_size, dtype=np.array, weights=None):
		sample_size = min(len(self.buffer), batch_size)
		sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=weights)
		samples = itemgetter(*sample_indices)(self.buffer)
		sample_arrays = samples if dtype is None else map(dtype, zip(*samples))
		return sample_arrays, sample_indices, 1

	def update_priorities(self, indices, errors, offset=0.1):
		pass

	def __len__(self):
		return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, maxlen=None):
		super().__init__(maxlen)
		self.priorities = deque(maxlen=maxlen)
		
	def add(self, experience):
		super().add(experience)
		self.priorities.append(max(self.priorities, default=1))
		return self

	def clear(self):
		super().clear()
		self.priorities.clear()
		return self
		
	def get_probabilities(self, priority_scale):
		scaled_priorities = np.array(self.priorities) ** priority_scale
		sample_probabilities = scaled_priorities / sum(scaled_priorities)
		return sample_probabilities
	
	def get_importance(self, probabilities):
		importance = 1/len(self.buffer) * 1/probabilities
		importance_normalized = importance / max(importance)
		return importance_normalized
		
	def sample(self, batch_size, dtype=np.array, priority_scale=0.6):
		sample_probs = self.get_probabilities(priority_scale)
		samples, sample_indices, _ = super().sample(batch_size, None, sample_probs)
		importance = self.get_importance(sample_probs[sample_indices])
		return map(dtype, zip(*samples)), sample_indices, dtype(importance)
						
	def update_priorities(self, indices, errors, offset=0.1):
		for i,e in zip(indices, errors):
			self.priorities[i] = abs(e) + offset

class NoisyLinear(torch.nn.Linear):
	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		super().__init__(in_features, out_features, bias=bias)
		self.sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
		self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
		if bias:
			self.sigma_bias = torch.nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
			self.register_buffer("epsilon_bias", torch.zeros(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		std = math.sqrt(3 / self.in_features)
		torch.nn.init.uniform_(self.weight, -std, std)
		torch.nn.init.uniform_(self.bias, -std, std)

	def forward(self, input):
		torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
		bias = self.bias
		if bias is not None:
			torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
			bias = bias + self.sigma_bias * torch.autograd.Variable(self.epsilon_bias)
		weight = self.weight + self.sigma_weight * torch.autograd.Variable(self.epsilon_weight)
		return torch.nn.functional.linear(input, weight, bias)