import os
import math
import torch
import random
import numpy as np

REG_LAMBDA = 1e-6             # Penalty multiplier to apply for the size of the network weights
LEARN_RATE = 0.0001           # Sets how much we want to update the network weights at each training step
TARGET_UPDATE_RATE = 0.0004   # How frequently we want to copy the local network to the target network (for double DQNs)
INPUT_LAYER = 256
ACTOR_HIDDEN = 256
CRITIC_HIDDEN = 256

class PTActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.layer3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, *action_size)

	def forward(self, state):
		state = self.layer1(state).relu() 
		state = self.layer2(state).relu() 
		state = self.layer3(state).relu() 
		action_mu = self.action_mu(state)
		return action_mu

class PTCritic(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.net_state = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, CRITIC_HIDDEN)
		self.layer3 = torch.nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
		self.value = torch.nn.Linear(CRITIC_HIDDEN, 1)

	def forward(self, state, action):
		state = self.net_state(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu() + state
		value = self.value(state)
		return value

class PTACNetwork():
	def __init__(self, state_size, action_size, actor, critic, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load=""): 
		self.tau = tau
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
		self.actor_local = actor(state_size, action_size).to(self.device)
		self.actor_target = actor(state_size, action_size).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, factor=0.5, patience=25, min_lr=1e-5)
		
		self.critic_local = critic(state_size, action_size).to(self.device)
		self.critic_target = critic(state_size, action_size).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, factor=0.5, patience=25, min_lr=1e-5)
		if load: self.load_model(load)

	def init_weights(self, model=None):
		model = self if model is None else model
		model.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		
	def step(self, optimizer, loss, retain=False):
		optimizer.zero_grad()
		loss.backward(retain_graph=retain)
		optimizer.step()

	def schedule(self, total_reward):
		self.actor_scheduler.step(-total_reward)
		self.critic_scheduler.step(-total_reward)
		
	def soft_copy(self, local, target):
		for t,l in zip(target.parameters(), local.parameters()):
			t.data.copy_(t.data + self.tau*(l.data - t.data))

	def save_model(self, net="qlearning", dirname="pytorch", name="checkpoint"):
		filepath = get_checkpoint_path(net, dirname, name)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.actor_local.state_dict(), filepath.replace(".pth", "_a.pth"))
		torch.save(self.critic_local.state_dict(), filepath.replace(".pth", "_c.pth"))
		
	def load_model(self, net="qlearning", dirname="pytorch", name="checkpoint"):
		filepath = get_checkpoint_path(net, dirname, name)
		if os.path.exists(filepath.replace(".pth", "_a.pth")):
			self.actor_local.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth")))
			self.actor_target.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth")))
			self.critic_local.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth")))
			self.critic_target.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth")))

class Conv(torch.nn.Module):
	def __init__(self, state_size, output_size):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(state_size[-1], 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)
		self.linear1 = torch.nn.Linear(1024, output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		out_dims = state.size()[:-3]
		state = state.view(-1, *state.size()[-3:])
		state = self.conv1(state).tanh() # state: (batch, 32, 31, 31)
		state = self.conv2(state).tanh() # state: (batch, 64, 14, 14)
		state = self.conv3(state).tanh() # state: (batch, 128, 6, 6)
		state = self.conv4(state).tanh() # state: (batch, 256, 2, 2)
		state = state.view(state.size(0), -1) # state: (batch, 1024)
		state = self.linear1(state).tanh() # state: (batch, 512)
		state = state.view(*out_dims, -1)
		return state

def init_weights(m):
	if type(m) == torch.nn.Linear:
		torch.nn.init.normal_(m.weight, mean=0., std=0.1)
		torch.nn.init.constant_(m.bias, 0.1)

def get_checkpoint_path(net="qlearning", dirname="pytorch", name="checkpoint"):
	return f"./saved_models/{net}/{dirname}/{name}.pth"