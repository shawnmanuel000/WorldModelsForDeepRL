import torch
import numpy as np
from utils.rand import ReplayBuffer
from utils.network import PTACNetwork, PTACAgent, PTCritic, LEARN_RATE, REPLAY_BATCH_SIZE, TARGET_UPDATE_RATE, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, Conv, EPS_DECAY, DISCOUNT_RATE, gsoftmax

class SACActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.discrete = type(action_size) != tuple
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.layer3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.action_sig = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		
	def forward(self, state, action=None, sample=True):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		action_mu = self.action_mu(state)
		action_sig = self.action_sig(state).clamp(-5,0).exp()
		dist = torch.distributions.Normal(action_mu, action_sig)
		action = dist.rsample() if sample else action_mu
		action = gsoftmax(action, hard=False) if self.discrete else action
		log_prob = torch.log(action+1e-6) if self.discrete else dist.log_prob(action)
		log_prob -= torch.log(1-action.tanh().pow(2)+1e-6)
		return action.tanh(), log_prob

class SACCritic(torch.nn.Module):
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

class SACNetwork(PTACNetwork):
	def __init__(self, state_size, action_size, actor=SACActor, critic=SACCritic, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load=None, name="sac"):
		self.discrete = type(action_size)!=tuple
		super().__init__(state_size, action_size, actor, critic if not self.discrete else lambda s,a: PTCritic(s,a), lr=lr, tau=tau, gpu=gpu, load=load, name=name)
		self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True).to(self.device))
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
		self.target_entropy = -np.product(action_size)

	def get_action_probs(self, state, action_in=None, grad=False, numpy=False, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			action, log_prob = self.actor_local(state.to(self.device), action_in, sample)
			return [x.cpu().numpy() if numpy else x for x in [action, log_prob]]

	def get_q_value(self, state, action, use_target=False, grad=False, numpy=False):
		with torch.enable_grad() if grad else torch.no_grad():
			critic = self.critic_local if not use_target else self.critic_target
			q_value = critic(state) if self.discrete else critic(state, action)
			return q_value.cpu().numpy() if numpy else q_value
	
	def optimize(self, states, actions, next_states, rewards, dones, gamma=DISCOUNT_RATE):
		alpha = self.log_alpha.clamp(-5, 0).detach().exp()
		next_actions, next_log_prob = self.actor_local(next_states)
		q_nexts = self.get_q_value(next_states, next_actions, use_target=True) - alpha*next_log_prob
		q_nexts = (next_actions*q_nexts).mean(-1, keepdim=True) if self.discrete else q_nexts
		q_targets = rewards.unsqueeze(-1) + gamma * q_nexts * (1 - dones.unsqueeze(-1))

		q_values = self.get_q_value(states, actions, grad=True)
		q_values = q_values.gather(-1, actions.argmax(-1, keepdim=True)) if self.discrete else q_values
		critic1_loss = (q_values - q_targets.detach()).pow(2)
		self.step(self.critic_optimizer, critic1_loss.mean(), self.critic_local.parameters())
		self.soft_copy(self.critic_local, self.critic_target)

		actor_action, log_prob = self.actor_local(states)
		q_actions = self.get_q_value(states, actor_action, grad=True)
		actor_loss = alpha*log_prob - (q_actions - q_values.detach())
		actor_loss = actor_action*actor_loss if self.discrete else actor_loss
		self.step(self.actor_optimizer, actor_loss.mean(), self.actor_local.parameters())
		
		alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy))
		self.step(self.alpha_optimizer, alpha_loss.mean())

class SACAgent(PTACAgent):
	def __init__(self, state_size, action_size, decay=EPS_DECAY, lr=LEARN_RATE, gpu=True, load=None):
		super().__init__(state_size, action_size, SACNetwork, decay=decay, lr=lr, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True, e_greedy=False):
		action = self.network.get_action_probs(self.to_tensor(state), numpy=True, sample=sample)[0]
		return action
		
	def train(self, state, action, next_state, reward, done):
		self.replay_buffer.extend(list(zip(state, action, next_state, reward, done)), shuffle=False)	
		if len(self.replay_buffer) > REPLAY_BATCH_SIZE:
			states, actions, next_states, rewards, dones = self.replay_buffer.sample(REPLAY_BATCH_SIZE, dtype=self.to_tensor)[0]
			self.network.optimize(states, actions, next_states, rewards, dones)
