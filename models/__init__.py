from models.singleagent.ppo import PPOAgent
from models.singleagent.ddpg import DDPGAgent, EPS_MIN

all_models = {"ppo":PPOAgent, "ddpg":DDPGAgent}
