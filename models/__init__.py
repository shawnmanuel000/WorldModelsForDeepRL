from models.singleagent.ppo import PPOAgent
from models.singleagent.ddpg import DDPGAgent, EPS_MIN
from models.singleagent.sac import SACAgent

all_models = {"ppo":PPOAgent, "ddpg":DDPGAgent, "sac":SACAgent}
