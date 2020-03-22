import gym
from envs.wrappers import GymEnv, VizDoomEnv

gym_envs = [
	"CartPole-v0", 
	"MountainCar-v0", 
	"Acrobot-v1", 
	"Pendulum-v0", 
	"MountainCarContinuous-v0", 
	"CarRacing-v0", 
	"BipedalWalker-v2", 
	"BipedalWalkerHardcore-v2", 
	"LunarLander-v2", 
	"LunarLanderContinuous-v2"
]

vzd_envs = [
	"basic",
	"my_way_home",
	"health_gathering",
	"predict_position",
	"defend_the_center",
	"defend_the_line",
	"take_cover",
]

env_names = [vzd_envs[-2], vzd_envs[-1], gym_envs[5]]
env_name = env_names[0]

def make_env(env_name=env_name):
	env = None
	if env_name in gym_envs:
		env = GymEnv(gym.make(env_name))
		env.unwrapped.verbose = 0
	elif env_name in vzd_envs:
		env = VizDoomEnv(env_name)
	return env
