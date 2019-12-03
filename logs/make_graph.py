import os
import re
import numpy as np
import matplotlib.pylab as plt
from collections import deque

ctrl0 = "logs_10.txt"
ctrl1 = "logs_9.txt"

ddpg = "logs_9.txt"
ddpg0 = "logs_7.txt"
ddpg1 = "logs_8.txt"

ppo = "logs_9.txt"
ppo0 = "logs_8.txt"
ppo1 = "logs_7.txt"

def read_qlearning(path="./logs/qlearning/logs_3.txt"):
	rewards = []
	avgs = []
	with open(path, "r") as f:
		for line in f:
			match = re.match(".*Reward: (.*), Avg: (.*)", line.strip('\n'))
			if match:
				rewards.append(float(match.groups()[0]))
				avgs.append(float(match.groups()[1]))
	return rewards, avgs

def read_ctrl(path=f"./logs/ctrl/{ctrl0}"):
	bests = []
	mins = []
	avgs = deque(maxlen=100)
	rolling = []
	with open(path, "r") as f:
		for line in f:
			match = re.match(".*score: (.*), Min: (.*), Avg: (.*)", line.strip('\n'))
			if match:
				bests.append(float(match.groups()[0]))
				mins.append(float(match.groups()[1]))
				avgs.append(float(match.groups()[0]))
				rolling.append(np.mean(avgs))
	return bests, mins, avgs, rolling

def main():
	_, ravgs = read_qlearning("./logs/qlearning/random.txt")
	rewards, qavgs = read_qlearning()
	bests, mins, cavgs, rolling = read_ctrl()

	plt.plot(range(len(rewards)), rewards, label="100-Rolling Average")
	plt.plot(range(len(rewards)), qavgs, label="Rewards")
	plt.plot(range(len(rewards)), ravgs, 'r', label="Random Average")
	plt.legend(loc="lower right")
	plt.title("Actor-Critic Training Rewards")
	plt.xlabel("Rollout")
	plt.ylabel("Total Score")
	plt.figure()
	plt.plot(range(len(bests)), bests, label="Max iteration score")
	plt.plot(range(len(bests)), rolling, label="100-Rolling Max Average")
	plt.plot(range(len(rewards)), ravgs, 'r', label="Random Average")
	plt.legend(loc="best")
	plt.title("World Model Rewards")
	plt.xlabel("Iteration")
	plt.ylabel("Best Total Score")
	plt.show()

main()