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

ctrls = [ctrl0, ctrl1]
ddpgs = [ddpg0, ddpg1, ddpg]
ppos = [ppo0, ppo1, ppo]

def read_ctrl(path=f"./logs/ctrl/{ctrl0}"):
	bests = []
	avgs = deque(maxlen=100)
	rolling = []
	with open(path, "r") as f:
		for line in f:
			match = re.match("Ep.*score: (.*), Min: (.*), Avg: (.*)", line.strip('\n'))
			if match:
				bests.append(float(match.groups()[0]))
				avgs.append(float(match.groups()[0]))
				rolling.append(np.mean(avgs))
	return bests, rolling

def read_a3c(path="./logs/qlearning/logs_3.txt"):
	rewards = []
	avgs = []
	with open(path, "r") as f:
		for line in f:
			match = re.match(".*Test: (.*), Avg: (.*)", line.strip('\n'))
			if match:
				rewards.append(float(match.groups()[0]))
				avgs.append(float(match.groups()[1]))
	return rewards, avgs

def graph_ctrl():
	# _, ravgs = read_a3c("./logs/qlearning/random.txt")
	bests, rolling = zip(*[read_ctrl(f"./logs/ctrl/{path}") for path in ctrls])
	plt.plot(range(len(bests[0])), bests[0], color="#00BFFF", linewidth=0.5, label="Baseline (10000 random rollouts)")
	plt.plot(range(len(bests[1])), bests[1], color="#FF1493", linewidth=0.5, label="Improved (1000 random + 1000 policy)")
	plt.plot(range(len(rolling[0])), rolling[0], color="#0000CD", label="Avg Baseline")
	plt.plot(range(len(rolling[1])), rolling[1], color="#FF0000", label="Avg Improved")
	# plt.plot(range(len(ravgs[:len(rolling[0])])), ravgs[:len(rolling[0])], color="#FF0000", label="Avg Random")
	
	plt.legend(loc="best")
	plt.title("World Model Rewards")
	plt.xlabel("Iteration")
	plt.ylabel("Best Total Score")

def graph_a3c(model="ddpg", logs=ddpgs):
	# _, ravgs = read_a3c("./logs/qlearning/random.txt")
	rewards, qavgs = zip(*[read_a3c(f"./logs/{model}/{path}") for path in logs])
	plt.plot(range(len(rewards[-1])), rewards[-1], color="#ADFF2F", linewidth=0.5, label="Baseline")
	plt.plot(range(len(rewards[0])), rewards[0], color="#00BFFF", linewidth=0.5, label="Using WM")
	plt.plot(range(len(rewards[1])), rewards[1], color="#FF1493", linewidth=0.5, label="Using WM Improved")
	plt.plot(range(len(qavgs[-1])), qavgs[-1], color="#008000", label="Avg Baseline")
	plt.plot(range(len(qavgs[0])), qavgs[0], color="#0000CD", label="Avg Using WM")
	plt.plot(range(len(qavgs[1])), qavgs[1], color="#FF0000", label="Avg Using WM Improved")
	# plt.plot(range(len(ravgs[:len(rewards[0])])), ravgs[:len(rewards[0])], color="#FF0000", label="Avg Random")
	
	plt.legend(loc="lower right")
	plt.title(f"{model.upper()} Training Rewards")
	plt.xlabel("Rollout")
	plt.ylabel("Total Score")


def main():
	graph_ctrl()
	plt.figure()
	graph_a3c("ddpg", ddpgs)
	plt.figure()
	graph_a3c("ppo", ppos)
	plt.show()

main()