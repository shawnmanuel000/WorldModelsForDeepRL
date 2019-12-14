import os
import re
import numpy as np
import matplotlib.pylab as plt
from collections import deque

ctrl = "logs_0.txt"
ctrl0 = "logs_10.txt"
ctrl1 = "logs_9.txt"

ddpg = "logs_11.txt"
# ddpg0 = "logs_10.txt"
ddpg0 = "iter1/logs_3.txt"
# ddpg1 = "logs_8.txt"
ddpg1 = "iter1/logs_6.txt"

ppo = "logs_13.txt"
ppo0 = "logs_11.txt"
ppo1 = "logs_15.txt"
# ppo1 = "iter1/logs_2.txt"

ctrls = [ctrl0, ctrl1, ctrl]
ddpgs = [ddpg0, ddpg1, ddpg]
ppos = [ppo0, ppo1, ppo]

def read_ctrl(path):
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

def read_a3c(path):
	rewards = []
	avgs = deque(maxlen=100)
	rolling = []
	with open(path, "r") as f:
		for line in f:
			match = re.match("^Ep.*Test: ([^ ]*).*, Avg: ([^ ]*)", line.strip('\n'))
			if match:
				rewards.append(float(match.groups()[0]))
				avgs.append(float(match.groups()[0]))
				rolling.append(np.mean(avgs))
	return rewards, rolling

def graph_ctrl():
	bests, rolling = zip(*[read_ctrl(f"./logs/controller/{path}") for path in ctrls])
	plt.plot(range(len(bests[-1])), bests[-1], color="#ADFF2F", linewidth=0.5, label="Best of Baseline")
	plt.plot(range(len(bests[0])), bests[0], color="#00BFFF", linewidth=0.5, label="Best of Iteration 1")
	plt.plot(range(len(bests[1])), bests[1], color="#FF1493", linewidth=0.5, label="Best of Iteration 2")
	plt.plot(range(len(rolling[-1])), rolling[-1], color="#008000", label="Avg of Baseline")
	plt.plot(range(len(rolling[0])), rolling[0], color="#0000CD", label="Avg of Iteration 1")
	plt.plot(range(len(rolling[1])), rolling[1], color="#FF0000", label="Avg of Iteration 2")
	print(f"Max-1: {max(bests[-1]):.0f}, Max0: {max(bests[0]):.0f}, Max1: {max(bests[1]):.0f}")
	print(f"Avg-1: {max(rolling[-1]):.0f}, Avg0: {max(rolling[0]):.0f}, Avg1: {max(rolling[1]):.0f}")
	
	plt.legend(loc="best", bbox_to_anchor=(0.6,0.5))
	plt.title("CMA-ES Best Rewards")
	plt.xlabel("Generation")
	plt.ylabel("Best Total Score")
	plt.grid(linewidth=0.3, linestyle='-')

def graph_a3c(model="ddpg", logs=ddpgs):
	# _, ravgs = read_a3c("./logs/qlearning/random.txt")
	rewards, qavgs = zip(*[read_a3c(f"./logs/{model}/{path}") for path in logs])
	plt.plot(range(len(rewards[-1])), rewards[-1], color="#ADFF2F", linewidth=0.5, label="Baseline")
	plt.plot(range(len(rewards[0])), rewards[0], color="#00BFFF", linewidth=0.5, label="Using Iteration 1 WM")
	plt.plot(range(len(rewards[1])), rewards[1], color="#FF1493", linewidth=0.5, label="Using Iteration 2 WM")
	plt.plot(range(len(qavgs[-1])), qavgs[-1], color="#008000", label="Avg Baseline")
	plt.plot(range(len(qavgs[0])), qavgs[0], color="#0000CD", label="Avg Using Iteration 1 WM")
	plt.plot(range(len(qavgs[1])), qavgs[1], color="#FF0000", label="Avg Using Iteration 2 WM")
	# plt.plot(range(len(ravgs[:len(rewards[0])])), ravgs[:len(rewards[0])], color="#FF0000", label="Avg Random")
	print(f"Max-1: {max(rewards[-1]):.0f}, Max0: {max(rewards[0]):.0f}, Max1: {max(rewards[1]):.0f}")
	print(f"Avg-1: {max(qavgs[-1]):.0f}, Avg0: {max(qavgs[0]):.0f}, Avg1: {max(qavgs[1]):.0f}")
	
	plt.legend(loc="upper left" if model=="ppo" else "best")
	plt.title(f"{model.upper()} Training Rewards")
	plt.xlabel("Rollout")
	plt.ylabel("Total Score")
	plt.grid(linewidth=0.3, linestyle='-')

def main():
	graph_ctrl()
	plt.figure()
	graph_a3c("ddpg", ddpgs)
	plt.figure()
	graph_a3c("ppo", ppos)
	plt.show()

main()