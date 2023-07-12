from itertools import count
import json
import csv
import os

import time
import numpy as np

from collections import deque

from utils import *
from reporter import *

import matplotlib.pyplot as plt

def main():
	TRAIN_NAME = "test_train9"
	NN_NAME = "0_0_0_end"

	TEST_NUM = 10
	QUEUE_SIZE = 10

	ACT_QUANT_NUM = 10 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	NN_PATH = "./onnx/" + TRAIN_NAME + NN_NAME

	action_list = np.array(range(-20,21)) / ACT_QUANT_NUM

	agent = DQNagent(3, action_list, "Pendulum-v1", device, 0, 0.99)
	agent.loadONNX(NN_PATH)

	reward_list = []
	return_list = []

	x_list = []
	y_list = []

	for i in range(TEST_NUM):
		agent.reset()
		done = False
		while not done:
			done, reward = agent.step()
			state = agent.state
			state = state[0]

			x = state[0].cpu().item()
			y = state[1].cpu().item()

			x_list.append(x)
			y_list.append(y)

	plt.plot(x_list)
	pass
			


if __name__ == "__main__":
	main()