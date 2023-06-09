from itertools import count
import json
import csv
import os

import time
import numpy as np

from utils import *
from reporter import *

def main(train_info, train_case, reporter, repeat_num):
    # train info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_NAME = train_info['TRAIN_NAME']
    ENV_NAME = train_info['ENV_NAME']

    EPISODE_NUM = int(train_info['EPISODE_NUM'])

    OBS_NUM = int(train_info['OBS_NUM'])
    ACT_QUANT_NUM = float(train_info['ACT_QUANT_NUM'])

    MEMORY_SIZE = int(train_info['MEMORY_SIZE'])

    # train case info
    TRAIN_CASE = train_case['TRAIN_CASE']
    TRAIN_AGENT_NUM = int(train_case['TRAIN_AGENT_NUM'])
    EPSILON = float(train_case['EPSILON'])
    GAMMA = train_case['GAMMA']

    # save train
    result_file = open("results/" + TRAIN_NAME + "/" + TRAIN_CASE + "-" + str(repeat_num) + ".csv", "w")
    result_writer = csv.writer(result_file)

    # rest of the pre-setting
    action_list = np.array(range(-20,21)) / ACT_QUANT_NUM
    # action_list = [-1,0,1]

    # agent and replay buffers define
    agent_list = []
    for i in range(TRAIN_AGENT_NUM):
        if i >= len(GAMMA):
            gamma = float(train_case['OTHER_GAMMA'])
        else:
            gamma = float(GAMMA[i]) 

        agent_list.append(
            DQNagent(OBS_NUM, action_list, ENV_NAME, device, EPSILON, gamma)
        )
    
    for i, agent in enumerate(agent_list):
        agent.saveONNX(TRAIN_CASE + "_" + str(repeat_num)  + "_" + str(i) + "_start", "./onnx/" +  TRAIN_NAME + "/")

    memory = ReplayMemory(MEMORY_SIZE)

    ADVISE_RATE = 0.2

    try:
        for i_episode in range(EPISODE_NUM):
            # initialization 
            reward_sum = [0 for k in range(TRAIN_AGENT_NUM)]
            dones = [0 for k in range(TRAIN_AGENT_NUM)]
            rewards = [0 for k in range(TRAIN_AGENT_NUM)]

            for agent in agent_list:
                agent.reset()

            for t in count():
			
                if len(agent_list) == 1:
                    done, reward = agent_list[0].train(memory, agent_list[0])

                    dones[0] = done
                    rewards[0] = reward
                else:
                    if ADVISE_RATE > random.random():
                        done0, reward0 = agent_list[0].train(memory, agent_list[1])
                        done1, reward1 = agent_list[1].train(memory, agent_list[0])
                        
                        dones = [done0, done1]
                        rewards = [reward0, reward1]
                    else:
                        done0, reward0 = agent_list[0].train(memory, agent_list[0])
                        done1, reward1 = agent_list[1].train(memory, agent_list[1])

                        dones = [done0, done1]
                        rewards = [reward0, reward1]

                reward_sum = [reward_sum[l] + rewards[l] for l in range(TRAIN_AGENT_NUM)]
                all_done = (sum(dones) == len(dones))

                if all_done:
                    # reporter.info(f"EPISODE {i_episode} \t| REWARD {np.round(reward_sum, 3)} \t| STEP {t}")
                    result_writer.writerow(reward_sum)
                    break
    finally:

        for i, agent in enumerate(agent_list):
            agent.saveONNX(TRAIN_CASE + "_" + str(repeat_num) + "_"  + str(i) + "_end", "./onnx/" +  TRAIN_NAME + "/")

        result_file.close()
        reporter.info('Complete')

if __name__ == "__main__":

    with open("train_info.json", "r") as train_info_json:
        train_info_file = json.load(train_info_json)
    train_info_json.close()

    train_info = train_info_file["TRAIN_INFO"]

    reporter = reporter_loader("info", train_info['TRAIN_NAME'])
    try:
        os.mkdir("./results/" + train_info['TRAIN_NAME'])
    except:
        pass

    try:
        os.mkdir("./onnx/" + train_info['TRAIN_NAME'])
    except:
        pass

    TRAIN_REPEAT = int(train_info['TRAIN_REPEAT'])
    
    for j, train_case in enumerate(train_info_file["TRAIN_CASE"]):
        for k in range(TRAIN_REPEAT):
            reporter.info(f"train_case: {j}")
            
            start = time.time()
            main(train_info, train_case, reporter, k)
            end = time.time()

            reporter.info(f" {end-start}")
