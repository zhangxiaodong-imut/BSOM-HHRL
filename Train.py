import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from QLearningAgent import QLearningAgent
from Environment import Environment


class Train:
    def __init__(self, pru_req, pop_size):
        self.pru_req = pru_req
        self.pop_size = pop_size
        self.bestStrategy = []  # 最优裁剪策略
        self.bestStrategy_U = -1e5  # 最优裁剪策略的裁剪效用
        self.bestStrategy_S = 0  # 最优裁剪策略的裁剪交易大小
        self.env = Environment(self.pru_req, self.pop_size)  # 环境
        self.agent = QLearningAgent(self.env)  # 智能体
        self.eposide = 100  # 调用低层LLH的次数
        self.bestSolution = [-15.31, -32.51, -73.69, -173.41, -415.62]
        self.offset = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        # self.acceptSolution = []
        self.acceptSolution = []
        self.actionList = []

    def reset(self):
        observation = self.env.reset()
        self.agent.reset()
        self.acceptSolution.clear()
        self.actionList.clear()
        return observation

    def play_qlearning(self):
        observation = self.reset()
        T = 1
        self.acceptSolution.append([])
        self.acceptSolution.append([])
        self.acceptSolution.append([])
        self.acceptSolution.append([])
        while T <= self.eposide:
            factor = T / self.eposide
            action = self.agent.decide(observation, factor)
            if len(self.acceptSolution) == 0:
                strategy, total_u, total_s = self.env.step(action, None)
            else:
                strategy, total_u, total_s = self.env.step(action, self.acceptSolution)
            # print("裁剪效用", total_u)
            # print("裁剪大小", total_s)
            # print("裁剪方案", strategy)
            if total_s >= self.env.pru_req:
                if T == 1:
                    reward = 0.0001
                else:
                    # reward = (total_u - self.bestStrategy_U)
                    reward = (total_u - self.bestStrategy_U) / abs(self.bestStrategy_U)
                # print("reward:", reward)
                if reward > 0:
                    self.bestStrategy = strategy
                    self.bestStrategy_U = total_u
                    self.bestStrategy_S = total_s
                    next_observation = 0
                else:
                    next_observation = 1
                self.agent.learn(observation, action, reward, next_observation)
                observation = next_observation
                # self.acceptSolution.append([strategy, [-total_u, [-total_u]]])
                self.acceptSolution[action].append([strategy, [-total_u, [-total_u]]])
                self.actionList.append(action)
                T += 1

        return self.bestStrategy_U

    def getAvg(self, utility, pru):
        return self.bestSolution[pru] - np.mean(utility)

    def getDert(self, utility, pru):
        avg = self.bestSolution[pru] - np.mean(utility)
        sum = 0
        for index in range(len(utility)):
            sum += (self.bestSolution[pru] - utility[index] - avg) ** 2
        return sum / len(utility)

    def getSucc(self, utility, pru):
        success = []
        for i in range(len(self.offset)):
            obj = (2 - self.offset[i]) * self.bestSolution[pru]
            sum = 0
            for j in range(len(utility)):
                if utility[j] >= obj:
                    sum += 1
            success.append(sum / len(utility))
        return success


popsizeList = [20, 30, 40, 50, 60]
pruList = [10240, 20480, 40960, 81920, 163840]
for pru in pruList:  # 裁剪请求大小
    for popsize in range(5):  # 种群大小
        print("裁剪请求为{}，种群大小为{}\n".format(pruList[pru], popsizeList[popsize]))
        test = Train(pruList[pru], popsizeList[popsize])
        utility = []
        useTimes = [0] * 4  # 调用低层算子次数
        times = []
        for index in range(100):
            print("第", index + 1, "次:")
            time1 = time.time()
            utility.append(test.play_qlearning())
            time2 = time.time()
            times.append(time2 - time1)
            counter_dict = {}
            for item in test.actionList:
                if item in counter_dict:
                    counter_dict[item] += 1
                else:
                    counter_dict[item] = 1
            for i in range(4):
                if i in counter_dict:
                    useTimes[i] += counter_dict[i]
            # print(test.actionList)
            # print(counter_dict)
            # print(test.bestStrategy_U)
            # print(test.agent.q)

        print("平均综合效用={}\n".format(np.mean(utility)))
        print("最优综合效用={}\n".format(np.max(utility)))
        print("平均目标偏差={}\n".format(test.getAvg(utility, pru)))
        print("目标标准方差={}\n".format(test.getDert(utility, pru)))
        print("成功率={}\n".format(" ".join(map(str, test.getSucc(utility, pru)))))
        print("调用次数={}\n".format(" ".join(map(str, useTimes))))
        with open(str(pruList[pru]) + "-" + str(popsizeList[popsize]) + "-" + "RL_result.txt", 'a') as f_result:
            f_result.seek(0)
            f_result.truncate()
            f_result.writelines("裁剪请求为{}，种群大小为{}\n".format(pruList[pru], popsizeList[popsize]))
            f_result.writelines("平均综合效用={}\n".format(np.mean(utility)))
            f_result.writelines("最优综合效用={}\n".format(np.max(utility)))
            f_result.writelines("平均目标偏差={}\n".format(test.getAvg(utility, pru)))
            f_result.writelines("目标标准方差={}\n".format(test.getDert(utility, pru)))
            f_result.writelines("成功率={}\n".format(" ".join(map(str, test.getSucc(utility, pru)))))
            f_result.writelines("调用次数={}\n".format(" ".join(map(str, useTimes))))
        with open(str(pruList[pru]) + "-" + str(popsizeList[popsize]) + "-" + "RL.txt", 'a') as f_out:
            f_out.seek(0)
            f_out.truncate()
            f_out.writelines(" ".join(map(str, utility)))
            f_out.writelines(" ".join(map(str, times)))
        # print("结果是:")
        # print(useTimes)
        # print("平均:", np.mean(utility))
        # print("最优:", np.max(utility))