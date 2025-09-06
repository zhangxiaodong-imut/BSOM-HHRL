import numpy as np
from TransactionInfo import TransactionInfo
from LLH import LLH


class Environment:
    def __init__(self, pru_req, pop_size):
        self.nS = 2  # 状态数量
        self.nA = 4  # 动作数量
        self.pru_req = pru_req  # 裁剪请求
        self.pop_size = pop_size  # 种群大小
        self.num = 20  # 本地保存的交易的车辆数
        self.tr = TransactionInfo(self.num)  # 本地保存交易信息
        self.currentState = 0  # 当前状态
        self.LLH = LLH(self.num, self.pru_req, self.tr, self.pop_size)  # 低层元启发式算子

    def reset(self):
        self.currentState = 0
        return self.currentState

    def step(self, action, acceptSolution):
        if action == 0:
            strategy, total_u, total_s = self.pru_result(self.LLH.search_GA(AcceptSolutionList=acceptSolution[0]))
        if action == 1:
            strategy, total_u, total_s = self.pru_result(self.LLH.search_DE(AcceptSolutionList=acceptSolution[1]))
        if action == 2:
            strategy, total_u, total_s = self.pru_result(self.LLH.search_PSO(AcceptSolutionList=acceptSolution[2]))
        if action == 3:
            strategy, total_u, total_s = self.pru_result(self.LLH.search_ACO(AcceptSolutionList=acceptSolution[3]))
        return strategy, total_u, total_s

    def pru_result(self, best):
        strategy = np.array(best)  # 转化为一维数组
        total_u = 0
        total_s = 0
        for i in range(self.num):
            pru_num = int(strategy[i])
            # strategy[i] = pru_num + 1
            strategy[i] = pru_num
            total_u += self.tr.trSumUtility[i][pru_num]
            total_s += self.tr.trSumSize[i][pru_num]
        return strategy, total_u, total_s