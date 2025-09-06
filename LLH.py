import random
import numpy as np
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.DE import BaseDE
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.ACOR import OriginalACOR


class LLH:
    def __init__(self, num, pru_req, tr, pop_size):
        # ####################################初始解参数######################################
        self.P = 0.5  # 初始种群中来源于接收解的比例
        #####################################################################################
        # ###########################元启发式算法通用参数######################################
        self.num = num  # 优化变量数
        self.pru_req = pru_req  # 裁剪请求
        self.pop_size = pop_size  # 种群大小
        self.tr = tr  # 本地保存交易信息
        self.pt = 1e5  # 惩罚因子目标函数
        self.epoch = 500  # 迭代次数
        self.l = [0] * self.num  # 最小值
        self.u = self.tr.trNum - 1  # 最大值
        self.problem_dict = {  # 解决问题字典
            "fit_func": self.fitness_function,
            "lb": self.l,
            "ub": self.u,
            "minmax": "min"
        }
        #####################################################################################
        # ###########################遗传算法参数#############################################
        self.pc = 0.9
        self.pm = 0.05
        #####################################################################################
        # ###########################差分优化算法参数#########################################
        self.wf = 0.7
        self.cr = 0.9
        self.strategy = 1
        #####################################################################################
        # ###########################粒子群算法参数###########################################
        self.c1 = 0.5
        self.c2 = 0.5
        self.w_min = 0.4
        self.w_max = 0.9
        #####################################################################################
        # ###########################蚁群算法参数#############################################
        self.sample_count = int(self.pop_size / 2)
        self.intent_factor = 0.5
        self.zeta = 1.0
        #####################################################################################

    # 计算适应度斐波纳挈列表，这里是为了求出累积的适应度
    def cumsum(self, new_fitness):
        for i in range(len(new_fitness) - 1):
            new_fitness[i + 1] += new_fitness[i]
        new_fitness[len(new_fitness) - 1] = 1

    def selection(self, solutions):
        new_fitness = []  # 新的适应度
        pop_len = len(solutions)  # 接收解的数量
        Num = int(self.pop_size * self.P)  # 抽取的数量
        new_pop = [0] * Num  # 新种群
        fitness = [1 / solutions[i][1][0] for i in range(pop_len)]  # 获取适应度
        total_fitness = sum(fitness)  # 将所有的适应度求和
        # 将所有个体的适应度概率化,类似于softmax
        for i in range(len(fitness)):
            new_fitness.append(fitness[i] / total_fitness)
        self.cumsum(new_fitness)  # 将所有个体的适应度正则化
        ms = []  # 抽取的种群
        # 根据随机数确定抽取哪几个
        for i in range(Num):
            ms.append(random.random())  # 产生种群个数的随机值
        ms.sort()  # 抽取的种群排序
        fitin = 0
        newin = 0
        # 轮盘赌方式
        while newin < Num:
            if ms[newin] < new_fitness[fitin]:
                new_pop[newin] = solutions[fitin]
                newin += 1
            else:
                fitin += 1
        return new_pop

    def iniPopInAcceptSolution(self, solutions):  # 获取初始种群中接收解
        betterSolution = None
        if solutions is not None:
            Num = int(self.pop_size * self.P)
            if len(solutions) <= Num:
                betterSolution = solutions
            else:
                betterSolution = self.selection(solutions)
        return betterSolution

    def fitness_function(self, p):
        x = np.array(p)  # 转化为一维数组
        total_utility = 0
        total_size = 0
        for i in range(self.num):
            pru_num = int(x[i])
            total_utility += self.tr.trSumUtility[i][pru_num]
            total_size += self.tr.trSumSize[i][pru_num]
        return abs(total_utility) + max(0, self.pru_req - total_size) * self.pt

    def search_GA(self, AcceptSolutionList=None):
        BetterSolutionList = self.iniPopInAcceptSolution(AcceptSolutionList)
        model1 = BaseGA(self.epoch, self.pop_size, self.pc, self.pm)
        best_position, best_fitness = model1.solve(self.problem_dict, betterpop=BetterSolutionList)
        return best_position

    def search_DE(self, AcceptSolutionList=None):
        BetterSolutionList = self.iniPopInAcceptSolution(AcceptSolutionList)
        model = BaseDE(self.epoch, self.pop_size, self.wf, self.cr, self.strategy)
        best_position, best_fitness = model.solve(self.problem_dict, betterpop=BetterSolutionList)
        return best_position

    def search_PSO(self, AcceptSolutionList=None):
        BetterSolutionList = self.iniPopInAcceptSolution(AcceptSolutionList)
        model = OriginalPSO(self.epoch, self.pop_size, self.c1, self.c2, self.w_min, self.w_max)
        best_position, best_fitness = model.solve(self.problem_dict, betterpop=BetterSolutionList)
        return best_position

    def search_ACO(self, AcceptSolutionList=None):
        BetterSolutionList = self.iniPopInAcceptSolution(AcceptSolutionList)
        model = OriginalACOR(self.epoch, self.pop_size, self.sample_count, self.intent_factor, self.zeta)
        best_position, best_fitness = model.solve(self.problem_dict, betterpop=BetterSolutionList)
        return best_position