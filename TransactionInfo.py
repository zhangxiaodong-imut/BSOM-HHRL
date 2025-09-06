from numpy import *
import numpy as np
import matplotlib.pyplot as plt


class TransactionInfo:
    def __init__(self, nodeNum):
        self.nodeNum = nodeNum
        self.nodePru_speed = 20
        self.path_nodeinfo = 'data/nodeinfo.txt'
        self.path_trinfo = 'data/trinfo.txt'
        self.tau = 0.1
        self.max_speed = 20
        self.u_cop = 0.25
        self.u_tim = 0.0028
        self.u_dis = 0.0003
        self.u_siz = 50
        self.u_sto = 100
        self.a = 0.1047
        self.b = 0.2583
        self.c = 0.6370
        self.phi = 10
        self.nodeInfo = self.readnodeInfo()
        self.trNum, self.trInfo = self.readtrInfo()
        self.trUtility = self.getUtility()
        self.trSumUtility, self.trSumSize = self.getSum()

    # 读取保存到本地的节点胡基本信息（速度、运动关系、距离）
    def readnodeInfo(self):
        info = np.zeros((self.nodeNum, 3))
        with open(self.path_nodeinfo, 'r') as file:
            for index in range(self.nodeNum):
                line = file.readline().strip("\n")
                list1 = line.split(" ")
                list2 = [int(i) for i in list1]
                info[index] = list2  # 速度、运动关系、距离
        return info

    # 读取保存到本地的节点的交易信息，交易账户、时间戳、交易ID、交易大小、保存数量
    def readtrInfo(self):
        trNum = np.zeros(self.nodeNum)
        trinfo = []
        with open(self.path_trinfo, 'r') as file:
            for index in range(self.nodeNum):
                line = file.readline().strip("\n")
                Num = line.split(" ")[1]  # 保存的节点index的交易的数量
                trNum[index] = int(Num)
                trinfo_node = np.zeros((int(Num), 5))  # 交易账户、时间戳、交易ID、交易大小、保存数量
                for index2 in range(int(Num)):
                    line = file.readline().strip("\n")
                    list1 = line.split(" ")
                    list2 = [int(i) for i in list1]
                    if list2[3] == 0:
                        list2[3] += 1
                    trinfo_node[index2] = list2
                trinfo.append(trinfo_node)
        return trNum, trinfo

    # 获取每条交易的效用
    def getUtility(self):
        utilityinfo = []
        currentTime = 1800
        for i in range(self.nodeNum):
            type = self.nodeInfo[i][1]
            node_u = []
            # 计算下次相遇概率
            Pr = 0
            if type == 0:
                Pr = np.exp(-self.u_dis * self.nodeInfo[i][2]) * (1 + self.tau * (self.nodePru_speed + self.nodeInfo[i][0]) / (2 * self.max_speed))
            if type == 1:
                Pr = np.exp(-self.u_dis * self.nodeInfo[i][2]) * (1 - self.tau * (self.nodePru_speed + self.nodeInfo[i][0]) / (2 * self.max_speed))
            if type == 2:
                Pr = np.exp(-self.u_dis * self.nodeInfo[i][2]) * (1 + self.tau * abs(self.nodePru_speed - self.nodeInfo[i][0]) / (2 * self.max_speed))
            if type == 3:
                Pr = np.exp(-self.u_dis * self.nodeInfo[i][2]) * (1 - self.tau * abs(self.nodePru_speed - self.nodeInfo[i][0]) / (2 * self.max_speed))
            # print(i+1, Pr)

            for j in range(len(self.trInfo[i])):
                sec = -np.exp(-self.u_cop * self.trInfo[i][j][4])  # 安全
                tim = -np.exp(-self.u_tim * (currentTime - self.trInfo[i][j][1]))  # 时效性
                net = -np.exp(-self.u_siz * (1 / (Pr * self.trInfo[i][j][3])))  # 网络资源
                sto = np.exp(-self.u_sto * (1 / self.trInfo[i][j][3]))  # 存储资源
                # U = self.a * sec + self.c * net + self.d * sto
                # U = self.a * sec + self.b * tim + self.c * net + self.d * sto
                U = self.a * sec + self.b * tim + self.c * net
                node_u.append(U)
            utilityinfo.append(node_u)
        return utilityinfo

    # 计算每个节点的累计效用和交易大小
    def getSum(self):
        sum_utility = []
        sum_size = []
        for i in range(self.nodeNum):
            u_sum = 0
            s_sum = 0
            U = []
            S = []
            for j in range(len(self.trUtility[i])):
                u_sum += self.trUtility[i][j]
                s_sum += self.trInfo[i][j][3]
                U.append(u_sum)
                S.append(s_sum)
            sum_utility.append(U)
            sum_size.append(S)
        return sum_utility, sum_size


# tr = TransactionInfo()
# print(tr.trUtility)
#
# plt.plot(test.trUtility[19])
# plt.show()