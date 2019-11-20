import numpy as np
from matplotlib import pyplot as plt


class ACO:

    # 函数功能：AOC类的初始化
    # 函数输入：coordinates城市坐标，格式为coordinates = np.array([[565.0, 575.0], [25.0, 185.0]])
    # 函数输出：
    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.num_ant = 100                                                                   # 蚂蚁个数
        self.num_city = coordinate.shape[0]                                                 # 城市个数
        self.alpha = 1                                                                      # 信息素重要程度因子
        self.beta = 2                                                                       # 启发函数重要程度因子
        self.rho = 0.1                                                                      # 信息素的挥发速度
        self.Q = 1
        self.iter = 0
        self.iter_max = 50
        self.dist_mat = self.get_dist_mat()                                                 # 距离矩阵，i到j的矩阵
        self.eta_table = 1.0 / (self.dist_mat + np.diag([1e10] * self.num_city))            # 启发函数矩阵，表示蚂蚁从城市i转移到矩阵j的期望程度
        self.pheromone_table = np.ones((self.num_city, self.num_city))                      # 信息素矩阵
        self.path_table = np.zeros((self.num_ant, self.num_city), dtype=int)                # 路径记录表
        self.length_aver = np.zeros(self.iter_max)                                          # 各代路径的平均长度
        self.length_best = np.zeros(self.iter_max)                                          # 各代及其之前遇到的最佳路径长度
        self.path_best = np.zeros((self.iter_max, self.num_city), dtype=int)                # 各代及其之前遇到的最佳路径
        self.length = np.zeros(self.num_ant)                                                # 所有蚂蚁走一次后，整条路的长度

    # 函数功能：计算两两城市之间的距离
    # 函数输入：coordinates城市坐标，num_city城市数量
    # 函数输出：dist_mat距离矩阵
    def get_dist_mat(self):
        dist_mat = np.zeros((self.num_city, self.num_city))
        for i in range(self.num_city):
            for j in range(i, self.num_city):
                dist_mat[i][j] = dist_mat[j][i] = np.linalg.norm(self.coordinate[i] - self.coordinate[j])
        return dist_mat

    # 函数功能：随机产生各个蚂蚁的起点城市,(城市数比蚂蚁数多),(蚂蚁数比城市数多，需要补足)
    # 函数输入：path_table,路径表
    # 函数输出：path_table,为第一列填充开始城市
    def set_begin_city(self):
        if self.num_ant <= self.num_city:
            self.path_table[:, 0] = np.random.permutation(range(0, self.num_city))[:self.num_ant]
        else:
            self.path_table[:self.num_city, 0] = np.random.permutation(range(0, self.num_city))[:]
            self.path_table[self.num_city:, 0] = np.random.permutation(range(0, self.num_city))[:self.num_ant - self.num_city]

    # 函数功能：每只蚂蚁遍历城市所走的路径
    # 函数输入：
    # 函数输出：path_table，所有的蚂蚁的路径都记录在这里
    def ant_traversal(self):
        # 所有的蚂蚁都走一遍
        self.length = np.zeros(self.num_ant)
        for index_ant in range(self.num_ant):
            # 当前处于第一个城市，开始的城市，城市采用整数编码
            visiting = self.path_table[index_ant, 0]
            unvisited = list(range(self.num_city))
            unvisited.remove(visiting)
            # 遍历剩余的num_city-1个为遍历的城市
            for index_city in range(1, self.num_city):
                # 选择概率,启发值(heuristic value),没有走过的路，都有可能走，按照轮盘赌的方式选择接下来要走的路
                heuristic = np.zeros(len(unvisited))
                for k in range(len(unvisited)):
                    heuristic[k] = np.power(self.pheromone_table[visiting][unvisited[k]], self.alpha) * np.power(self.eta_table[visiting][unvisited[k]], self.beta)
                # 累计概率
                cum_sum_heuristic = (heuristic / sum(heuristic)).cumsum()
                cum_sum_heuristic -= np.random.rand()
                next_city = unvisited[np.where(cum_sum_heuristic > 0)[0][0]]  # 第一个概率大于零的就是下一个要访问的城市
                # 更新数据,路径和长度
                self.path_table[index_ant, index_city] = next_city
                self.length[index_ant] += self.dist_mat[visiting][next_city]
                unvisited.remove(next_city)
                visiting = next_city
            self.length[index_ant] += self.dist_mat[visiting][self.path_table[index_ant, 0]]  # 蚂蚁的路径距离包括最后一个城市和第一个城市的距离

    # 函数功能：更新信息素
    # 函数输入：
    # 函数输出：pheromone_table，更新之后的信息素
    def update_pheromone(self):
        if self.iter == 0:
            self.length_best[self.iter] = self.length.min()
            self.path_best[self.iter] = self.path_table[self.length.argmin()].copy()
        else:
            if self.length.min() > self.length_best[self.iter - 1]:
                self.length_best[self.iter] = self.length_best[self.iter - 1]
                self.path_best[self.iter] = self.path_best[self.iter - 1].copy()
            else:
                self.length_best[self.iter] = self.length.min()
                self.path_best[self.iter] = self.path_table[self.length.argmin()].copy()
        print(self.length_best[self.iter])
        print(self.path_best[self.iter])
        increment_pheromone_table = np.zeros((self.num_city, self.num_city))
        for i in range(self.num_ant):
            for j in range(self.num_city - 1):
                increment_pheromone_table[self.path_table[i, j]][self.path_table[i, j + 1]] += self.Q / self.dist_mat[self.path_table[i, j]][self.path_table[i, j + 1]]
            increment_pheromone_table[self.path_table[i, j + 1]][self.path_table[i, 0]] += self.Q / self.dist_mat[self.path_table[i, j + 1]][self.path_table[i, 0]]    # 加上回到起点的信息素增量
        self.pheromone_table = (1 - self.rho) * self.pheromone_table + increment_pheromone_table

    # 函数功能：一遍蚁群算法的过程
    # 函数输入：
    # 函数输出：
    def ACO_progress(self):
        while self.iter < self.iter_max:
            self.set_begin_city()
            self.ant_traversal()
            self.update_pheromone()
            self.iter += 1
        # 作出找到的最优路径图
        bestpath = self.path_best[-1]
        coordinates = self.coordinate
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker=u'$\cdot$')
        plt.xlim([0, 2000])
        plt.ylim([0, 2000])
        for i in range(self.num_city - 1):  #
            m, n = bestpath[i], bestpath[i + 1]
            plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')
        plt.plot([coordinates[bestpath[0]][0], coordinates[n][0]], [coordinates[bestpath[0]][1], coordinates[n][1]], 'b')
        plt.show()


if __name__ == '__main__':
    coordinates = np.array([
        [1380, 939],
        [2848, 96],
        [3510, 1671],
        [457, 334],
        [3888, 666],
        [984, 965],
        [2721, 1482],
        [1286, 525],
        [2716, 1432],
        [738, 1325],
        [1251, 1832],
        [2728, 1698],
        [3815, 169],
        [3683, 1533],
        [1247, 1945],
        [123, 862],
        [1234, 1946],
        [252, 1240],
        [611, 673],
        [2576, 1676],
        [928, 1700],
        [53, 857],
        [1807, 1711],
        [274, 1420],
        [2574, 946],
        [178, 24],
        [2678, 1825],
        [1795, 962],
        [3384, 1498],
        [3520, 1079],
        [1256, 61],
        [1424, 1728],
        [3913, 192],
        [3085, 1528],
        [2573, 1969],
        [463, 1670],
        [3875, 598],
        [298, 1513],
        [3479, 821],
        [2542, 236],
        [3955, 1743],
        [1323, 280],
        [3447, 1830],
        [2936, 337],
        [1621, 1830],
        [3373, 1646],
        [1393, 1368],
        [3874, 1318],
        [938, 955],
        [3022, 474],
        [2482, 1183],
        [3854, 923],
        [376, 825],
        [2519, 135],
        [2945, 1622],
        [953, 268],
        [2628, 1479],
        [2097, 981],
        [890, 1846],
        [2139, 1806],
        [2421, 1007],
        [2290, 1810],
        [1115, 1052],
        [2588, 302],
        [327, 265],
        [241, 341],
        [1917, 687],
        [2991, 792],
        [2573, 599],
        [19, 674],
        [3911, 1673],
        [872, 1559],
        [2863, 558],
        [929, 1766],
        [839, 620],
        [3893, 102],
        [2178, 1619],
        [3822, 899],
        [378, 1048],
        [1178, 100],
        [2599, 901],
        [3416, 143],
        [2961, 1605],
        [611, 1384],
        [3113, 885],
        [2597, 1830],
        [2586, 1286],
        [161, 906],
        [1429, 134],
        [742, 1025],
        [1625, 1651],
        [1187, 706],
        [1787, 1009],
        [22, 987],
        [3640, 43],
        [3756, 882],
        [776, 392],
        [1724, 1642],
        [198, 1810],
        [3950, 1558]
    ])
    ACO = ACO(coordinates)
    ACO.ACO_progress()
