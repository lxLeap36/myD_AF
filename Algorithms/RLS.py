import numpy as np

class RLSFilter:

    def __init__(self, filter_length=8, lambda_=0.98, delta=0.1):
        """
        初始化RLS滤波器

        参数:
            filter_length: 滤波器长度
            lambda_: 遗忘因子(0 < lambda_ <= 1)
            delta: 初始化P矩阵的常数(delta > 0)
        """
        self.filter_length = filter_length
        self.lambda_ = lambda_
        self.delta = delta
        # 初始化权重和输入缓冲区
        self.weights = np.zeros(filter_length)
        self.x_buffer = np.zeros(filter_length)
        # 初始化P矩阵 (P = delta * I)
        self.P = np.eye(filter_length) * delta
        # 历史记录
        self.error_history = []
        self.weight_history = []

    def update(self, x_n, d_n):
        """
        单步更新RLS滤波器

        参数:
            x_n: 当前输入样本
            d_n: 当前期望输出

        返回:
            e_n: 当前误差
        """
        # 更新输入缓冲区
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x_n
        # 计算先验输出
        y_n = np.dot(self.weights, self.x_buffer)
        # 计算先验误差
        e_n = d_n - y_n
        # 计算增益向量
        P_x = np.dot(self.P, self.x_buffer)
        denom = self.lambda_ + np.dot(self.x_buffer, P_x)
        k_n = P_x / denom
        # 更新权重
        self.weights += k_n * e_n
        # 更新P矩阵
        k_xT = np.outer(k_n, self.x_buffer)
        self.P = (self.P - np.dot(k_xT, self.P)) / self.lambda_
        # 保存历史数据
        self.weight_history.append(self.weights.copy())
        self.error_history.append(e_n)

        return e_n

    def reset(self):
        """重置滤波器状态"""
        self.weights = np.zeros(self.filter_length)
        self.x_buffer = np.zeros(self.filter_length)
        self.P = np.eye(self.filter_length) * self.delta
        self.weight_history = []
        self.error_history = []

    def process(self, x, d):
        """
        处理整个信号

        参数:
            x: 输入信号
            d: 期望输出信号

        返回:
            e: 误差信号
        """
        min_len = min(len(x), len(d))
        x = x[:min_len]
        d = d[:min_len]

        e = np.zeros(min_len)
        for n in range(min_len):
            e[n] = self.update(x[n], d[n])
        return e