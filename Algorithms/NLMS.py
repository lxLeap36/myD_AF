import numpy as np

class NLMSFilter:
    def __init__(self, filter_length=8, step_size=0.8, epsilon=1e-1):  #1e-6
        self.filter_length = filter_length
        self.step_size = step_size
        self.epsilon = epsilon

        self.weights = np.zeros(filter_length)
        self.x_buffer = np.zeros(filter_length)

        self.error_history = []
        self.weight_history = []

    def update(self, x_n, d_n):
        # 更新输入缓冲区
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x_n
        # 计算滤波器输出
        y_n = np.dot(self.weights, self.x_buffer)
        # 计算误差信号
        e_n = d_n - y_n
        # 实时归一化因子
        norm_factor = np.dot(self.x_buffer, self.x_buffer) + self.epsilon
        # 权重更新
        self.weights += self.step_size * e_n * self.x_buffer / norm_factor
        # 保存历史数据（可选）
        self.weight_history.append(self.weights.copy())
        self.error_history.append(e_n)

        return e_n

    def reset(self):
        self.weights = np.zeros(self.filter_length)
        self.x_buffer = np.zeros(self.filter_length)
        self.weight_history = []
        self.error_history = []

    def process(self, x, d):
        min_len = min(len(x), len(d))
        x = x[:min_len]
        d = d[:min_len]
        e = np.zeros(min_len)
        for n in range(min_len):
            e[n] = self.update(x[n], d[n])
        return e