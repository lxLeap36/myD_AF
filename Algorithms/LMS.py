import numpy as np

class LMSFilter:
    def __init__(self, filter_length=128, step_size=0.5):
        self.filter_length = filter_length
        self.mu = step_size
        self.weights = np.zeros(filter_length)  # 滤波器权重系数
        self.input_buffer = np.zeros(filter_length)  # 输入信号缓冲区
        self.weight_history = []  # 权重历史记录
        self.error_history = []  # 误差历史记录

    def update(self, x_n, d_n):
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = x_n
        y_n = np.dot(self.weights, self.input_buffer)
        e_n = d_n - y_n
        self.weights = self.weights + self.mu * e_n * self.input_buffer
        self.weight_history.append(self.weights.copy())
        self.error_history.append(e_n)
        return e_n

    def process(self, x, d):
        min_len = min(len(x), len(d))
        x = x[:min_len]
        d = d[:min_len]
        e = np.zeros(min_len)
        for n in range(min_len):
            e[n] = self.update(x[n], d[n])
        return e

    def reset(self):
        self.weights = np.zeros(self.filter_length)
        self.input_buffer = np.zeros(self.filter_length)
        self.weight_history = []
        self.error_history = []