import numpy as np


class KLMSFilter:
    def __init__(self, filter_length=128, step_size=0.5, kernel_param=0.1, budget=1000):
        """
        Kernel Least Mean Square (KLMS) 滤波器

        参数:
        filter_length:  与LMS对齐，作为时间延迟嵌入(Time-delay embedding)的向量长度
        step_size:      学习率 (对应公式中的 eta)
        kernel_param:   高斯核的带宽参数 (通常记为 a 或 gamma，计算: exp(-a * ||x1 - x2||^2))
        budget:         字典大小上限。如果不设限，处理几万个音频采样点时内存和计算会爆炸。
        """
        self.filter_length = filter_length
        self.mu = step_size
        self.kernel_param = kernel_param
        self.budget = budget

        self.input_buffer = np.zeros(filter_length)  # 输入信号缓冲区，即当前的 x(n)

        self.centers = []  # 字典：保存历史输入向量 x(j)
        self.coefficients = []  # 系数：保存对应的权重 a_j

        self.error_history = []  # 误差历史记录
        self.weight_history = []  # 对于KLMS，权重是一个动态增长的列表，这里记录字典大小作为历史参考

    def update(self, x_n, d_n):
        # 1. 更新输入缓冲区，形成当前的输入向量 x(n)
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = x_n

        current_x = self.input_buffer.copy()

        # 2. 滤波器输出 y(n) = sum( a_j * k(x(n), x(j)) )
        if len(self.centers) == 0:
            y_n = 0.0
        else:
            # 为了计算速度，采用 numpy 向量化广播计算核函数
            centers_mat = np.array(self.centers)
            # 计算欧氏距离平方 ||x(n) - x(j)||^2
            sq_dists = np.sum((centers_mat - current_x) ** 2, axis=1)
            # 计算高斯核 kappa
            k_vec = np.exp(-self.kernel_param * sq_dists)
            # 内积得到输出 y(n)
            y_n = np.dot(np.array(self.coefficients), k_vec)

        # 3. 计算误差 e(n) = d(n) - y(n)
        e_n = d_n - y_n

        # 4. 字典与系数更新 (向字典中追加新元素)
        self.centers.append(current_x)
        # a_n = eta * e(n)
        self.coefficients.append(self.mu * e_n)

        # 5. 字典截断 (Sliding window budget)，防止 $O(N)$ 增长
        if self.budget is not None and len(self.centers) > self.budget:
            self.centers.pop(0)  # 移除最老的输入向量
            self.coefficients.pop(0)  # 移除对应的系数

        self.error_history.append(e_n)
        self.weight_history.append(len(self.coefficients))  # 记录当前字典大小

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
        self.input_buffer = np.zeros(self.filter_length)
        self.centers = []
        self.coefficients = []
        self.error_history = []
        self.weight_history = []