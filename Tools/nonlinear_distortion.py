import numpy as np


def apply_nonlinear_distortion(r, delta_1=4, delta_2=4):
    """
    根据陈捷老师论文实现的双级非线性失真模型 (功放软截断 + 扬声器Sigmoid失真)
    """
    r_max = np.max(np.abs(r))
    if r_max == 0:
        return r

    # 1. 功放软截断
    r_soft = (r_max * r) / np.sqrt(r_max ** 2 + r ** 2)

    # 2. 扬声器非线性
    zeta = 1.5 * r_soft - 0.3 * (r_soft ** 2)
    delta = np.where(zeta > 0, delta_1, delta_2)
    r_NL = (1.0 / (1.0 + np.exp(-delta * zeta))) - 0.5

    return r_NL