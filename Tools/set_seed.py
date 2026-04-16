import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    普通复现模式：
    - 固定 Python / NumPy / PyTorch 的随机种子
    - 不启用严格确定性算法
    - 不关闭所有性能优化

    适合：
    - 日常训练
    - 实验复现
    - 不想明显牺牲性能的情况
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)