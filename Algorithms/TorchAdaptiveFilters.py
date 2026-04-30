# -*- coding: utf-8 -*-
"""
PyTorch implementations of LMS / NLMS / RLS.

目的：
    给 V2-4A 平台提供 PyTorch 版本传统自适应滤波器，
    便于和 DL_HYBRID 在同一套 PyTorch/CUDA 后端下测试推理时间。

注意：
    这些算法本质上仍然是逐采样点递推算法，存在严格的时序依赖。
    即使用 PyTorch/CUDA 实现，也不像深度模型那样容易整段并行。
    所以它们的测速结果代表“当前 PyTorch 实现速度”，不是理论极限速度。
"""

import numpy as np
import torch


def _to_torch_1d(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        x = x.detach().to(device=device, dtype=dtype).squeeze()
    else:
        x = torch.as_tensor(np.asarray(x), device=device, dtype=dtype).squeeze()

    if x.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got shape {tuple(x.shape)}")

    return x


class _BaseTorchAdaptiveFilter:
    def __init__(self, filter_length=128, device="cuda", dtype=torch.float32):
        self.filter_length = int(filter_length)
        self.device_name = device
        self.dtype = dtype

        if self.device_name == "cuda" and not torch.cuda.is_available():
            print(f"[{self.__class__.__name__}] CUDA requested but not available. Falling back to CPU.")
            self.device_name = "cpu"

        self.device = torch.device(self.device_name)

        # 兼容平台字段；默认不记录历史，避免长语音 OOM。
        self.weight_history = []
        self.error_history = []

    @property
    def weights(self):
        """
        为了兼容 run_basic.py / evaluator_basic.py，
        暴露为 CPU numpy array。
        """
        return self._weights.detach().cpu().numpy().astype(np.float32)

    @property
    def x_buffer(self):
        if hasattr(self, "_x_buffer"):
            return self._x_buffer.detach().cpu().numpy().astype(np.float32)
        return None

    @property
    def input_buffer(self):
        if hasattr(self, "_x_buffer"):
            return self._x_buffer.detach().cpu().numpy().astype(np.float32)
        return None

    def _init_common_state(self):
        self._weights = torch.zeros(
            self.filter_length,
            device=self.device,
            dtype=self.dtype,
        )
        self._x_buffer = torch.zeros(
            self.filter_length,
            device=self.device,
            dtype=self.dtype,
        )
        self.weight_history = []
        self.error_history = []

    def reset(self):
        self._init_common_state()


class TorchLMSFilter(_BaseTorchAdaptiveFilter):
    def __init__(self, filter_length=128, step_size=0.5, device="cuda", dtype=torch.float32):
        super().__init__(filter_length=filter_length, device=device, dtype=dtype)
        self.mu = float(step_size)
        self.reset()

    def process(self, x, d):
        x = _to_torch_1d(x, self.device, self.dtype)
        d = _to_torch_1d(d, self.device, self.dtype)

        n_samples = min(x.numel(), d.numel())
        x = x[:n_samples]
        d = d[:n_samples]

        e = torch.empty(n_samples, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for n in range(n_samples):
                # 更新输入缓冲区：[x(n), x(n-1), ...]
                self._x_buffer = torch.roll(self._x_buffer, shifts=1, dims=0)
                self._x_buffer[0] = x[n]

                y_n = torch.dot(self._weights, self._x_buffer)
                e_n = d[n] - y_n

                self._weights = self._weights + self.mu * e_n * self._x_buffer
                e[n] = e_n

        return e.detach().cpu().numpy().astype(np.float32)


class TorchNLMSFilter(_BaseTorchAdaptiveFilter):
    def __init__(self, filter_length=128, step_size=0.8, epsilon=1e-1, device="cuda", dtype=torch.float32):
        super().__init__(filter_length=filter_length, device=device, dtype=dtype)
        self.step_size = float(step_size)
        self.epsilon = float(epsilon)
        self.reset()

    def process(self, x, d):
        x = _to_torch_1d(x, self.device, self.dtype)
        d = _to_torch_1d(d, self.device, self.dtype)

        n_samples = min(x.numel(), d.numel())
        x = x[:n_samples]
        d = d[:n_samples]

        e = torch.empty(n_samples, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for n in range(n_samples):
                self._x_buffer = torch.roll(self._x_buffer, shifts=1, dims=0)
                self._x_buffer[0] = x[n]

                y_n = torch.dot(self._weights, self._x_buffer)
                e_n = d[n] - y_n

                norm_factor = torch.dot(self._x_buffer, self._x_buffer) + self.epsilon
                self._weights = self._weights + self.step_size * e_n * self._x_buffer / norm_factor

                e[n] = e_n

        return e.detach().cpu().numpy().astype(np.float32)


class TorchRLSFilter(_BaseTorchAdaptiveFilter):
    def __init__(self, filter_length=128, lambda_=0.98, delta=0.1, device="cuda", dtype=torch.float32):
        super().__init__(filter_length=filter_length, device=device, dtype=dtype)
        self.lambda_ = float(lambda_)
        self.delta = float(delta)
        self.reset()

    @property
    def P(self):
        return self._P.detach().cpu().numpy().astype(np.float32)

    def reset(self):
        self._init_common_state()
        self._P = torch.eye(
            self.filter_length,
            device=self.device,
            dtype=self.dtype,
        ) * self.delta

    def process(self, x, d):
        x = _to_torch_1d(x, self.device, self.dtype)
        d = _to_torch_1d(d, self.device, self.dtype)

        n_samples = min(x.numel(), d.numel())
        x = x[:n_samples]
        d = d[:n_samples]

        e = torch.empty(n_samples, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for n in range(n_samples):
                self._x_buffer = torch.roll(self._x_buffer, shifts=1, dims=0)
                self._x_buffer[0] = x[n]

                y_n = torch.dot(self._weights, self._x_buffer)
                e_n = d[n] - y_n

                p_x = torch.mv(self._P, self._x_buffer)
                denom = self.lambda_ + torch.dot(self._x_buffer, p_x)
                k_n = p_x / denom

                self._weights = self._weights + k_n * e_n

                # P = (P - k x^T P) / lambda
                # 因为 p_x = P x，所以 x^T P = p_x^T，P 对称时成立。
                self._P = (self._P - torch.outer(k_n, p_x)) / self.lambda_

                e[n] = e_n

        return e.detach().cpu().numpy().astype(np.float32)