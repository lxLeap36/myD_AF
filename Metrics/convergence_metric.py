import numpy as np


def _to_1d_float32(x):
    """
    将输入转成 1 维 float32 numpy 数组
    """
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError("Input must be 1-D after squeeze().")
    return x


def compute_error_curve(error):
    """
    返回瞬时误差曲线 e(n)

    说明：
    - 对于 LMS / NLMS / RLS，这里的输入通常就是 algo.process(x, d) 的返回值
    - 这条曲线一般比较抖，不适合直接判断收敛趋势，但适合做原始观察
    """
    error = _to_1d_float32(error)
    return error.copy()


def compute_squared_error_curve(error):
    """
    返回平方误差曲线 e^2(n)

    说明：
    - 这是最基础的学习曲线原始量
    - 但逐点平方误差仍然可能比较抖，所以一般还会再做平滑
    """
    error = _to_1d_float32(error)
    return (error ** 2).astype(np.float32)


def moving_average(x, window_size=512):
    """
    对输入曲线做滑动平均

    参数：
        x: 1D 曲线
        window_size: 滑动窗口长度

    返回：
        平滑后的曲线，长度与输入相同（使用 same 模式）

    说明：
    - 用于把 e^2(n) 平滑成更容易观察的 learning curve
    - window_size 越大，曲线越平滑，但瞬态细节会被抹掉更多
    """
    x = _to_1d_float32(x)

    if window_size <= 1:
        return x.copy()

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    y = np.convolve(x, kernel, mode="same")
    return y.astype(np.float32)


def compute_learning_curve(error, window_size=512):
    """
    计算平滑后的 learning curve（线性刻度）

    流程：
        e(n) -> e^2(n) -> moving average

    返回：
        平滑后的均方误差曲线
    """
    se = compute_squared_error_curve(error)
    lc = moving_average(se, window_size=window_size)
    return lc.astype(np.float32)


def compute_learning_curve_db(error, window_size=512, eps=1e-8):
    """
    计算 dB 形式的 learning curve

    定义：
        LC_dB(n) = 10 * log10( moving_average(e^2(n)) + eps )

    参数：
        error: 误差信号
        window_size: 滑动平均窗口
        eps: 防止 log(0)

    返回：
        dB 标度的 learning curve

    说明：
    - 这是最适合画“收敛曲线”的版本
    - 对比 LMS / NLMS / RLS 时，通常看这条最直观
    """
    lc = compute_learning_curve(error, window_size=window_size)
    lc_db = 10.0 * np.log10(lc + eps)
    return lc_db.astype(np.float32)


def compute_windowed_mse(error, frame_size=512, hop_size=None):
    """
    逐帧计算 MSE 曲线

    参数：
        error: 误差信号 e(n)
        frame_size: 帧长
        hop_size: 帧移；如果为 None，则等于 frame_size（无重叠）

    返回：
        每一帧的 MSE 值，长度为 num_frames

    说明：
    - 这个版本和 ERLE curve 的风格更接近
    - 如果你想统一“按帧看性能随时间变化”，这个函数很实用
    """
    error = _to_1d_float32(error)

    if hop_size is None:
        hop_size = frame_size

    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive integers.")

    if len(error) < frame_size:
        raise ValueError("Signal length must be >= frame_size.")

    num_frames = 1 + (len(error) - frame_size) // hop_size
    mse_list = []

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = error[start:end]
        mse = np.mean(frame ** 2)
        mse_list.append(mse)

    return np.asarray(mse_list, dtype=np.float32)


def compute_windowed_mse_db(error, frame_size=512, hop_size=None, eps=1e-8):
    """
    逐帧计算 dB 形式的 MSE 曲线
    """
    mse = compute_windowed_mse(error, frame_size=frame_size, hop_size=hop_size)
    return (10.0 * np.log10(mse + eps)).astype(np.float32)


def compute_ensemble_learning_curve(error_list, window_size=512):
    """
    对多次独立实验的误差曲线做 ensemble-average learning curve

    参数：
        error_list: list，每个元素是一条误差曲线 e(n)

    返回：
        多次实验平均后的 learning curve（线性刻度）

    说明：
    - 这个函数是为以后 Monte Carlo 多次实验准备的
    - 当前你做单次实验可以先不用
    """
    if len(error_list) == 0:
        raise ValueError("error_list must not be empty.")

    curves = [compute_learning_curve(err, window_size=window_size) for err in error_list]
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]

    mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
    return mean_curve.astype(np.float32)


def compute_ensemble_learning_curve_db(error_list, window_size=512, eps=1e-8):
    """
    对多次独立实验的 learning curve 做平均后再转 dB
    """
    mean_curve = compute_ensemble_learning_curve(error_list, window_size=window_size)
    return (10.0 * np.log10(mean_curve + eps)).astype(np.float32)


if __name__ == "__main__":
    # 简单测试
    rng = np.random.default_rng(42)
    e = rng.normal(0, 1, 16000).astype(np.float32)

    err_curve = compute_error_curve(e)
    se_curve = compute_squared_error_curve(e)
    lc = compute_learning_curve(e, window_size=400)
    lc_db = compute_learning_curve_db(e, window_size=400)
    mse_db = compute_windowed_mse_db(e, frame_size=400, hop_size=160)

    print("error curve shape:", err_curve.shape)
    print("squared error curve shape:", se_curve.shape)
    print("learning curve shape:", lc.shape)
    print("learning curve db shape:", lc_db.shape)
    print("windowed mse db shape:", mse_db.shape)