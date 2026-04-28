from typing import Tuple

import torch


def stft_complex(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        wav: [N] or [B, N]
    Returns:
        spec: complex tensor
            [F, T] or [B, F, T]
    """
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )


def complex_mag_phase(spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        spec: complex tensor
    Returns:
        mag: |spec|
        phase: angle(spec)
    """
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    return mag, phase


def log1p_mag(spec: torch.Tensor) -> torch.Tensor:
    """
    spec: complex tensor
    return: log1p(|spec|)
    """
    mag = torch.abs(spec)
    return torch.log1p(mag)


def mag_phase_to_complex(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mag: [..., F, T]
        phase: [..., F, T]
    Returns:
        complex spec
    """
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    return torch.complex(real, imag)


def istft_complex(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """
    Args:
        spec: [F, T] or [B, F, T], complex
    Returns:
        wav: [N] or [B, N]
    """
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )

def complex_to_ri_channels(spec: torch.Tensor) -> torch.Tensor:
    """
    Convert complex spectrogram to real/imag channels.

    Args:
        spec:
            [F, T] or [B, F, T], complex

    Returns:
        ri:
            [2, T, F] or [B, 2, T, F], float
            channel 0 = real
            channel 1 = imag
    """
    if not torch.is_complex(spec):
        raise TypeError("spec must be a complex tensor.")

    if spec.ndim == 2:
        real = spec.real.transpose(0, 1).contiguous()   # [T, F]
        imag = spec.imag.transpose(0, 1).contiguous()   # [T, F]
        return torch.stack([real, imag], dim=0)         # [2, T, F]

    if spec.ndim == 3:
        real = spec.real.transpose(1, 2).contiguous()   # [B, T, F]
        imag = spec.imag.transpose(1, 2).contiguous()   # [B, T, F]
        return torch.stack([real, imag], dim=1)         # [B, 2, T, F]

    raise ValueError(f"Unsupported spec ndim: {spec.ndim}")


def ri_channels_to_complex(ri: torch.Tensor) -> torch.Tensor:
    """
    Convert real/imag channels back to complex spectrogram.

    Args:
        ri:
            [2, T, F] or [B, 2, T, F]

    Returns:
        spec:
            [F, T] or [B, F, T], complex
    """
    if ri.ndim == 3:
        if ri.shape[0] != 2:
            raise ValueError("For 3-D input, shape must be [2, T, F].")
        real = ri[0].transpose(0, 1).contiguous()   # [F, T]
        imag = ri[1].transpose(0, 1).contiguous()   # [F, T]
        return torch.complex(real, imag)

    if ri.ndim == 4:
        if ri.shape[1] != 2:
            raise ValueError("For 4-D input, shape must be [B, 2, T, F].")
        real = ri[:, 0].transpose(1, 2).contiguous()   # [B, F, T]
        imag = ri[:, 1].transpose(1, 2).contiguous()   # [B, F, T]
        return torch.complex(real, imag)

    raise ValueError(f"Unsupported ri ndim: {ri.ndim}")


def apply_complex_mask_ri(mask_ri: torch.Tensor, d_ri: torch.Tensor) -> torch.Tensor:
    """
    Apply complex ratio mask on real/imag channels.

    Args:
        mask_ri:
            [2, T, F] or [B, 2, T, F]
            channel 0 = M_r
            channel 1 = M_i

        d_ri:
            [2, T, F] or [B, 2, T, F]
            channel 0 = D_r
            channel 1 = D_i

    Returns:
        s_hat_ri:
            same shape as input, representing:
                S_hat = M_c * D
    """
    squeeze_back = False

    if mask_ri.ndim == 3:
        mask_ri = mask_ri.unsqueeze(0)
        d_ri = d_ri.unsqueeze(0)
        squeeze_back = True

    if mask_ri.ndim != 4 or d_ri.ndim != 4:
        raise ValueError("mask_ri and d_ri must be [2,T,F] or [B,2,T,F].")

    if mask_ri.shape != d_ri.shape:
        raise ValueError(f"Shape mismatch: {mask_ri.shape} vs {d_ri.shape}")

    mr = mask_ri[:, 0]   # [B, T, F]
    mi = mask_ri[:, 1]
    dr = d_ri[:, 0]
    di = d_ri[:, 1]

    sr = mr * dr - mi * di
    si = mr * di + mi * dr

    out = torch.stack([sr, si], dim=1)   # [B, 2, T, F]

    if squeeze_back:
        out = out[0]

    return out

def apply_real_mask_to_ri(mask_mag: torch.Tensor, d_ri: torch.Tensor) -> torch.Tensor:
    """
    Apply a real-valued magnitude mask on complex spectrogram represented by RI channels.

    Args:
        mask_mag:
            [T, F] or [B, T, F]
            real-valued mask (usually nonnegative)

        d_ri:
            [2, T, F] or [B, 2, T, F]
            channel 0 = D_r
            channel 1 = D_i

    Returns:
        base_s_ri:
            same batch shape as d_ri
            [2, T, F] or [B, 2, T, F]
            where:
                S_base_r = mask_mag * D_r
                S_base_i = mask_mag * D_i
    """
    squeeze_back = False

    if mask_mag.ndim == 2:
        mask_mag = mask_mag.unsqueeze(0)   # [1, T, F]
        d_ri = d_ri.unsqueeze(0)           # [1, 2, T, F]
        squeeze_back = True

    if mask_mag.ndim != 3 or d_ri.ndim != 4:
        raise ValueError("mask_mag must be [T,F] or [B,T,F], and d_ri must be [2,T,F] or [B,2,T,F].")

    if d_ri.shape[1] != 2:
        raise ValueError("d_ri must have shape [B, 2, T, F].")

    if mask_mag.shape[0] != d_ri.shape[0] or mask_mag.shape[1] != d_ri.shape[2] or mask_mag.shape[2] != d_ri.shape[3]:
        raise ValueError(f"Shape mismatch: mask_mag {mask_mag.shape} vs d_ri {d_ri.shape}")

    dr = d_ri[:, 0]   # [B, T, F]
    di = d_ri[:, 1]   # [B, T, F]

    sr = mask_mag * dr
    si = mask_mag * di

    out = torch.stack([sr, si], dim=1)   # [B, 2, T, F]

    if squeeze_back:
        out = out[0]

    return out