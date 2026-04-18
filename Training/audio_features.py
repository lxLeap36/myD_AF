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