import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def normalize_audio(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-12:
        return x.copy()
    return (peak / m) * x


def generate_white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


# AR(1): x[n] = a1 * x[n-1] + w[n]
def generate_ar1(n: int, a1: float, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float32)
    x = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        x[i] = a1 * x[i - 1] + w[i]
    return x


# AR(2): x[n] = a1 * x[n-1] + a2 * x[n-2] + w[n]
def generate_ar2(n: int, a1: float, a2: float, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float32)
    x = np.zeros(n, dtype=np.float32)
    for i in range(2, n):
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + w[i]
    return x


# A very simple, causal, stable FIR echo path / synthetic RIR
# You can replace this with any short vector you want to identify.
def make_easy_rir(kind: str = "short") -> np.ndarray:
    if kind == "short":
        h = np.array([0.9, 0.5, -0.25, 0.12, 0.05], dtype=np.float32)
    elif kind == "exp":
        h = np.array([0.9, 0.65, 0.46, 0.33, 0.24, 0.17, 0.12, 0.08], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported rir kind: {kind}")
    return h


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise = noise / (np.std(noise) + 1e-12)
    noise = noise * np.sqrt(noise_power)
    return x + noise.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate easy synthetic sequences for AEC / adaptive filter verification.")
    parser.add_argument("--outdir", type=str, default="./easy_aec_data", help="Output directory")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--signal", type=str, default="ar2", choices=["white", "ar1", "ar2"], help="Reference signal type")
    parser.add_argument("--a1", type=float, default=0.8, help="AR coefficient a1")
    parser.add_argument("--a2", type=float, default=-0.3, help="AR coefficient a2 (for ar2)")
    parser.add_argument("--rir", type=str, default="exp", choices=["short", "exp"], help="Synthetic RIR type")
    parser.add_argument("--snr", type=float, default=None, help="Add white noise to microphone at this SNR in dB")
    parser.add_argument("--add_nearend_tone", action="store_true", help="Add a simple near-end tone to simulate double-talk")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = int(args.sr * args.duration)

    if args.signal == "white":
        x = generate_white(n, rng)
    elif args.signal == "ar1":
        x = generate_ar1(n, args.a1, rng)
    else:
        x = generate_ar2(n, args.a1, args.a2, rng)

    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-12)

    h = make_easy_rir(args.rir)
    echo = np.convolve(x, h, mode="full")[:n].astype(np.float32)

    near_end = np.zeros(n, dtype=np.float32)
    if args.add_nearend_tone:
        t = np.arange(n, dtype=np.float32) / float(args.sr)
        near_end = 0.15 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    mic = echo + near_end
    if args.snr is not None:
        mic = add_noise(mic, args.snr, rng)

    x_wav = normalize_audio(x)
    echo_wav = normalize_audio(echo)
    near_end_wav = normalize_audio(near_end) if np.max(np.abs(near_end)) > 0 else near_end
    mic_wav = normalize_audio(mic)

    # Save wav files
    sf.write(outdir / "far_end_reference.wav", x_wav, args.sr)
    sf.write(outdir / "synthetic_rir.wav", h, args.sr)
    sf.write(outdir / "echo_only.wav", echo_wav, args.sr)
    sf.write(outdir / "near_end.wav", near_end_wav, args.sr)
    sf.write(outdir / "mic_signal.wav", mic_wav, args.sr)

    # Save raw arrays too, easier for exact verification
    np.savez(
        outdir / "easy_aec_data.npz",
        sr=args.sr,
        far_end=x.astype(np.float32),
        rir=h.astype(np.float32),
        echo=echo.astype(np.float32),
        near_end=near_end.astype(np.float32),
        mic=mic.astype(np.float32),
        signal_type=args.signal,
        a1=args.a1,
        a2=args.a2,
        rir_type=args.rir,
        snr=args.snr,
    )

    readme = f"""Easy synthetic AEC data
======================

Reference signal type: {args.signal}
Sample rate: {args.sr}
Duration: {args.duration} s
RIR type: {args.rir}
AR coefficients: a1={args.a1}, a2={args.a2}
SNR(dB): {args.snr}
Near-end tone added: {args.add_nearend_tone}

Files:
- far_end_reference.wav : reference / far-end signal x[n]
- synthetic_rir.wav     : short synthetic FIR echo path h[n]
- echo_only.wav         : echo = x[n] * h[n]
- near_end.wav          : optional near-end component
- mic_signal.wav        : microphone signal = echo + near_end (+ noise)
- easy_aec_data.npz     : raw numpy arrays for exact debugging

Recommended use:
1. Verify your LMS/NLMS/RLS code first with a SHORT known FIR h[n].
2. Use white input first, then AR(1), then AR(2).
3. Start with no near-end, no noise.
4. After the code is correct, add noise / near-end / longer FIR.
"""
    (outdir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"Saved files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
