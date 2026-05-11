"""
Simple reproducible white noise generator for the project.
Generates two 15-second WAV files (float32, 16 kHz) under Dataset/white_noise_test/
Usage: run this script once before running experiments that expect test audio.

The generator uses a fixed RNG seed so the produced files are reproducible.
"""
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Dataset" / "white_noise_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
DUR_SEC = 15.0
N_SAMPLES = int(SR * DUR_SEC)

# Use fixed seeds for reproducibility
SEEDS = [12345, 54321]
FILENAMES = ["white_noise_a.wav", "white_noise_b.wav"]

def generate_white_noise(seed: int, n_samples: int, sr: int):
    rng = np.random.RandomState(seed)
    # Gaussian white noise with unit variance, then scaled to -20 dB FS RMS
    noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples,)).astype(np.float32)
    # Normalize to unit RMS then scale to target RMS (in dBFS)
    rms = np.sqrt(np.mean(noise**2))
    if rms <= 0:
        return noise
    target_dbfs = -20.0
    target_rms = 10.0 ** (target_dbfs / 20.0)
    noise = noise / rms * target_rms
    return noise

if __name__ == "__main__":
    for seed, fname in zip(SEEDS, FILENAMES):
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"Skipping existing: {out_path}")
            continue
        print(f"Generating {out_path} (seed={seed})")
        wav = generate_white_noise(seed, N_SAMPLES, SR)
        sf.write(str(out_path), wav, SR, subtype='FLOAT')
    print("Done.")

