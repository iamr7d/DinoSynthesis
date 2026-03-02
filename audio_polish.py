"""
Organic Post-Processing for VAE-synthesized audio.
Simulates physiological acoustics (laryngeal resonance, sub-harmonics, ADSR).
"""
import librosa
import numpy as np
import scipy.signal as signal


def organic_polish(y: np.ndarray, sr: int = 16000, 
                   attack_ms: float = 150, release_ms: float = 300,
                   lpf_hz: float = 3500, sub_blend: float = 0.3,
                   delay_ms: float = 40, delay_blend: float = 0.15) -> np.ndarray:
    """
    Apply physiological post-processing to synthesized audio.
    
    Args:
        y: Waveform array
        sr: Sample rate (default 16000 Hz)
        attack_ms: ADSR attack time in ms (default 150)
        release_ms: ADSR release time in ms (default 300)
        lpf_hz: Low-pass filter cutoff in Hz (default 3500, simulates soft tissue)
        sub_blend: Sub-harmonic blend amount 0-1 (default 0.3 = 30% of -12 step shift)
        delay_ms: Slapback delay in ms (default 40)
        delay_blend: Delay blend amount 0-1 (default 0.15)
    
    Returns:
        Polished waveform
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    
    # ─────────────────────────────────────────────────────
    # 1. ADSR ENVELOPE (Biological attack/release)
    # ─────────────────────────────────────────────────────
    envelope = np.ones_like(y)
    attack_len = int((attack_ms / 1000.0) * sr)
    release_len = int((release_ms / 1000.0) * sr)
    
    if attack_len > 0 and attack_len < len(y):
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
    if release_len > 0 and release_len < len(y):
        envelope[-release_len:] = np.linspace(1, 0, release_len)
    
    y = y * envelope
    
    # ─────────────────────────────────────────────────────
    # 2. THROAT FILTER (Low-pass: removes digital sizzle)
    # ─────────────────────────────────────────────────────
    if lpf_hz > 0 and lpf_hz < sr / 2:
        try:
            sos = signal.butter(10, lpf_hz, 'lp', fs=sr, output='sos')
            y = signal.sosfilt(sos, y)
        except Exception:
            pass  # Skip filter if it fails
    
    # ─────────────────────────────────────────────────────
    # 3. SUB-HARMONIC BOOST (Adds mass/weight)
    # ─────────────────────────────────────────────────────
    if sub_blend > 0:
        try:
            # Pitch shift down one octave (-12 semitones)
            y_low = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12)
            y = y + (sub_blend * y_low)
        except Exception:
            pass  # Skip if pitch shift fails
    
    # ─────────────────────────────────────────────────────
    # 4. SLAPBACK DELAY (Simulates canyon/jungle reverb)
    # ─────────────────────────────────────────────────────
    if delay_blend > 0 and delay_ms > 0:
        delay_samples = int((delay_ms / 1000.0) * sr)
        if delay_samples > 0 and delay_samples < len(y):
            y_delayed = np.zeros_like(y)
            y_delayed[delay_samples:] = y[:-delay_samples]
            y = y + (delay_blend * y_delayed)
    
    # ─────────────────────────────────────────────────────
    # 5. NORMALIZE
    # ─────────────────────────────────────────────────────
    try:
        y = librosa.util.normalize(y)
    except Exception:
        # Fallback normalization
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak
    
    return np.clip(y, -1.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    # Test: polish a downloaded WAV
    import sys
    if len(sys.argv) > 1:
        import soundfile as sf
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "polished_" + input_file
        
        y, sr = librosa.load(input_file, sr=None)
        y_polished = organic_polish(y, sr=sr)
        sf.write(output_file, y_polished, sr)
        print(f"✓ Polished: {input_file} → {output_file}")
