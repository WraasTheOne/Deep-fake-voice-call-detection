# utils/audio_processing.py
import librosa
import numpy as np
from config import SAMPLE_RATE, NUM_MELS, FFT_N, HOP_LENGTH, FMAX

def trim_silence(signal, top_db=30):
    """Trims leading and trailing silence from an audio signal."""
    if top_db is None or top_db <= 0:
        return signal # Skip trimming if top_db is not valid
    trimmed_signal, _ = librosa.effects.trim(signal, top_db=top_db)
    # Handle cases where trimming removes the entire signal
    if len(trimmed_signal) == 0:
        return signal # Return original if trimming resulted in empty signal
    return trimmed_signal

def load_audio_file(file_path, target_sr=SAMPLE_RATE, trim_db=30):
    """Loads audio, trims silence (optional), and resamples."""
    try:
        signal, sr = librosa.load(file_path, sr=None) # Load native sample rate first

        # Trim silence before resampling
        if trim_db is not None:
            signal = trim_silence(signal, top_db=trim_db)

        if len(signal) == 0:
             print(f"Warning: Signal became empty after trimming for {file_path}. Returning None.")
             return None

        # Resample if necessary
        if sr != target_sr:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)

        # Final check for empty signal after potential resampling issues
        if len(signal) == 0:
            print(f"Warning: Signal is empty for {file_path} after processing. Returning None.")
            return None

        return signal
    except Exception as e:
        print(f"Error loading or processing {file_path}: {e}")
        return None # Return None if loading or processing fails

def extract_mel_spectrogram(signal, sr=SAMPLE_RATE, n_mels=NUM_MELS, n_fft=FFT_N, hop_length=HOP_LENGTH, fmax=FMAX):
    """Extracts a Log Mel spectrogram from an audio signal."""
    if signal is None or len(signal) == 0:
        return None # Cannot process None or empty signal

    try:
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax
        )
        # Use reference power from the spectrogram itself for numerical stability
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec.T  # shape: (time, mel)
    except Exception as e:
        print(f"Error extracting Mel spectrogram: {e}")
        return None

def normalize_spectrogram(mel_spec, eps=1e-8):
    """Applies Z-score normalization to each frequency band (Mel bin)."""
    if mel_spec is None:
        return None
    try:
        mean = np.mean(mel_spec, axis=0)
        std = np.std(mel_spec, axis=0)
        # Add epsilon to std deviation denominator for numerical stability
        normalized_spec = (mel_spec - mean) / (std + eps)
        return normalized_spec
    except Exception as e:
        print(f"Error normalizing spectrogram: {e}")
        return None