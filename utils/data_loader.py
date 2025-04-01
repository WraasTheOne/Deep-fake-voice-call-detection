# utils/data_loader.py
import os
import numpy as np
import time
from .audio_processing import load_audio_file, extract_mel_spectrogram, normalize_spectrogram
from config import LABELS

def load_and_preprocess_data(data_dir, labels_dict=LABELS, maxlen=None, trim_db=30, normalize=True, is_training=False):
    """
    Loads audio files from a directory structured with class subfolders,
    preprocesses them (load, trim, spectrogram, normalize), and pads/truncates.

    Args:
        data_dir (str): Path to the data directory (e.g., training, validation, testing).
        labels_dict (dict): Dictionary mapping class folder names to integer labels.
        maxlen (int, optional): The target sequence length. If None and is_training is True,
                                it's calculated from the data. If None and is_training is False,
                                an error is raised. Required for validation/testing.
        trim_db (int, optional): Top dB for silence trimming. Use None to disable.
        normalize (bool): Whether to apply Z-score normalization to spectrograms.
        is_training (bool): If True, calculate maxlen if not provided. If False, maxlen is required.

    Returns:
        tuple: (X_padded, y, calculated_maxlen)
               X_padded (np.ndarray): The preprocessed and padded data.
               y (np.ndarray): The corresponding labels.
               calculated_maxlen (int): The maximum sequence length used for padding.
                                        (returns the input maxlen if provided).
    """
    X, y = [], []
    print(f"Loading data from {data_dir}...")
    start_time = time.time()
    processed_count = 0
    error_count = 0
    skipped_empty_count = 0

    all_lengths = [] # To determine maxlen if needed

    if not os.path.isdir(data_dir):
         raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for label_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(class_dir) or label_name not in labels_dict:
            continue # Skip non-directory items or folders not in LABELS

        label = labels_dict[label_name]
        print(f"  Processing class: {label_name}")
        files_in_class = [f for f in os.listdir(class_dir) if f.lower().endswith((".wav", ".flac", ".mp3"))] # Support more formats if needed
        print(f"    Found {len(files_in_class)} audio files.")

        for fname in files_in_class:
            path = os.path.join(class_dir, fname)
            signal = load_audio_file(path, trim_db=trim_db)

            if signal is None:
                # print(f"    Skipping due to loading error: {fname}")
                error_count += 1
                continue

            mel = extract_mel_spectrogram(signal)

            if mel is None:
                # print(f"    Skipping due to spectrogram error: {fname}")
                error_count += 1
                continue

            if normalize:
                mel = normalize_spectrogram(mel)
                if mel is None:
                    # print(f"    Skipping due to normalization error: {fname}")
                    error_count += 1
                    continue

            if mel.shape[0] > 0: # Check if mel spectrogram is not empty
                X.append(mel)
                y.append(label)
                all_lengths.append(mel.shape[0])
                processed_count += 1
            else:
                 print(f"    Skipping file with empty spectrogram after processing: {fname}")
                 skipped_empty_count += 1

    end_time = time.time()
    print(f"Finished loading from {data_dir}.")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped (empty/error): {error_count + skipped_empty_count} files")
    print(f"  Time taken: {end_time - start_time:.2f} seconds.")

    if not X:
        # Raise a more specific error or return empty arrays depending on desired behavior
        raise ValueError(f"No valid data loaded from {data_dir}. Check directory structure, file formats, and processing steps.")

    # Determine maxlen
    if maxlen is None:
        if is_training:
            calculated_maxlen = max(all_lengths) if all_lengths else 0
            print(f"Determined max sequence length from training data: {calculated_maxlen}")
        else:
            raise ValueError("maxlen must be provided when loading validation or testing data (is_training=False).")
    else:
        calculated_maxlen = maxlen
        if is_training:
             print(f"Using provided max sequence length: {calculated_maxlen}")


    if calculated_maxlen <= 0 and processed_count > 0 :
         print(f"Warning: Calculated maxlen is {calculated_maxlen}, but {processed_count} files were processed. Check data.")
         # Decide how to handle this - raise error or set a minimum length?
         # For now, let padding handle it if calculated_maxlen is 0, though it's weird.

    # Pad sequences
    # Using 'constant' padding with 0.0. Consider 'edge' or other modes if beneficial.
    try:
        X_padded = np.array([np.pad(x, ((0, calculated_maxlen - x.shape[0]), (0, 0)), mode='constant', constant_values=0.0)
                            if x.shape[0] <= calculated_maxlen
                            else x[:calculated_maxlen, :] # Truncate if longer
                            for x in X])
    except ValueError as e:
         print("\nError during padding. Spectrogram shapes:")
         for i, x in enumerate(X):
             print(f"  Index {i}: {x.shape}")
         raise ValueError(f"Error padding sequences. Check spectrogram shapes and calculated maxlen ({calculated_maxlen}). Original error: {e}")


    return X_padded, np.array(y), calculated_maxlen