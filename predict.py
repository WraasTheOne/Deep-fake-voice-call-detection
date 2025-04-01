# predict.py
import os
import argparse
import tensorflow as tf
import numpy as np
import time
import json

# Import project modules
import config # Default configurations
from utils.audio_processing import load_audio_file, extract_mel_spectrogram, normalize_spectrogram
from utils.model_utils import get_model_paths, load_model_metadata, CUSTOM_OBJECTS # Import CUSTOM_OBJECTS dict

def predict(args):
    """Loads a trained model and predicts on files in the test directory."""

    # --- 1. Determine Model and Metadata Paths ---
    if args.model_path:
        model_path = args.model_path
        # Try to infer iteration/version and find metadata path
        base_name = os.path.basename(model_path).replace(".keras", "")
        metadata_path = os.path.join(os.path.dirname(model_path), base_name + "_meta.json")
    elif args.iteration is not None and args.version is not None:
        model_path, metadata_path = get_model_paths(args.load_dir, args.iteration, args.version)
    else:
        raise ValueError("Must provide either --model_path or both --iteration and --version.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # --- 2. Load Model Metadata (Crucial for `maxlen`) ---
    print(f"Loading metadata from: {metadata_path}")
    metadata = load_model_metadata(metadata_path)
    if metadata is None:
        print("Warning: Could not load metadata. Attempting to proceed without it, but preprocessing might be inconsistent.")
        # Try to get maxlen from model input shape later, but it's risky
        loaded_maxlen = None
        # Use defaults or assume values for other metadata if needed
        trim_db = args.trim_db if args.trim_db is not None else config.DEFAULT_TRIM_DB
        normalize = args.normalize if args.normalize is not None else config.DEFAULT_NORMALIZE
        labels_inv = {v: k for k, v in config.LABELS.items()} # Default labels

    else:
        loaded_maxlen = metadata.get('maxlen')
        if loaded_maxlen is None:
            raise ValueError("Metadata file loaded, but 'maxlen' key is missing.")
        # Use metadata values for consistent preprocessing, allowing overrides from args
        trim_db = args.trim_db if args.trim_db is not None else metadata.get('trim_db', config.DEFAULT_TRIM_DB)
        normalize = args.normalize if args.normalize is not None else metadata.get('normalize', config.DEFAULT_NORMALIZE)
        labels_inv = {v: k for k, v in metadata.get('labels', config.LABELS).items()} # Invert labels map

        print(f"  Loaded maxlen: {loaded_maxlen}")
        print(f"  Using trim_db: {trim_db} (Override: {'Yes' if args.trim_db is not None else 'No'})")
        print(f"  Using normalize: {normalize} (Override: {'Yes' if args.normalize is not None else 'No'})")


    # --- 3. Load Model ---
    print(f"\nLoading model from: {model_path}")
    # IMPORTANT: Provide custom objects when loading
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    print("Model loaded successfully.")
    model.summary() # Optional: print model summary

    # Double-check maxlen if metadata wasn't available
    if loaded_maxlen is None:
        try:
            loaded_maxlen = model.input_shape[1]
            if loaded_maxlen is None: # Handle cases where input shape isn't fully defined
                 raise ValueError("Model input shape is not fixed, cannot infer maxlen.")
            print(f"Inferred maxlen from model input shape: {loaded_maxlen}")
        except Exception as e:
             raise ValueError(f"Could not determine maxlen from metadata or model shape. Cannot proceed. Error: {e}")


    # --- 4. Find Test Files ---
    print(f"\nLooking for test files in: {args.test_dir}")
    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    test_files = []
    # Look inside 'real' and 'fake' subdirectories if they exist
    has_subdirs = any(os.path.isdir(os.path.join(args.test_dir, item)) for item in os.listdir(args.test_dir))

    if has_subdirs:
        print("Found subdirectories, looking for audio files within 'real' and 'fake'...")
        for subdir in os.listdir(args.test_dir):
            subdir_path = os.path.join(args.test_dir, subdir)
            if os.path.isdir(subdir_path):
                 for fname in os.listdir(subdir_path):
                     if fname.lower().endswith((".wav", ".flac", ".mp3")):
                         # Store relative path from test_dir for cleaner output
                         test_files.append(os.path.join(subdir, fname))
    else:
        print("No subdirectories found, looking for audio files directly in test_dir...")
        for fname in os.listdir(args.test_dir):
            if fname.lower().endswith((".wav", ".flac", ".mp3")):
                test_files.append(fname) # Store only filename

    test_files.sort() # Sort for consistent order

    if not test_files:
        print("No audio files (.wav, .flac, .mp3) found in the test directory.")
        return

    print(f"Found {len(test_files)} test files.")

    # --- 5. Predict on Test Files ---
    results = []
    num_batches = (len(test_files) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    for i in range(num_batches):
        batch_fpaths_relative = test_files[i * args.batch_size : (i + 1) * args.batch_size]
        batch_data = []
        valid_fpaths_in_batch = [] # Keep track of files successfully processed in this batch

        print(f"\nProcessing batch {i+1}/{num_batches}...")
        for fpath_relative in batch_fpaths_relative:
            # Construct full path
            full_path = os.path.join(args.test_dir, fpath_relative)
            print(f"  Processing: {fpath_relative}")

            # Load and preprocess using settings consistent with training (from metadata/args)
            signal = load_audio_file(full_path, trim_db=trim_db)
            if signal is None: continue # Skip if loading failed

            mel = extract_mel_spectrogram(signal)
            if mel is None: continue # Skip if spectrogram failed

            if normalize:
                mel = normalize_spectrogram(mel)
                if mel is None: continue # Skip if normalization failed

            if mel.shape[0] == 0:
                 print(f"    Skipping due to empty spectrogram after processing.")
                 continue

            # Pad or truncate to the maxlen used during training
            if mel.shape[0] > loaded_maxlen:
                mel = mel[:loaded_maxlen, :] # Truncate
            else:
                # Pad at the end
                mel = np.pad(mel, ((0, loaded_maxlen - mel.shape[0]), (0, 0)), mode='constant', constant_values=0.0)

            batch_data.append(mel)
            valid_fpaths_in_batch.append(fpath_relative) # Store the relative path

        if not batch_data:
            print("  No valid data processed in this batch.")
            continue

        # Predict on the batch
        try:
            batch_data_np = np.array(batch_data)
            # Ensure batch data has the expected shape (batch, maxlen, features)
            if batch_data_np.ndim != 3 or batch_data_np.shape[1] != loaded_maxlen or batch_data_np.shape[2] != model.input_shape[2]:
                 print(f"Error: Batch data shape mismatch. Expected (~, {loaded_maxlen}, {model.input_shape[2]}), Got {batch_data_np.shape}")
                 continue # Skip this batch

            predictions = model.predict(batch_data_np, batch_size=args.batch_size, verbose=0) # Use specified batch_size

            # Store results for this batch
            for fpath_relative, pred in zip(valid_fpaths_in_batch, predictions):
                prob = float(pred[0]) # Probability of being fake (class 1)
                pred_label_idx = 1 if prob > 0.5 else 0
                pred_label_name = labels_inv.get(pred_label_idx, f"Class_{pred_label_idx}") # Get 'real' or 'fake'
                certainty = prob if pred_label_idx == 1 else 1 - prob
                results.append((fpath_relative, pred_label_name, certainty))
                # Print immediately for feedback
                print(f"    -> {fpath_relative}: {pred_label_name} (Confidence: {certainty:.4f}, Raw score: {prob:.4f})")

        except Exception as e:
            print(f"Error during prediction for batch {i+1}: {e}")
            import traceback
            traceback.print_exc()
            # Optionally continue to next batch or stop

    end_time = time.time()
    print(f"\nFinished predictions in {end_time - start_time:.2f} seconds.")

    # --- 6. Print Summary ---
    print("\n--- Prediction Summary ---")
    if results:
        # Sort results by filename (relative path)
        results.sort(key=lambda item: item[0])
        for fpath_relative, label, certainty in results:
             print(f"{fpath_relative}: {label} (Confidence: {certainty:.4f})")

        # Optional: Save results to a CSV or JSON file
        # output_file = os.path.join(args.test_dir, "predictions.csv")
        # try:
        #     with open(output_file, 'w', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["filepath", "predicted_label", "confidence"])
        #         for row in results:
        #             writer.writerow(row)
        #     print(f"\nPredictions saved to {output_file}")
        # except Exception as e:
        #     print(f"\nError saving predictions to file: {e}")

    else:
        print("No predictions were made (check for errors during processing).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction using a trained Conformer model.")

    # --- Model Loading ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_path", type=str, help="Direct path to the trained model (.keras file).")
    group.add_argument("--load_dir", type=str, default=config.DEFAULT_MODEL_SAVE_DIR, help="Directory containing saved models (used with --iteration and --version).")

    parser.add_argument("--iteration", type=int, help="Model iteration number (required if --model_path not used).")
    parser.add_argument("--version", type=int, help="Model version number (required if --model_path not used).")

    # --- Data and Prediction ---
    parser.add_argument("--test_dir", type=str, default=config.DEFAULT_TEST_DIR, help="Path to the testing data directory.")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size for prediction.")

    # --- Preprocessing Overrides (Optional) ---
    # Allow overriding preprocessing settings detected from metadata, use with caution.
    parser.add_argument("--trim_db", type=int, default=None, # Default to None (use metadata value)
                        help="Override silence trimming dB (0 or None to disable). Uses metadata value by default.")
    parser.add_argument("--normalize", dest='normalize', action='store_true', default=None, # Default to None (use metadata)
                         help="Override: Force enable spectrogram normalization.")
    parser.add_argument("--no_normalize", dest='normalize', action='store_false',
                         help="Override: Force disable spectrogram normalization.")


    args = parser.parse_args()

     # Handle --model_path vs --iteration/--version logic more explicitly if needed
    if args.model_path and (args.iteration is not None or args.version is not None):
         print("Warning: --model_path provided, ignoring --iteration and --version.")
         # Clear iteration/version if model_path is primary
         args.iteration = None
         args.version = None
    elif not args.model_path:
         if args.iteration is None or args.version is None:
              parser.error("If --model_path is not provided, both --iteration and --version are required.")
         # Ensure load_dir exists if using iteration/version
         if not os.path.isdir(args.load_dir):
              parser.error(f"Load directory specified (--load_dir {args.load_dir}) not found.")


    # Convert trim_db=0 to None if necessary
    if args.trim_db is not None and args.trim_db <= 0:
        args.trim_db = None


    # Run prediction
    try:
        predict(args)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check the model path or data directory (--test_dir).")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check arguments, metadata file, or model compatibility.")
    except Exception as e: # Catch other potential errors
        print(f"\nAn unexpected error occurred during prediction: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging