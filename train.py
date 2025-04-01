# train.py
import os
import argparse
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks # Import optimizers and callbacks
import numpy as np

# Import project modules
import config # Default configurations
from utils.data_loader import load_and_preprocess_data
from utils.model_utils import (get_model_paths, get_next_version_number,
                               save_model_metadata, CUSTOM_OBJECTS)
from model.build_model import build_conformer_model

def train(args):
    """Loads data, builds model, trains, and saves."""

    # --- 1. Load Data ---
    print("Loading training data...")
    X_train, y_train, train_maxlen = load_and_preprocess_data(
        data_dir=args.train_dir,
        maxlen=None, # Calculate maxlen from training data
        trim_db=args.trim_db,
        normalize=args.normalize,
        is_training=True
    )
    print(f"Training data shape: {X_train.shape}, Max length: {train_maxlen}")

    print("\nLoading validation data...")
    X_val, y_val, _ = load_and_preprocess_data(
        data_dir=args.val_dir,
        maxlen=train_maxlen, # Use maxlen from training data
        trim_db=args.trim_db,
        normalize=args.normalize,
        is_training=False # maxlen must be provided
    )
    print(f"Validation data shape: {X_val.shape}")

    # Ensure data was loaded
    if X_train.size == 0 or X_val.size == 0:
        print("Error: Training or validation data is empty. Please check data directories and preprocessing steps.")
        return

    # --- 2. Build Model ---
    print("\nBuilding model...")
    # Infer input shape from training data: (maxlen, num_mels)
    # Assuming NUM_MELS is consistent (defined in config)
    input_shape = (train_maxlen, config.NUM_MELS)

    model = build_conformer_model(
        input_shape=input_shape,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ff_expansion=args.ff_expansion,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        drop_path_rate=args.drop_path,
        use_se_in_conv=args.use_se,
        use_spec_augment=args.use_specaugment,
        freq_mask_param=args.freq_mask,
        time_mask_param=args.time_mask
    )

    # --- 3. Compile Model ---
    # Consider AdamW optimizer and learning rate schedules for potentially better results
    optimizer = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary() # Print model structure

    # --- 4. Setup Callbacks and Saving ---
    # Determine version and paths
    version = get_next_version_number(args.save_dir, args.iteration)
    model_path, metadata_path = get_model_paths(args.save_dir, args.iteration, version)
    print(f"\nModel will be saved as Version {version} in {args.save_dir}")
    print(f"  Model file: {os.path.basename(model_path)}")
    print(f"  Metadata file: {os.path.basename(metadata_path)}")

    # Save metadata (including maxlen) before training starts
    metadata = {
        'maxlen': train_maxlen,
        'sample_rate': config.SAMPLE_RATE,
        'num_mels': config.NUM_MELS,
        'fft_n': config.FFT_N,
        'hop_length': config.HOP_LENGTH,
        'fmax': config.FMAX,
        'labels': config.LABELS,
        'trim_db': args.trim_db,
        'normalize': args.normalize,
        # Add model hyperparameters from args if needed for reproducibility
        'd_model': args.d_model,
        'num_blocks': args.num_blocks,
        # ... other relevant args ...
    }
    save_model_metadata(metadata_path, metadata)

    # Callbacks for training
    # Save the best model based on validation accuracy
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=model_path, # Save directly to the final path
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    # Stop training early if validation accuracy doesn't improve
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=args.patience,
        restore_best_weights=True, # Restore weights from the epoch with the best val_accuracy
        verbose=1
    )
    # Optional: Reduce learning rate on plateau
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2, # Reduce LR by a factor of 5
        patience=max(5, args.patience // 2), # Reduce LR if no improvement for half the patience
        min_lr=1e-6, # Don't reduce LR below this value
        verbose=1
    )

    # --- 5. Train Model ---
    print(f"\nStarting training (Iteration {args.iteration}, Version {version})...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb] # Add ReduceLROnPlateau
    )

    print(f"\nTraining complete. Best model weights (during training) restored.")
    print(f"Best model potentially saved at: {model_path}")

    # Optional: Evaluate the final (best restored) model on validation set
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy (best epoch): {accuracy:.4f}")

    # Note: The best model is saved by ModelCheckpoint during training.
    # EarlyStopping with restore_best_weights=True means the 'model' variable
    # here holds the weights from the best epoch after training finishes.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Conformer model for Deepfake Audio Detection.")

    # --- Directories and Model ID ---
    parser.add_argument("--train_dir", type=str, default=config.DEFAULT_TRAIN_DIR, help="Path to the training data directory.")
    parser.add_argument("--val_dir", type=str, default=config.DEFAULT_VAL_DIR, help="Path to the validation data directory.")
    parser.add_argument("--save_dir", type=str, default=config.DEFAULT_MODEL_SAVE_DIR, help="Directory to save trained models and metadata.")
    parser.add_argument("--iteration", type=int, default=1, help="Model iteration number (used for saving).")

    # --- Preprocessing ---
    parser.add_argument("--trim_db", type=int, default=config.DEFAULT_TRIM_DB, help="Top dB for silence trimming (set to 0 or None to disable).")
    parser.add_argument("--normalize", dest='normalize', action='store_true', default=config.DEFAULT_NORMALIZE, help="Enable spectrogram Z-score normalization (default).")
    parser.add_argument("--no_normalize", dest='normalize', action='store_false', help="Disable spectrogram Z-score normalization.")

    # --- Model Architecture ---
    parser.add_argument("--d_model", type=int, default=config.DEFAULT_D_MODEL, help="Internal dimension of the model.")
    parser.add_argument("--num_blocks", type=int, default=config.DEFAULT_NUM_BLOCKS, help="Number of Conformer blocks.")
    parser.add_argument("--num_heads", type=int, default=config.DEFAULT_NUM_HEADS, help="Number of attention heads.")
    parser.add_argument("--ff_expansion", type=int, default=config.DEFAULT_FF_EXPANSION, help="Expansion factor for feed-forward layers.")
    parser.add_argument("--kernel_size", type=int, default=config.DEFAULT_KERNEL_SIZE, help="Kernel size for depthwise convolution.")
    parser.add_argument("--use_se", dest='use_se', action='store_true', default=config.DEFAULT_USE_SE, help="Enable Squeeze-and-Excitation in Convolution Module (default).")
    parser.add_argument("--no_se", dest='use_se', action='store_false', help="Disable Squeeze-and-Excitation.")

    # --- Regularization & Augmentation ---
    parser.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT, help="Dropout rate.")
    parser.add_argument("--drop_path", type=float, default=config.DEFAULT_DROP_PATH, help="Stochastic depth (DropPath) max rate.")
    parser.add_argument("--use_specaugment", dest='use_specaugment', action='store_true', default=config.DEFAULT_USE_SPECAUGMENT, help="Enable SpecAugment (default).")
    parser.add_argument("--no_specaugment", dest='use_specaugment', action='store_false', help="Disable SpecAugment.")
    parser.add_argument("--freq_mask", type=int, default=config.DEFAULT_FREQ_MASK, help="Frequency masking parameter F for SpecAugment.")
    parser.add_argument("--time_mask", type=int, default=config.DEFAULT_TIME_MASK, help="Time masking parameter T for SpecAugment.")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE, help="Early stopping patience.")

    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert trim_db=0 to None if necessary for logic in audio_processing
    if args.trim_db is not None and args.trim_db <= 0:
        args.trim_db = None
        print("Silence trimming disabled (trim_db <= 0).")

    # Run training
    try:
        train(args)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check the data directory paths (--train_dir, --val_dir).")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check arguments, data integrity, or model compatibility.")
    except Exception as e: # Catch other potential errors
        print(f"\nAn unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging