import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import argparse
import time  # For timing data loading

# --- PARAMETERS ---
SAMPLE_RATE = 16000
DURATION = 2.0  # Target duration in seconds (used for MAX_LEN calculation, but trimming might change actual length before padding)
NUM_MELS = 80    # Increased common value
FFT_N = 2048     # Standard FFT size
HOP_LENGTH = 512 # Standard hop length
FMAX = 8000      # Max frequency for Mel scale
LABELS = {'real': 0, 'fake': 1}
DATA_DIR = "for-2seconds/training"
TEST_DIR = "for-2seconds/testing"
# MAX_LEN will be determined dynamically from the training data after trimming/feature extraction

# --- AUDIO PROCESSING & AUGMENTATION ---

def trim_silence(signal, top_db=30):
    """Trims leading and trailing silence from an audio signal."""
    trimmed_signal, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed_signal

def load_audio_file(file_path, target_sr=SAMPLE_RATE, trim_db=30):
    """Loads audio, trims silence, and resamples."""
    try:
        signal, sr = librosa.load(file_path, sr=None) # Load native sample rate first
        if trim_db is not None:
            signal = trim_silence(signal, top_db=trim_db)
        if sr != target_sr:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        return signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None # Return None if loading fails

def extract_mel_spectrogram(signal, sr=SAMPLE_RATE, n_mels=NUM_MELS, n_fft=FFT_N, hop_length=HOP_LENGTH, fmax=FMAX):
    """Extracts a Mel spectrogram from an audio signal."""
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) # Use np.max for better normalization reference
    return log_mel_spec.T  # shape: (time, mel)

def normalize_spectrogram(mel_spec, eps=1e-8):
    """Applies Z-score normalization to each frequency band (Mel bin)."""
    mean = np.mean(mel_spec, axis=0)
    std = np.std(mel_spec, axis=0)
    return (mel_spec - mean) / (std + eps)

# --- TensorFlow Layers ---

class SpecAugment(layers.Layer):
    """Applies SpecAugment (frequency and time masking).
    Reference: Park et al. (2019) SpecAugment: A Simple Data Augmentation Method
               for Automatic Speech Recognition (https://arxiv.org/abs/1904.08779)
    """
    def __init__(self, freq_mask_param, time_mask_param, num_freq_masks=1, num_time_masks=1, name="spec_augment", **kwargs):
        super(SpecAugment, self).__init__(name=name, **kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def call(self, inputs, training=None):
        if not training:
            return inputs

        outputs = inputs
        freq_axis = -1 # Assume shape is (batch, time, freq)
        time_axis = -2

        input_shape = tf.shape(inputs)
        num_freq_bins = input_shape[freq_axis]
        num_time_steps = input_shape[time_axis]

        # Frequency Masking
        for _ in range(self.num_freq_masks):
            f = tf.random.uniform([], minval=0, maxval=self.freq_mask_param, dtype=tf.int32)
            f0 = tf.random.uniform([], minval=0, maxval=num_freq_bins - f, dtype=tf.int32)
            mask = tf.concat(
                (tf.ones(shape=(input_shape[0], input_shape[1], f0, input_shape[3] if len(input_shape) == 4 else 1)),
                 tf.zeros(shape=(input_shape[0], input_shape[1], f, input_shape[3] if len(input_shape) == 4 else 1)),
                 tf.ones(shape=(input_shape[0], input_shape[1], num_freq_bins - f0 - f, input_shape[3] if len(input_shape) == 4 else 1)),
                 ), axis=freq_axis + 1 # Adjust axis index based on rank
            )
            outputs = outputs * mask

        # Time Masking
        for _ in range(self.num_time_masks):
            t = tf.random.uniform([], minval=0, maxval=self.time_mask_param, dtype=tf.int32)
            t0 = tf.random.uniform([], minval=0, maxval=num_time_steps - t, dtype=tf.int32)
            mask = tf.concat(
                 (tf.ones(shape=(input_shape[0], t0, input_shape[2], input_shape[3] if len(input_shape) == 4 else 1)),
                 tf.zeros(shape=(input_shape[0], t, input_shape[2], input_shape[3] if len(input_shape) == 4 else 1)),
                 tf.ones(shape=(input_shape[0], num_time_steps - t0 - t, input_shape[2], input_shape[3] if len(input_shape) == 4 else 1)),
                 ), axis=time_axis + 1 # Adjust axis index based on rank
            )
            outputs = outputs * mask

        return outputs

    def get_config(self):
        config = super(SpecAugment, self).get_config()
        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "num_freq_masks": self.num_freq_masks,
            "num_time_masks": self.num_time_masks,
        })
        return config


class PositionalEncoding(layers.Layer):
    """Adds sinusoidal positional encoding.
    Reference: Vaswani et al. (2017) Attention Is All You Need (https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        # Precompute the positional encoding matrix
        self.pos_encoding = self._build_encoding(max_len, d_model)

    def _build_encoding(self, length, depth):
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        # Ensure the positional encoding is not larger than the input sequence length
        # This assumes the input shape is (batch, seq_len, features)
        return x + self.pos_encoding[tf.newaxis, :length, :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return config

class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block."""
    def __init__(self, input_channels, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(input_channels // ratio, activation='relu')
        self.fc2 = layers.Dense(input_channels, activation='sigmoid')

    def call(self, inputs):
        se = self.pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        # Reshape se to (batch, 1, channels) to broadcast correctly
        se = tf.expand_dims(se, axis=1)
        return inputs * se

    def get_config(self):
        config = super(SEBlock, self).get_config()
        # Add relevant parameters if needed, e.g., ratio
        # config.update({"ratio": self.ratio}) # Assuming self.ratio is stored
        return config

class DropPath(layers.Layer):
    """Stochastic depth layer (DropPath).
    Randomly drops residual connections during training.
    Reference: Huang et al. (2016) Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
    """
    def __init__(self, drop_prob=0.1, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if self.drop_prob == 0. or not training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)  # work with diff ranks, shape (B, 1, 1, ...)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
        random_tensor = tf.floor(random_tensor)  # binarize
        output = tf.math.divide(x, keep_prob) * random_tensor
        return output

    def get_config(self):
        config = super(DropPath, self).get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


# --- CONFORMER COMPONENTS (Updated) ---

class FeedForwardModule(layers.Layer):
    # Keep original structure, but can add DropPath externally
    def __init__(self, d_model, expansion_factor=4, dropout=0.1, **kwargs):
        super().__init__(name="ff_module", **kwargs)
        self.norm = layers.LayerNormalization()
        self.dense1 = layers.Dense(d_model * expansion_factor, activation='swish') # Use Swish activation
        self.dropout1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(d_model)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        # Original: return x + 0.5 * self.seq(x)
        # Standard Conformer: LayerNorm -> Dense -> Act -> Dropout -> Dense -> Dropout
        x_norm = self.norm(x)
        x_ff = self.dense1(x_norm)
        x_ff = self.dropout1(x_ff)
        x_ff = self.dense2(x_ff)
        x_ff = self.dropout2(x_ff)
        return x_ff # Residual connection is handled in ConformerBlock

    def get_config(self):
        config = super(FeedForwardModule, self).get_config()
        # Store parameters if needed for loading
        # config.update({...})
        return config

class MultiHeadSelfAttentionModule(layers.Layer):
    # Keep original structure, but can add DropPath externally
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(name="mha_module", **kwargs)
        self.norm = layers.LayerNormalization()
        # Ensure key_dim * num_heads = d_model if possible, or adjust d_model
        if d_model % num_heads != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.dropout = layers.Dropout(dropout) # Dropout after MHA output

    def call(self, x):
        # Standard Conformer: LayerNorm -> MHA -> Dropout
        x_norm = self.norm(x)
        attn_output = self.mha(query=x_norm, value=x_norm, key=x_norm)
        attn_output = self.dropout(attn_output)
        return attn_output # Residual connection is handled in ConformerBlock

    def get_config(self):
        config = super(MultiHeadSelfAttentionModule, self).get_config()
        # Store parameters if needed for loading
        # config.update({...})
        return config

class ConvolutionModule(layers.Layer):
    def __init__(self, d_model, kernel_size=31, dropout=0.1, use_se=True, **kwargs):
        super().__init__(name="conv_module", **kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.use_se = use_se

        self.norm = layers.LayerNormalization()
        # Pointwise Conv -> GLU Activation
        self.pointwise_conv1 = layers.Conv1D(filters=2 * d_model, kernel_size=1, padding='same')
        # Depthwise Conv
        self.depthwise_conv = layers.DepthwiseConv1D(kernel_size=kernel_size, padding='same', depth_multiplier=1)
        self.batch_norm = layers.BatchNormalization()
        # Optional Squeeze-and-Excitation
        if self.use_se:
            self.se_block = SEBlock(d_model) # SE operates on d_model channels
        # Swish Activation
        self.activation = layers.Activation('swish')
        # Pointwise Conv
        self.pointwise_conv2 = layers.Conv1D(filters=d_model, kernel_size=1, padding='same')
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        # Standard Conformer: LayerNorm -> PointwiseConv1 -> GLU -> DepthwiseConv -> BN -> Swish -> SE(optional) -> PointwiseConv2 -> Dropout
        x_norm = self.norm(x)
        x_conv = self.pointwise_conv1(x_norm)

        # GLU Activation: Gated Linear Unit
        x_gate, x_filter = tf.split(x_conv, num_or_size_splits=2, axis=-1)
        x_conv = x_gate * tf.nn.sigmoid(x_filter)

        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.batch_norm(x_conv)
        x_conv = self.activation(x_conv)

        if self.use_se:
            x_conv = self.se_block(x_conv)

        x_conv = self.pointwise_conv2(x_conv)
        x_conv = self.dropout(x_conv)
        return x_conv # Residual connection handled in ConformerBlock

    def get_config(self):
        config = super(ConvolutionModule, self).get_config()
        config.update({
            "d_model": self.d_model,
            "kernel_size": self.kernel_size,
            "use_se": self.use_se,
            # Add dropout if needed
        })
        return config


class ConformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, kernel_size=31, ff_expansion=4, dropout=0.1, drop_path_rate=0.1, use_se_in_conv=True, name="conformer_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ffm1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mha = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout, use_se=use_se_in_conv)
        self.ffm2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm = layers.LayerNormalization() # Final LayerNorm

        # Stochastic Depth (DropPath) for each module application
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        self.drop_path3 = DropPath(drop_path_rate)
        self.drop_path4 = DropPath(drop_path_rate)

    def call(self, x, training=None):
        # Conformer structure: FFN -> MHA -> Conv -> FFN -> LayerNorm
        # Each main module application is wrapped in a residual connection with DropPath

        # FFN Module 1
        ffm1_output = self.ffm1(x)
        x = x + 0.5 * self.drop_path1(ffm1_output, training=training)

        # MHA Module
        mha_output = self.mha(x)
        x = x + self.drop_path2(mha_output, training=training)

        # Conv Module
        conv_output = self.conv(x)
        x = x + self.drop_path3(conv_output, training=training)

        # FFN Module 2
        ffm2_output = self.ffm2(x)
        x = x + 0.5 * self.drop_path4(ffm2_output, training=training)

        # Final LayerNorm
        x = self.norm(x)
        return x

    def get_config(self):
        config = super(ConformerBlock, self).get_config()
        # Add necessary parameters here if they aren't automatically captured
        # from the sub-modules. DropPath rate is important.
        # config.update({"drop_path_rate": self.drop_path_rate}) # Assuming stored
        return config


# --- LOAD DATASET ---
def load_and_preprocess_data(data_dir, labels_dict, maxlen=None, trim_db=30, normalize=True):
    """Loads audio, preprocesses, and pads/truncates."""
    X, y = [], []
    print(f"Loading data from {data_dir}...")
    start_time = time.time()
    processed_count = 0
    error_count = 0

    all_lengths = [] # To determine maxlen if not provided

    for label_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(class_dir) or label_name not in labels_dict:
            continue
        label = labels_dict[label_name]
        print(f"  Processing class: {label_name}")
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".wav"):
                path = os.path.join(class_dir, fname)
                signal = load_audio_file(path, trim_db=trim_db)
                if signal is None or len(signal) == 0:
                    print(f"    Skipping empty or invalid file: {fname}")
                    error_count += 1
                    continue

                mel = extract_mel_spectrogram(signal)

                if normalize:
                    mel = normalize_spectrogram(mel)

                if mel.shape[0] > 0: # Check if mel spectrogram is not empty
                    X.append(mel)
                    y.append(label)
                    all_lengths.append(mel.shape[0])
                    processed_count += 1
                else:
                     print(f"    Skipping file with empty spectrogram: {fname}")
                     error_count += 1


    end_time = time.time()
    print(f"Processed {processed_count} files ({error_count} errors) in {end_time - start_time:.2f} seconds.")

    if not X:
        raise ValueError("No valid data loaded. Check data directory and file formats.")

    # Determine maxlen from loaded training data if not provided
    current_maxlen = max(all_lengths) if not maxlen else maxlen
    print(f"Determined max sequence length (time steps): {current_maxlen}")

    # Pad sequences
    # Padnačuje na konci sekvence (post-padding)
    X_padded = np.array([np.pad(x, ((0, current_maxlen - x.shape[0]), (0, 0)), mode='constant', constant_values=0.0) for x in X])

    return X_padded, np.array(y), current_maxlen


# --- BUILD MODEL ---
def build_conformer_model(input_shape, d_model=144, num_blocks=4, num_heads=4, ff_expansion=4, kernel_size=31, dropout=0.1, drop_path_rate=0.1, use_se_in_conv=True, use_spec_augment=True, freq_mask_param=27, time_mask_param=50):
    """Builds the Conformer model with configurable options."""
    inputs = layers.Input(shape=input_shape) # e.g., (maxlen, NUM_MELS)

    # Initial Projection/Subsampling (Optional, common in ASR Conformers)
    # Example: x = layers.Conv2D(d_model, kernel_size=3, strides=2, padding='same')(tf.expand_dims(inputs, axis=-1))
    # x = layers.Reshape((shape[1]//2, shape[2]//2 * d_model))(x) # Adjust shape
    # For now, use a Dense layer like before
    x = layers.Dense(d_model, activation='relu')(inputs) # Project features to d_model

    # Add Positional Encoding
    x = PositionalEncoding(d_model)(x)
    x = layers.Dropout(dropout)(x) # Dropout after pos encoding

    # Optional SpecAugment
    if use_spec_augment:
        x = SpecAugment(freq_mask_param=freq_mask_param, time_mask_param=time_mask_param)(x) # Applied only during training

    # Conformer Blocks
    for i in range(num_blocks):
        # Linearly scale drop path rate
        block_drop_path = drop_path_rate * float(i) / num_blocks
        x = ConformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            kernel_size=kernel_size,
            ff_expansion=ff_expansion,
            dropout=dropout,
            drop_path_rate=block_drop_path,
            use_se_in_conv=use_se_in_conv,
            name=f"conformer_block_{i}"
        )(x)

    # Pooling and Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x) # Dropout before final dense layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

# --- UTILITY FUNCTIONS ---
def get_model_path(iteration, version):
    """Generates the model filename."""
    return f"CONFORMERv{iteration}.{version}.keras"

def get_next_version_number(iteration):
    """Finds the next available version number for a given iteration."""
    existing_versions = [0] # Start with 0 in case no models exist
    for fname in os.listdir("."):
        if fname.startswith(f"CONFORMERv{iteration}.") and fname.endswith(".keras"):
            try:
                # Handle potential format issues (e.g., "CONFORMERv1.1_final.keras")
                version_part = fname.split(".")[1]
                if version_part.isdigit():
                    existing_versions.append(int(version_part))
            except (IndexError, ValueError):
                continue # Ignore files that don't match the expected format
    return max(existing_versions) + 1

# --- TRAIN ---
def train_model(args):
    """Loads data, builds model, trains, and saves."""
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'SpecAugment': SpecAugment,
        'SEBlock': SEBlock,
        'DropPath': DropPath,
        'FeedForwardModule': FeedForwardModule,
        'MultiHeadSelfAttentionModule': MultiHeadSelfAttentionModule,
        'ConvolutionModule': ConvolutionModule,
        'ConformerBlock': ConformerBlock
    }

    # Load or Retrain Logic
    if not args.retrain:
        if args.version is None:
            raise ValueError("Must specify --version to load an existing model.")
        model_path = get_model_path(args.iteration, args.version)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at: {model_path}")

        print(f"Loading saved model from {model_path}...")
        model = models.load_model(model_path, custom_objects=custom_objects)
        # Need to know the maxlen used during training for prediction padding
        # Ideally, save maxlen with the model, or infer from input shape if fixed
        try:
            # Attempt to get maxlen from the model's input shape
            # Assumes input shape is (batch, maxlen, features)
            loaded_maxlen = model.input_shape[1]
            if loaded_maxlen is None: # Handle cases where input shape isn't fully defined
                 raise ValueError("Could not determine maxlen from loaded model's input shape. Consider saving it separately.")
            print(f"Inferred maxlen from model: {loaded_maxlen}")
            return model, loaded_maxlen
        except Exception as e:
             print(f"Warning: Could not automatically determine maxlen from loaded model ({e}). You might need to specify it or load training data to find it.")
             # As a fallback, try loading a small part of training data just to get maxlen
             _, _, loaded_maxlen = load_and_preprocess_data(
                 DATA_DIR, LABELS, maxlen=None, trim_db=args.trim_db, normalize=args.normalize
            )
             print(f"Determined maxlen by reloading training data: {loaded_maxlen}")
             if loaded_maxlen is None: raise ValueError("Failed to determine maxlen.")
             return model, loaded_maxlen


    # --- Retraining Path ---
    print("Loading training data and preprocessing...")
    X, y, train_maxlen = load_and_preprocess_data(
        DATA_DIR, LABELS, maxlen=None, trim_db=args.trim_db, normalize=args.normalize
    )

    # Train/Validation Split
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    # Build Model
    print("Building model...")
    model = build_conformer_model(
        input_shape=X_train.shape[1:], # (maxlen, num_mels)
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

    # Compile Model
    # Consider using AdamW optimizer and learning rate schedules
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    version = get_next_version_number(args.iteration)
    model_path = get_model_path(args.iteration, version)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=args.patience, restore_best_weights=True, verbose=1
    )

    # Train Model
    print(f"Starting training (Iteration {args.iteration}, Version {version})...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    print(f"Training complete. Best model saved to {model_path}")

    # Load the best saved model (due to EarlyStopping restoring best weights)
    # Or explicitly load if save_best_only was used without restore_best_weights
    # model = models.load_model(model_path, custom_objects=custom_objects) # Redundant if restore_best_weights=True

    return model, train_maxlen

# --- PREDICT ON TEST FILES ---
def predict_on_test(model, maxlen, test_dir, labels_dict, trim_db=30, normalize=True, batch_size=16):
    """Predicts on files in the test directory."""
    print(f"\nPredicting on test files in {test_dir}...")
    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".wav")])
    if not test_files:
        print("No .wav files found in the test directory.")
        return

    results = []
    num_batches = (len(test_files) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_files = test_files[i * batch_size : (i + 1) * batch_size]
        batch_data = []
        valid_fnames = []

        print(f"Processing batch {i+1}/{num_batches}...")
        for fname in batch_files:
            path = os.path.join(test_dir, fname)
            signal = load_audio_file(path, trim_db=trim_db)
            if signal is None or len(signal) == 0:
                print(f"  Skipping empty/invalid test file: {fname}")
                continue

            mel = extract_mel_spectrogram(signal)
            if mel.shape[0] == 0:
                 print(f"  Skipping test file with empty spectrogram: {fname}")
                 continue

            if normalize:
                mel = normalize_spectrogram(mel)

            # Pad/truncate to the maxlen used during training
            if mel.shape[0] > maxlen:
                mel = mel[:maxlen, :]
            else:
                mel = np.pad(mel, ((0, maxlen - mel.shape[0]), (0, 0)), mode='constant', constant_values=0.0)

            batch_data.append(mel)
            valid_fnames.append(fname)

        if not batch_data:
            print("  No valid data in this batch.")
            continue

        # Predict on the batch
        batch_data_np = np.array(batch_data)
        predictions = model.predict(batch_data_np, verbose=0) # verbose=0 to reduce console spam

        # Store results for this batch
        for fname, pred in zip(valid_fnames, predictions):
            prob = float(pred[0])
            label = 'FAKE' if prob > 0.5 else 'REAL'
            certainty = prob if label == 'FAKE' else 1 - prob
            results.append((fname, label, certainty))
            print(f"  {fname}: {label} (confidence: {certainty:.3f})") # Print immediately

    print("\n--- Prediction Summary ---")
    if results:
        # Sort by filename for consistent output
        results.sort(key=lambda item: item[0])
        for fname, label, certainty in results:
             print(f"{fname}: {label} (confidence: {certainty:.3f})")
    else:
        print("No predictions were made.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a Conformer model for Deepfake Audio Detection.")

    # Model Loading/Saving
    parser.add_argument("--retrain", action="store_true", help="Force retraining the model instead of loading.")
    parser.add_argument("--iteration", type=int, default=1, help="Model iteration number (e.g., 1, 2, 3...).")
    parser.add_argument("--version", type=int, help="Specify model version to load (required if not retraining).")

    # Data Preprocessing
    parser.add_argument("--trim_db", type=int, default=30, help="Top dB for silence trimming (set to None to disable).")
    parser.add_argument("--no_normalize", dest='normalize', action="store_false", help="Disable spectrogram Z-score normalization.")

    # Model Architecture
    parser.add_argument("--d_model", type=int, default=144, help="Internal dimension of the model.")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of Conformer blocks.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff_expansion", type=int, default=4, help="Expansion factor for feed-forward layers.")
    parser.add_argument("--kernel_size", type=int, default=31, help="Kernel size for depthwise convolution.")
    parser.add_argument("--no_se", dest='use_se', action="store_false", help="Disable Squeeze-and-Excitation in Convolution Module.")

    # Regularization & Augmentation
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--drop_path", type=float, default=0.1, help="Stochastic depth (DropPath) rate.")
    parser.add_argument("--no_specaugment", dest='use_specaugment', action="store_false", help="Disable SpecAugment.")
    parser.add_argument("--freq_mask", type=int, default=27, help="Frequency masking parameter F for SpecAugment.")
    parser.add_argument("--time_mask", type=int, default=50, help="Time masking parameter T for SpecAugment.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.") # Increased default
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.") # Increased default
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.") # Common starting LR
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.") # Added patience
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set split ratio.") # Slightly smaller default

    # Directories
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the training data directory.")
    parser.add_argument("--test_dir", type=str, default=TEST_DIR, help="Path to the testing data directory.")


    args = parser.parse_args()

    # Update global vars based on args if needed (though passing args is cleaner)
    DATA_DIR = args.data_dir
    TEST_DIR = args.test_dir

    try:
        model, trained_maxlen = train_model(args)
        if trained_maxlen is None:
             raise ValueError("Could not determine the required 'maxlen' for prediction.")

        predict_on_test(
             model,
             maxlen=trained_maxlen,
             test_dir=args.test_dir,
             labels_dict=LABELS, # Pass LABELS dict
             trim_db=args.trim_db,
             normalize=args.normalize,
             batch_size=args.batch_size # Use same batch size for prediction efficiency
         )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the specified model path or data directories.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the arguments or data integrity.")
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        # Optionally add traceback here for debugging
        # import traceback
        # traceback.print_exc()