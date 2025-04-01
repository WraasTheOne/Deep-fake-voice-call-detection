# model/custom_layers.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# --- SpecAugment ---
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

    @tf.function
    def call(self, inputs, training=None):

        def augment_internal():
            """Applies SpecAugment logic assuming rank is 3 or 4."""
            # Initialize loop variable from inputs
            augmented_outputs = inputs
            # Get static shape info for use in set_shape later
            input_static_shape = inputs.get_shape()

            rank = tf.rank(inputs)
            input_shape = tf.shape(inputs) # Symbolic shape

            # Determine axes (assuming axis 2 for freq, axis 1 for time in 3D/4D)
            freq_axis = 2
            time_axis = 1

            num_freq_bins = input_shape[freq_axis]
            num_time_steps = input_shape[time_axis]

            # --- Frequency Masking ---
            for _ in tf.range(self.num_freq_masks): # Use tf.range
                f = tf.random.uniform([], minval=0, maxval=self.freq_mask_param, dtype=tf.int32)
                f = tf.minimum(f, num_freq_bins)
                f0 = tf.random.uniform([], minval=0, maxval=num_freq_bins - f, dtype=tf.int32)

                # Create 1D mask for the frequency dimension
                mask_1d = tf.concat([
                    tf.ones([f0], dtype=augmented_outputs.dtype),
                    tf.zeros([f], dtype=augmented_outputs.dtype),
                    tf.ones([num_freq_bins - f0 - f], dtype=augmented_outputs.dtype)
                ], axis=0)

                # Reshape for broadcasting across other dimensions
                reshape_target = tf.ones(rank, dtype=tf.int32)
                reshape_target = tf.tensor_scatter_nd_update(
                    reshape_target, [[freq_axis]], [num_freq_bins]
                )
                mask = tf.reshape(mask_1d, reshape_target)

                # Apply the broadcasted mask
                augmented_outputs = augmented_outputs * mask
                # ***** SOLUTION: Explicitly set shape *****
                augmented_outputs.set_shape(input_static_shape)

            # --- Time Masking ---
            for _ in tf.range(self.num_time_masks): # Use tf.range
                t = tf.random.uniform([], minval=0, maxval=self.time_mask_param, dtype=tf.int32)
                t = tf.minimum(t, num_time_steps)
                t0 = tf.random.uniform([], minval=0, maxval=num_time_steps - t, dtype=tf.int32)

                # Create 1D mask for the time dimension
                mask_1d = tf.concat([
                    tf.ones([t0], dtype=augmented_outputs.dtype),
                    tf.zeros([t], dtype=augmented_outputs.dtype),
                    tf.ones([num_time_steps - t0 - t], dtype=augmented_outputs.dtype)
                ], axis=0)

                # Reshape for broadcasting across other dimensions
                reshape_target = tf.ones(rank, dtype=tf.int32)
                reshape_target = tf.tensor_scatter_nd_update(
                    reshape_target, [[time_axis]], [num_time_steps]
                )
                mask = tf.reshape(mask_1d, reshape_target)

                # Apply the broadcasted mask
                augmented_outputs = augmented_outputs * mask
                # ***** SOLUTION: Explicitly set shape *****
                augmented_outputs.set_shape(input_static_shape)

            return augmented_outputs # Return the final augmented tensor
        # --- End of augment_internal definition ---

        # Use tf.cond to decide whether to run augmentation or just return inputs
        if training is None:
             return inputs # Treat None as False
        elif isinstance(training, bool):
             if training:
                 rank = tf.rank(inputs)
                 pred = tf.logical_or(tf.equal(rank, 3), tf.equal(rank, 4))
                 return tf.cond(pred, true_fn=augment_internal, false_fn=lambda: inputs)
             else:
                 return inputs
        else: # Assume symbolic tensor for training
             rank = tf.rank(inputs)
             pred_rank_valid = tf.logical_or(tf.equal(rank, 3), tf.equal(rank, 4))
             return tf.cond(tf.logical_and(training, pred_rank_valid),
                            true_fn=augment_internal,
                            false_fn=lambda: inputs)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        return input_shape

    def get_config(self):
        # Config remains the same
        config = super(SpecAugment, self).get_config()
        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "num_freq_masks": self.num_freq_masks,
            "num_time_masks": self.num_time_masks,
        })
        return config

# --- PositionalEncoding ---
# ... (Keep the PositionalEncoding class code as it was) ...
class PositionalEncoding(layers.Layer):
    """Adds sinusoidal positional encoding.
    Reference: Vaswani et al. (2017) Attention Is All You Need (https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        if not isinstance(d_model, int) or d_model <= 0:
             raise ValueError(f"d_model must be a positive integer, got {d_model}")
        if not isinstance(max_len, int) or max_len <= 0:
             raise ValueError(f"max_len must be a positive integer, got {max_len}")

        self.d_model = d_model
        self.max_len = max_len
        # Precompute the positional encoding matrix
        self.pos_encoding = self._build_encoding(max_len, d_model)

    def _build_encoding(self, length, depth):
        if depth % 2 != 0:
            # Handle odd depth if necessary, or raise error earlier
            print(f"Warning: PositionalEncoding depth (d_model={depth}) is odd. Ensure handling is intended.")
            # Raise ValueError(f"d_model must be even for sinusoidal positional encoding, got {depth}")

        depth_calc = depth / 2
        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth_calc)[np.newaxis, :]/depth_calc   # (1, depth/2)

        angle_rates = 1 / (10000**depths)         # (1, depth/2)
        angle_rads = positions * angle_rates      # (pos, depth/2)

        sines = np.sin(angle_rads)
        cosines = np.cos(angle_rads)

        # Ensure concatenation matches d_model, especially if d_model was odd
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        # If d_model was odd, the calculated shape might be d_model-1. Pad if needed.
        if pos_encoding.shape[-1] < self.d_model:
            padding = np.zeros((length, self.d_model - pos_encoding.shape[-1]))
            pos_encoding = np.concatenate([pos_encoding, padding], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        # x shape: (batch, sequence_length, d_model)
        length = tf.shape(x)[1] # Get symbolic sequence length from input tensor

        # Calculate the slice length, capped by the precomputed max_len
        slice_length = tf.minimum(length, self.max_len)

        # Get the base slice from the precomputed encoding table
        # Shape: (1, slice_length, d_model)
        pos_encoding_slice_base = self.pos_encoding[tf.newaxis, :slice_length, :]

        # Defensive check: Ensure feature dimension matches d_model
        tf.debugging.assert_equal(tf.shape(pos_encoding_slice_base)[-1], self.d_model,
                                   message=f"PositionalEncoding feature dimension mismatch")

        # --- Use tf.cond for conditional padding ---
        # Condition: Is the actual input length greater than the slice we took?
        # This only happens if input length > self.max_len
        padding_needed_cond = tf.greater(length, slice_length)

        def pad_encoding():
            # This branch executes if input length > self.max_len
            tf.print("Warning: Input sequence length > PositionalEncoding max_len. Padding encoding.") # Optional runtime warning
            padding_size = length - slice_length
            # Paddings format: [[before_dim0, after_dim0], [before_dim1, after_dim1], ...]
            paddings = [[0, 0],             # Batch dim: no padding
                        [0, padding_size],  # Time dim: pad at the end
                        [0, 0]]             # Feature dim: no padding
            # Pad with zeros
            return tf.pad(pos_encoding_slice_base, paddings, "CONSTANT", constant_values=0.0)

        def no_pad_encoding():
            # This branch executes if input length <= self.max_len
            # The slice length already matches input length in this case
            return pos_encoding_slice_base

        # Conditionally select the padded or original slice based on the condition
        pos_encoding_slice_final = tf.cond(padding_needed_cond,
                                           true_fn=pad_encoding,    # Execute if length > slice_length
                                           false_fn=no_pad_encoding) # Execute if length <= slice_length

        # Add the (potentially padded) positional encoding to the input tensor 'x'
        return x + pos_encoding_slice_final

    # Add this method to explicitly define the output shape
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        # PositionalEncoding adds element-wise, so shape doesn't change.
        return input_shape

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return config


# --- Squeeze-and-Excitation Block ---
# ... (Keep the SEBlock class code as it was) ...
class SEBlock(layers.Layer):
    # ... (Full code for SEBlock) ...
    """Squeeze-and-Excitation block."""
    def __init__(self, input_channels, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.ratio = ratio
        reduced_channels = max(1, input_channels // ratio)
        self.pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(reduced_channels, activation='relu', name="se_fc1")
        self.fc2 = layers.Dense(input_channels, activation='sigmoid', name="se_fc2")

    def call(self, inputs):
        se = self.pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.expand_dims(se, axis=1)
        return inputs * se

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "input_channels": self.input_channels,
            "ratio": self.ratio,
        })
        return config

# --- DropPath (Stochastic Depth) ---
# ... (Keep the DropPath class code as it was) ...
class DropPath(layers.Layer):
     # ... (Full code for DropPath) ...
    """Stochastic depth layer (DropPath)."""
    def __init__(self, drop_prob=0.0, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if self.drop_prob == 0. or not training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.floor(keep_prob + random_tensor)
        output = tf.math.divide(x, keep_prob) * binary_mask
        return output

    def get_config(self):
        config = super(DropPath, self).get_config()
        config.update({"drop_prob": self.drop_prob})
        return config