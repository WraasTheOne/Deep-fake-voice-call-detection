# model/build_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from .custom_layers import PositionalEncoding, SpecAugment # Relative imports
from .conformer_blocks import ConformerBlock # Relative imports

def build_conformer_model(input_shape, d_model, num_blocks, num_heads,
                          ff_expansion, kernel_size, dropout, drop_path_rate,
                          use_se_in_conv, use_spec_augment, freq_mask_param, time_mask_param):
    """Builds the complete Conformer model."""

    inputs = layers.Input(shape=input_shape, name="input_spectrogram") # e.g., (maxlen, NUM_MELS)

    # 1. Initial Projection/Embedding
    # Project input features (Mel bins) to the model dimension (d_model)
    x = layers.Dense(d_model, activation='relu', name="input_projection")(inputs)

    # 2. Positional Encoding
    # Add positional information. Max length should be sufficient for expected sequences.
    x = PositionalEncoding(d_model, max_len=input_shape[0] * 2, name="positional_encoding")(x) # Dynamic max_len estimate
    x = layers.Dropout(dropout, name="pos_encoding_dropout")(x)

    # 3. Optional SpecAugment (Applied only during training)
    if use_spec_augment:
        x = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            name="spec_augment"
        )(x) # Training flag handled internally by the layer

    # 4. Conformer Blocks
    for i in range(num_blocks):
        # Linearly scale drop path rate for deeper blocks (optional, common practice)
        block_drop_path = drop_path_rate * float(i + 1) / num_blocks # Scale from 0 up to drop_path_rate
        x = ConformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            kernel_size=kernel_size,
            ff_expansion=ff_expansion,
            dropout=dropout,
            drop_path_rate=block_drop_path, # Use scaled rate
            use_se_in_conv=use_se_in_conv,
            name=f"conformer_block_{i}"
        )(x) # Pass training flag implicitly via Keras fit/predict

    # 5. Pooling & Classification Head
    # Average features across the time dimension
    x = layers.GlobalAveragePooling1D(name="global_avg_pooling")(x)
    x = layers.Dropout(dropout, name="pooling_dropout")(x) # Dropout before final dense layer
    # Output layer for binary classification
    outputs = layers.Dense(1, activation='sigmoid', name="output_sigmoid")(x)

    # Create and return the Keras model
    model = models.Model(inputs=inputs, outputs=outputs, name="ConformerDeepfakeDetector")
    return model