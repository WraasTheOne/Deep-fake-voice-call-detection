# model/conformer_blocks.py
import tensorflow as tf
from tensorflow.keras import layers
from .custom_layers import SEBlock, DropPath # Use relative import

# --- Feed Forward Module (FFN) ---
class FeedForwardModule(layers.Layer):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1, name="ff_module", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        self.norm = layers.LayerNormalization()
        # Use 'swish' (equivalent to silu) activation
        self.dense1 = layers.Dense(d_model * expansion_factor, activation='swish')
        self.dropout1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(d_model)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        x_norm = self.norm(x)
        x_ff = self.dense1(x_norm)
        x_ff = self.dropout1(x_ff)
        x_ff = self.dense2(x_ff)
        x_ff = self.dropout2(x_ff)
        return x_ff # Residual connection handled externally

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "expansion_factor": self.expansion_factor,
            "dropout": self.dropout,
        })
        return config


# --- Multi-Head Self-Attention Module (MHSA) ---
class MultiHeadSelfAttentionModule(layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, name="mha_module", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        if d_model % num_heads != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.key_dim = d_model // num_heads

        self.norm = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.key_dim, dropout=dropout # Pass dropout to MHA layer itself
        )
        self.dropout_output = layers.Dropout(dropout) # Separate dropout after MHA output

    def call(self, x, mask=None): # Allow mask passing if needed later
        x_norm = self.norm(x)
        # MHA expects query, value, key. For self-attention, they are the same.
        attn_output = self.mha(query=x_norm, value=x_norm, key=x_norm, attention_mask=mask)
        attn_output = self.dropout_output(attn_output)
        return attn_output # Residual connection handled externally

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        })
        return config


# --- Convolution Module ---
class ConvolutionModule(layers.Layer):
    def __init__(self, d_model, kernel_size=31, dropout=0.1, use_se=True, name="conv_module", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dropout_rate = dropout # Store dropout rate
        self.use_se = use_se

        self.norm = layers.LayerNormalization()
        # Pointwise Conv -> GLU Activation
        self.pointwise_conv1 = layers.Conv1D(filters=2 * d_model, kernel_size=1, padding='same')
        # Depthwise Conv
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=kernel_size, padding='same', depth_multiplier=1
        )
        self.batch_norm = layers.BatchNormalization()
        # Optional Squeeze-and-Excitation
        if self.use_se:
            # Pass d_model as input_channels to SEBlock
            self.se_block = SEBlock(input_channels=d_model, name="se_block")
        # Swish Activation
        self.activation = layers.Activation('swish')
        # Pointwise Conv
        self.pointwise_conv2 = layers.Conv1D(filters=d_model, kernel_size=1, padding='same')
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        x_norm = self.norm(x)
        x_conv = self.pointwise_conv1(x_norm)

        # GLU Activation
        x_gate, x_filter = tf.split(x_conv, num_or_size_splits=2, axis=-1)
        x_conv = x_gate * tf.nn.sigmoid(x_filter) # Sigmoid for gating

        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.batch_norm(x_conv) # Apply BN before activation
        x_conv = self.activation(x_conv) # Apply Swish

        if self.use_se:
            x_conv = self.se_block(x_conv) # Apply SE block

        x_conv = self.pointwise_conv2(x_conv) # Final pointwise conv
        x_conv = self.dropout(x_conv) # Apply dropout
        return x_conv # Residual connection handled externally

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_rate, # Use stored rate
            "use_se": self.use_se,
        })
        return config


# --- Conformer Block ---
class ConformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, kernel_size=31, ff_expansion=4, dropout=0.1, drop_path_rate=0.0, use_se_in_conv=True, name="conformer_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.ff_expansion = ff_expansion
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.use_se_in_conv = use_se_in_conv

        # Instantiate modules
        self.ffm1 = FeedForwardModule(d_model, ff_expansion, dropout, name="ffm_1")
        self.mha = MultiHeadSelfAttentionModule(d_model, num_heads, dropout, name="mha")
        self.conv = ConvolutionModule(d_model, kernel_size, dropout, use_se=use_se_in_conv, name="conv")
        self.ffm2 = FeedForwardModule(d_model, ff_expansion, dropout, name="ffm_2")
        self.norm_final = layers.LayerNormalization(name="final_norm") # Final LayerNorm

        # DropPath layers for residual connections
        self.drop_path = DropPath(drop_path_rate)

    def call(self, x, training=None, mask=None): # Pass training flag and optional mask
        # FFN Module 1 (Scaled residual)
        ffm1_output = self.ffm1(x)
        x = x + 0.5 * self.drop_path(ffm1_output, training=training)

        # MHA Module
        mha_output = self.mha(x, mask=mask) # Pass mask if needed
        x = x + self.drop_path(mha_output, training=training)

        # Conv Module
        conv_output = self.conv(x)
        x = x + self.drop_path(conv_output, training=training)

        # FFN Module 2 (Scaled residual)
        ffm2_output = self.ffm2(x)
        x = x + 0.5 * self.drop_path(ffm2_output, training=training)

        # Final LayerNorm
        x = self.norm_final(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "kernel_size": self.kernel_size,
            "ff_expansion": self.ff_expansion,
            "dropout": self.dropout,
            "drop_path_rate": self.drop_path_rate,
            "use_se_in_conv": self.use_se_in_conv,
        })
        return config