import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import argparse

# --- PARAMETERS ---
SAMPLE_RATE = 16000
DURATION = 2.0  # seconds
NUM_MELS = 40
LABELS = {'real': 0, 'fake': 1}
DATA_DIR = "for-2seconds/training"
TEST_DIR = "for-2seconds/testing"
MAX_LEN = int(SAMPLE_RATE * DURATION)


# --- AUDIO PROCESSING ---
def load_audio_file(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(signal) > MAX_LEN:
        signal = signal[:MAX_LEN]
    else:
        signal = np.pad(signal, (0, MAX_LEN - len(signal)))
    return signal

def extract_mel_spectrogram(signal):
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_mels=NUM_MELS,
        fmax=8000
    )
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec.T  # shape: (time, mel)

# --- LOAD DATASET ---
def load_dataset():
    X, y = [], []
    for label_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(class_dir):
            continue
        label = LABELS[label_name]
        for fname in os.listdir(class_dir):
            if fname.endswith(".wav"):
                path = os.path.join(class_dir, fname)
                signal = load_audio_file(path)
                mel = extract_mel_spectrogram(signal)
                X.append(mel)
                y.append(label)
    return np.array(X), np.array(y)

# --- CONFORMER BLOCK ---
class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(d_model * expansion_factor, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])

    def call(self, x):
        return x + 0.5 * self.seq(x)

class MultiHeadSelfAttentionModule(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.dropout = layers.Dropout(0.1)

    def call(self, x):
        x_norm = self.norm(x)
        attn_output = self.mha(x_norm, x_norm)
        return x + self.dropout(attn_output)

class ConvolutionModule(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.pointwise_conv1 = layers.Conv1D(filters=2*d_model, kernel_size=1, padding='same', activation='relu')
        self.depthwise_conv = layers.DepthwiseConv1D(kernel_size=kernel_size, padding='same', activation='relu')
        self.batch_norm = layers.BatchNormalization()
        self.pointwise_conv2 = layers.Conv1D(filters=d_model, kernel_size=1, padding='same')
        self.dropout = layers.Dropout(0.1)

    def call(self, x):
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.pointwise_conv2(x)
        return x

class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, kernel_size=31):
        super().__init__()
        self.ffm1 = FeedForwardModule(d_model)
        self.mha = MultiHeadSelfAttentionModule(d_model, num_heads)
        self.conv = ConvolutionModule(d_model, kernel_size)
        self.ffm2 = FeedForwardModule(d_model)
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x = self.ffm1(x)
        x = self.mha(x)
        x = x + self.conv(x)
        x = self.ffm2(x)
        return self.norm(x)

# --- BUILD MODEL ---
def build_conformer_model(input_shape, d_model=144, num_blocks=2, num_heads=4):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(d_model)(inputs)
    for _ in range(num_blocks):
        x = ConformerBlock(d_model, num_heads)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

# --- TRAIN ---
def train_model(force_retrain=False, iteration=1, version=None):
    if not force_retrain:
        if version is None:
            raise ValueError("Must specify --version when loading an existing model.")
        model_path = get_model_path(iteration, version)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at: {model_path}")
        print(f"Loading saved model from {model_path}...")
        model = tf.keras.models.load_model(model_path, custom_objects={
            'FeedForwardModule': FeedForwardModule,
            'MultiHeadSelfAttentionModule': MultiHeadSelfAttentionModule,
            'ConvolutionModule': ConvolutionModule,
            'ConformerBlock': ConformerBlock
        })
        dummy_input_shape = model.input_shape[1]
        return model, dummy_input_shape

    # Otherwise: retrain and save
    print("Loading data and training model...")
    X, y = load_dataset()
    maxlen = max([x.shape[0] for x in X])
    X = np.array([np.pad(x, ((0, maxlen - x.shape[0]), (0, 0))) for x in X])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_conformer_model(input_shape=X.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val))

    # Save with incremented version
    version = get_next_version_number(iteration)
    model_path = get_model_path(iteration, version)
    print(f"Saving model to {model_path}...")
    model.save(model_path)

    return model, maxlen

# --- PREDICT ON TEST FILES ---
def predict_on_test(model, maxlen):
    print("Predicting on test files...")
    predictions = []
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".wav")])[:100]
    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        signal = load_audio_file(path)
        mel = extract_mel_spectrogram(signal)
        mel = np.pad(mel, ((0, maxlen - mel.shape[0]), (0, 0)))  # pad to match training shape
        mel = np.expand_dims(mel, axis=0)  # batch dim
        pred = model.predict(mel)[0][0]
        predictions.append((fname, float(pred)))

    for fname, prob in predictions:
        label = 'FAKE' if prob > 0.5 else 'REAL'
        certainty = prob if label == 'FAKE' else 1 - prob
        print(f"{fname}: {label} (confidence: {certainty:.2f})")

def get_model_path(iteration, version):
    return f"CONFORMERv{iteration}.{version}.keras"

def get_next_version_number(iteration):
    existing_versions = []
    for fname in os.listdir("."):
        if fname.startswith(f"CONFORMERv{iteration}.") and fname.endswith(".keras"):
            try:
                ver_str = fname.split(".")[1]
                existing_versions.append(int(ver_str))
            except ValueError:
                continue
    return max(existing_versions, default=0) + 1

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retraining the model.")
    parser.add_argument("--iteration", type=int, default=1, help="Model iteration number (e.g., 1, 2, 3...)")
    parser.add_argument("--version", type=int, help="Specify model version to load (required if not retraining)")
    args = parser.parse_args()

    model, maxlen = train_model(
        force_retrain=args.retrain,
        iteration=args.iteration,
        version=args.version
    )
    predict_on_test(model, maxlen)