# config.py
import os

# --- Data Parameters ---
# ... (Keep SAMPLE_RATE, NUM_MELS, etc. as they were) ...
SAMPLE_RATE = 16000
NUM_MELS = 80      # Mel frequency bins
FFT_N = 2048       # FFT window size
HOP_LENGTH = 512   # Hop length for STFT
FMAX = 8000        # Maximum frequency for Mel scale
LABELS = {'real': 0, 'fake': 1} # Class labels


# --- Directory Structure Setup ---

# Get the absolute path to the directory containing this config.py file
# Assuming config.py is in the project root directory THIS time.
# Example: /Users/nichoalsmazur/Documents/GitHub/Deep-fake-voice-call-detection
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the base data directory relative to the project root
# Assumes 'for-2seconds' is directly inside the PROJECT_ROOT_DIR
# Example: /Users/nichoalsmazar/Documents/GitHub/Deep-fake-voice-call-detection/for-2seconds
BASE_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'for-2seconds')

# --- Default Data Directories ---
DEFAULT_TRAIN_DIR = os.path.join(BASE_DATA_DIR, "training")
DEFAULT_VAL_DIR = os.path.join(BASE_DATA_DIR, "validation")
DEFAULT_TEST_DIR = os.path.join(BASE_DATA_DIR, "testing")

# --- Default Model Save Directory ---
# Place 'saved_models' inside the main project directory (PROJECT_ROOT_DIR)
DEFAULT_MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, "saved_models")


# --- Default Model Hyperparameters ---
# ... (Keep these defaults as they were) ...
DEFAULT_D_MODEL = 144
DEFAULT_NUM_BLOCKS = 4   # <--- MAKE SURE THIS LINE EXISTS AND ISN'T COMMENTED
DEFAULT_NUM_HEADS = 4
DEFAULT_FF_EXPANSION = 4
DEFAULT_KERNEL_SIZE = 31
DEFAULT_DROPOUT = 0.1
DEFAULT_DROP_PATH = 0.1
DEFAULT_USE_SE = True
DEFAULT_USE_SPECAUGMENT = True
DEFAULT_FREQ_MASK = 27
DEFAULT_TIME_MASK = 50

# --- Default Training Hyperparameters ---
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_PATIENCE = 10

# --- Default Preprocessing Parameters ---
DEFAULT_TRIM_DB = 30 # Use None to disable trimming
DEFAULT_NORMALIZE = True
