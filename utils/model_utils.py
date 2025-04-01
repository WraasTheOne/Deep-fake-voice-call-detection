# utils/model_utils.py
import os
import json
import sys # Import sys for path checking if needed

# --- Attempt to import model components ---
# When train.py/predict.py run from the project root, Python's path
# should include the root, allowing us to find the 'model' directory.
try:
    # Direct imports assuming 'model' is findable from the project root
    from model.custom_layers import PositionalEncoding, SpecAugment, SEBlock, DropPath
    from model.conformer_blocks import FeedForwardModule, MultiHeadSelfAttentionModule, ConvolutionModule, ConformerBlock

    # Define the dictionary needed for loading saved models
    CUSTOM_OBJECTS = {
        'PositionalEncoding': PositionalEncoding,
        'SpecAugment': SpecAugment,
        'SEBlock': SEBlock,
        'DropPath': DropPath,
        'FeedForwardModule': FeedForwardModule,
        'MultiHeadSelfAttentionModule': MultiHeadSelfAttentionModule,
        'ConvolutionModule': ConvolutionModule,
        'ConformerBlock': ConformerBlock
    }
    print("Successfully imported model components in model_utils.")

except ImportError as e:
    print(f"ERROR in model_utils.py: Could not import components from the 'model' directory. Error: {e}")
    print("Please ensure:")
    print("1. You are running train.py or predict.py from the main project root directory.")
    print("2. The 'model' directory exists at the same level as 'utils', 'train.py', etc.")
    print(f"Current Python Path (sys.path): {sys.path}") # Print path for debugging
    # Define an empty dict or raise an error if imports fail, as model loading will break
    CUSTOM_OBJECTS = {}
    raise ImportError("Failed to import necessary model components for CUSTOM_OBJECTS.") from e


# --- Utility Functions ---

def get_model_filename(iteration, version):
    """Generates the base model filename (without extension)."""
    return f"CONFORMERv{iteration}.{version}"

def get_model_paths(save_dir, iteration, version):
    """Generates paths for the model file (.keras) and metadata file (.json)."""
    base_filename = get_model_filename(iteration, version)
    model_path = os.path.join(save_dir, base_filename + ".keras")
    metadata_path = os.path.join(save_dir, base_filename + "_meta.json")
    return model_path, metadata_path

def get_next_version_number(save_dir, iteration):
    """Finds the next available version number for a given iteration in the save directory."""
    os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists
    prefix = f"CONFORMERv{iteration}."
    existing_versions = [0] # Start with 0 in case no models exist
    for fname in os.listdir(save_dir):
        if fname.startswith(prefix) and (fname.endswith(".keras") or fname.endswith("_meta.json")):
            try:
                # Extract version number between prefix and extension/suffix
                version_part = fname[len(prefix):].split('.')[0].split('_')[0]
                if version_part.isdigit():
                    existing_versions.append(int(version_part))
            except (IndexError, ValueError):
                continue # Ignore files that don't match the expected format
    return max(existing_versions) + 1

def save_model_metadata(metadata_path, metadata):
    """Saves model metadata (like maxlen) to a JSON file."""
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved model metadata to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata to {metadata_path}: {e}")

def load_model_metadata(metadata_path):
    """Loads model metadata from a JSON file."""
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}")
        return None
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
        return metadata
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}")
        return None