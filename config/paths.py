import os

# Explicitly set the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to important directories
LOG_FILE_PATH = os.path.join(project_root, "ribbitradar.log")
LOCAL_MODEL_DIR = os.path.join(project_root, "src", "models")
CONFIG_DIR = os.path.join(project_root, "config")
TEMP_FILE_STORAGE = os.path.join(project_root, "processing", "temp_file_storage")
RESAMPLED_AUDIO_PATH = os.path.join(project_root, "processing", "resampled_audio")

# Paths to specific files
REQUIREMENTS_FILE = os.path.join(CONFIG_DIR, "requirements.txt")
LOCAL_MODEL_METADATA_FILE = os.path.join(CONFIG_DIR, "model_metadata.txt")

METADATA_URL = "https://drive.google.com/uc?id=1ry4-tguDnA1rFXFZ65KslO6lE2mL1LQv"

# Utility function to create directories if they don't exist
def ensure_directories_exist():
    os.makedirs(TEMP_FILE_STORAGE, exist_ok=True)
    os.makedirs(RESAMPLED_AUDIO_PATH, exist_ok=True)
