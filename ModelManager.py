import os
import gdown
import logging

import os
import gdown
import logging

def download_latest_model(model_url, local_model_dir):
    """
    Download the latest model from Google Drive and delete old model files.

    Args:
        model_url (str): The Google Drive URL of the model file.
        local_model_dir (str): The local directory to store the model file.

    Returns:
        str: The path to the downloaded model file.
    """
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    # Extract the file ID from the URL
    file_id = model_url.split('id=')[1]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Download the model file to get the original file name
    gdown_response = gdown.download(download_url, quiet=False, fuzzy=True)

    if gdown_response:
        downloaded_file_name = os.path.basename(gdown_response)

        # Move the downloaded file to the model directory
        local_model_path = os.path.join(local_model_dir, downloaded_file_name)
        os.rename(gdown_response, local_model_path)

        # Find and delete the old model files
        model_files = [f for f in os.listdir(local_model_dir) if f.startswith('best_audio_model_V')]
        for model_file in model_files:
            old_model_path = os.path.join(local_model_dir, model_file)
            if old_model_path != local_model_path:
                os.remove(old_model_path)
                logging.info(f"Deleted old model file: {old_model_path}")

        logging.info(f"Downloaded the model to {local_model_path}")
        return local_model_path
    else:
        logging.error("Failed to download the model.")
        return None


def get_highest_local_model_version(local_model_dir):
    """
    Get the highest model version number in the local directory.

    Args:
        local_model_dir (str): The directory containing the model files.

    Returns:
        int: The highest model version number found in the directory. Returns 0 if no models are found.
    """
    files = os.listdir(local_model_dir)
    model_files = [f for f in files if f.startswith("best_audio_model_V")]
    if not model_files:
        return 0
    highest_version = max(int(f.split("_V")[1].split(".")[0]) for f in model_files)
    return highest_version

def get_latest_local_model_file(model_dir):
    """
    Get the path to the latest model file in the local directory.

    Args:
        model_dir (str): The directory containing the model files.

    Returns:
        str: The path to the latest model file.

    Raises:
        FileNotFoundError: If no model files are found in the directory.
    """
    files = os.listdir(model_dir)
    model_files = [f for f in files if f.startswith("best_audio_model_V")]
    if not model_files:
        raise FileNotFoundError("No model files found in the directory.")
    latest_model_file = max(
        model_files, key=lambda x: int(x.split("_V")[1].split(".")[0])
    )
    return os.path.join(model_dir, latest_model_file)

def update_local_model(model_url, local_model_dir):
    """
    Update the local model by downloading the latest version from a specified URL.

    Args:
        model_url (str): The URL to download the model from.
        local_model_dir (str): The directory to save the downloaded model.

    Returns:
        str: The path to the downloaded model file.
    """
    logging.info("Checking for new model version...")
    local_model_path = download_latest_model(model_url, local_model_dir)
    return local_model_path
