import os
import gdown
import logging
from config import paths


def extract_version_from_filename(filename):
    """
    Extract the version number from the model filename.

    Args:
        filename (str): The model filename.

    Returns:
        int: The version number extracted from the filename.
    """
    try:
        version_str = filename.split("_V")[1].split(".")[0]
        return int(version_str)
    except (IndexError, ValueError):
        return 0  # Default to 0 if the version cannot be extracted


def download_model_metadata(metadata_url, local_metadata_path):
    """
    Download the metadata file using gdown.

    Args:
        metadata_url (str): The Google Drive URL of the metadata file.
        local_metadata_path (str): The local path to store the downloaded metadata.

    Returns:
        str: The path to the downloaded metadata file, or None if the download failed.
    """
    try:
        gdown.download(metadata_url, local_metadata_path, quiet=False)
        if os.path.exists(local_metadata_path):
            return local_metadata_path
        else:
            raise Exception("Metadata download failed")
    except Exception as e:
        logging.error(f"Failed to download metadata: {e}")
        return None


def parse_model_metadata(metadata_path):
    """
    Parse the metadata file to extract model version and file_id.

    Args:
        metadata_path (str): The local path to the metadata file.

    Returns:
        dict: A dictionary containing the version and file_id.
    """
    metadata = {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":")
                metadata[key.strip()] = value.strip()
        return metadata
    except Exception as e:
        logging.error(f"Failed to parse metadata: {e}")
        return None


def download_and_update_model(
    model_url, local_model_dir, new_version, current_version, progress_callback=None
):
    """
    Download the latest model from Google Drive if it's a newer version, handle network interruptions,
    and delete old model files.

    Args:
        model_url (str): The Google Drive URL of the model file.
        local_model_dir (str): The local directory to store the model file.
        current_version (int): The version number of the currently installed model.
        new_version (int): The version number of the new model.
        progress_callback (function, optional): Function to update progress, if provided.

    Returns:
        str: The path to the downloaded model file, or None if no new model was downloaded.
    """
    if new_version <= current_version:
        logging.info(f"Model is up to date. Skipping download.")
        if progress_callback:
            progress_callback("Model is up to date. Skipping download.")
        return None

    # Extract the file ID from the URL
    file_id = model_url.split("id=")[1]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if progress_callback:
        progress_callback("Starting model download...")

    try:
        # Perform the download using gdown
        gdown_response = gdown.download(download_url, quiet=False, fuzzy=True)

        if gdown_response:
            downloaded_file_name = os.path.basename(gdown_response)
            local_model_path = os.path.join(local_model_dir, downloaded_file_name)
            os.rename(gdown_response, local_model_path)

            if progress_callback:
                progress_callback("Download complete. Cleaning up old models...")

            # Delete old models except the newly downloaded one
            model_files = [
                f
                for f in os.listdir(local_model_dir)
                if f.startswith("best_audio_model_V")
            ]
            for model_file in model_files:
                old_model_path = os.path.join(local_model_dir, model_file)
                if old_model_path != local_model_path:
                    os.remove(old_model_path)
                    logging.info(f"Deleted old model file: {old_model_path}")

            logging.info(f"Downloaded the model to {local_model_path}")
            if progress_callback:
                progress_callback("Model update completed.")
            return local_model_path
        else:
            raise Exception("Download failed")

    except Exception as e:
        logging.error(f"Failed to download the model: {e}")
        if progress_callback:
            progress_callback("Failed to download model.")
        return None


def get_latest_local_model_version(local_model_dir):
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
    highest_version = max(extract_version_from_filename(f) for f in model_files)
    return highest_version


def get_latest_model_file_path(model_dir):
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
    latest_model_file = max(model_files, key=lambda x: extract_version_from_filename(x))
    return os.path.join(model_dir, latest_model_file)


def update_local_model(local_model_dir, progress_callback=None):
    """
    Update the local model by downloading the latest version if it's newer.

    Args:
        local_model_dir (str): The directory to save the downloaded model.
        progress_callback (function, optional): Function to update progress, if provided.

    Returns:
        str: The path to the downloaded model file, or None if no new model was downloaded.
    """
    logging.info("Downloading model metadata...")
    local_metadata_path = "model_metadata.txt"
    metadata_file = download_model_metadata(paths.METADATA_URL, local_metadata_path)

    if metadata_file is None:
        logging.error("Failed to download metadata. Aborting update.")
        return None

    metadata = parse_model_metadata(metadata_file)
    if metadata is None:
        logging.error("Failed to parse metadata. Aborting update.")
        return None

    current_version = get_latest_local_model_version(local_model_dir)
    new_version = int(metadata.get("version", 0))

    # Proceed to download the model if a newer version is available
    model_url = f"https://drive.google.com/uc?id={metadata['file_id']}"
    return download_and_update_model(
        model_url, local_model_dir, new_version, current_version, progress_callback
    )
