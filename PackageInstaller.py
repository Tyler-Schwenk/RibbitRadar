import subprocess
import sys
import os
import requests
import logging

def check_and_install_packages():
    """Install required Python packages with error handling."""
    packages_to_install = ['torchaudio', 'timm==0.4.5', 'wget', 'pydub', 'requests', 'soundfile', 'pandas', 'openpyxl', 'gdown']
    
    for package in packages_to_install:
        try:
            # Check if package is already installed
            __import__(package.split('==')[0])
        except ImportError:
            try:
                # Attempt to install the package
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logging.info("Successfully installed %s", package)
            except subprocess.CalledProcessError as e:
                logging.error("Failed to install %s. Error: %s", package, e)
            except Exception as e:
                logging.error("An unexpected error occurred while installing %s. Error: %s", package, e)


def get_remote_version_info(file_id):
    """Retrieve version information from a remote file."""
    version_url = f'https://drive.google.com/uc?id={file_id}'
    try:
        response = requests.get(version_url)
        if response.status_code == 200:
            logging.info("Successfully fetched remote version info.")
            return response.text.strip()
        else:
            logging.error("Failed to fetch remote version info. Status code: %s", response.status_code)
    except Exception as e:
        logging.error("Unexpected error fetching remote version info. Error: %s", e)


def download_model_weights(file_id, output_path):
    """Download model weights from a remote file with validation."""
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        logging.debug(f"Downloading model weights from {url} to {output_path}")
        subprocess.check_call([sys.executable, '-m', 'gdown', url, '-O', output_path])
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Model downloaded successfully and saved to {output_path}")
        else:
            raise Exception("Downloaded model file is not valid.")
    except Exception as e:
        logging.error(f"Failed to download model weights. Error: {e}")
        # Consider additional error handling or retry logic here


def check_and_update_model(local_version_path, remote_version_file_id, model_output_dir):
    """
    Check for model updates and download the new version if available.

    This function compares the version of the model stored locally with the version available remotely. If the remote 
    version is newer, it downloads and updates the local model. The local model version is stored in a file, and the 
    function updates this file with the new version number after a successful update.

    Parameters:
    local_version_path (str): The file path where the local model version is stored. This file should contain a single string representing the version number.
    remote_version_file_id (str): A unique identifier for the remote file that contains information about the latest model version. The information is expected to be in the format '<file_id>,<version_number>'.
    model_output_dir (str): The directory where the updated model weights should be saved.
    """
    try:
        remote_model_info = get_remote_version_info(remote_version_file_id)

        if not os.path.exists(local_version_path):
            local_model_version = None
        else:
            with open(local_version_path, 'r') as file:
                local_model_version = file.read().strip()
        if remote_model_info:
            remote_model_file_id, remote_model_version = remote_model_info.split(',')
            logging.info(f"Remote model version: {remote_model_version}, Local model version: {local_model_version}")

            if local_model_version != remote_model_version:
                logging.info("New model version detected. Downloading...")
                model_output_path = os.path.join(model_output_dir, f'model_{remote_model_version}.pth')
                download_model_weights(remote_model_file_id, model_output_path)
                with open(local_version_path, 'w') as file:
                    file.write(remote_model_version)
            else:
                logging.info("Model is up to date.")
        else:
            logging.warning("Could not fetch remote model version info. Skipping model update.")
    except Exception as e:
        logging.error(f"Error updating model: {e}", exc_info=True)

def get_model_info(local_version_path, model_output_dir):
    """
    Retrieves the path to the local weights file and the current model version.

    Parameters:
    local_version_path (str): The file path where the local model version is stored.
    model_output_dir (str): The directory where the model weights are saved.

    Returns:
    tuple: A tuple containing the path to the model weights file and the model version.
    """

    # Read the local model version
    with open(local_version_path, 'r') as file:
        local_model_version = file.read().strip()

    # Construct the path to the model weights file
    model_weights_file = os.path.join(model_output_dir, f'model_{local_model_version}.pth')

    if not os.path.exists(model_weights_file):
        logging.warning(f"Model weights file not found for version {local_model_version}.")
        return None, None

    return model_weights_file, local_model_version


