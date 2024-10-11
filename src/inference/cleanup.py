# src/inference/cleanup.py
import logging
from src.preprocessing.audio_preprocessing import clear_directory

def clean_up(temp_file_storage, resampled_audio_dir):
    """
    Cleans up temporary files after inference.
    """
    logging.info("Cleaning up temporary files...")
    clear_directory(temp_file_storage)
    clear_directory(resampled_audio_dir)
