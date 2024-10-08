import os
import logging
from src.preprocessing import audio_preprocessing 
from tkinter import messagebox

def preprocess_audio_pipeline(input_dir, temp_file_storage, resampled_audio_dir, progress_callback):
    """
    Full preprocessing pipeline to handle metadata extraction, splitting, resampling, etc.
    """
    try:
        # Extract metadata
        progress_callback("Preprocessing Step 1/5: Extracting metadata...", 5, "Extracting metadata...")
        metadata_dict = audio_preprocessing.extract_metadata_from_files_in_directory(input_dir, progress_callback)

        # Split audio files
        progress_callback("Preprocessing Step 2/5: Splitting audio files...", 10, "Splitting audio files...")
        audio_preprocessing.split_all_audio_files(input_dir, temp_file_storage, progress_callback)

        # Resample and convert audio files
        progress_callback("Preprocessing Step 3/5: Converting to mono...", 20, "Converting to mono...")
        audio_preprocessing.stereo_to_mono(temp_file_storage)

        progress_callback("Preprocessing Step 4/5: Resampling audio...", 50, "Resampling audio...")
        audio_preprocessing.preprocess_audio(input_dir, temp_file_storage, resampled_audio_dir, progress_callback)

        progress_callback("Preprocessing complete.", 100, "Finished Preprocessing")
        return metadata_dict

    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        messagebox.showerror("Error", f"An error occurred during preprocessing: {str(e)}")
        return None
