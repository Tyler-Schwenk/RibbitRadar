from config import paths
import logging
from tkinter import messagebox
from src.preprocessing.audio_preprocessing import (
    extract_metadata_from_files_in_directory,
    split_all_audio_files,
    stereo_to_mono,
    clear_directory,
    resample_audio_files,
)

def preprocess_audio_pipeline(input_dir, progress_callback):
    """
    Full preprocessing pipeline to handle metadata extraction, splitting, resampling, etc.
    """
    try:
        logging.info("Starting audio preprocessing...")

        # Step 1: Extract metadata
        progress_callback("Step 1/4: Extracting metadata...", 5, "Extracting metadata...")
        metadata_dict = extract_metadata_from_files_in_directory(input_dir, progress_callback)

        # Step 2: Clear directories
        progress_callback("Step 2/4: Clearing temporary directories...", 10, "Clearing temp directories...")
        clear_directory(paths.TEMP_FILE_STORAGE)
        clear_directory(paths.RESAMPLED_AUDIO_PATH)

        # Step 3: Split and convert to mono
        progress_callback("Step 3/4: Splitting and converting to mono...", 50, "Splitting and converting to mono...")
        split_all_audio_files(input_dir, paths.TEMP_FILE_STORAGE, progress_callback)
        stereo_to_mono(paths.TEMP_FILE_STORAGE)

        # Step 4: Resample audio files
        progress_callback("Step 4/4: Resampling audio files...", 70, "Resampling audio files...")
        resample_audio_files(paths.TEMP_FILE_STORAGE, paths.RESAMPLED_AUDIO_PATH, progress_callback)

        progress_callback("Preprocessing complete.", 100, "Finished Preprocessing")
        return metadata_dict

    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        messagebox.showerror("Error", f"An error occurred during preprocessing: {str(e)}")
        return None
