import tkinter as tk
from tkinter import messagebox
from gui import RibbitRadarGUI
import os
import GetFFMPEG
import PackageInstaller
import AudioPreprocessing
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="ribbitradar.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger("").addHandler(console_handler)

logging.info("Application started")


import re
import logging
import os

import re
import logging
import os


def generate_unique_filename(directory, filename):
    # Define a list of known extensions
    known_extensions = [
        ".xlsx",
        ".xls",
        ".csv",
        ".txt",
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
    ]

    # Initialize base and extension
    base, extension = filename, ""

    # Check if the filename ends with a known extension
    for ext in known_extensions:
        if filename.lower().endswith(ext):
            base = filename[: -len(ext)]
            extension = ext
            break

    # If no known extension is found, default to .xlsx
    if not extension:
        extension = ".xlsx"

    counter = 1
    unique_filename = f"{base}{extension}"

    logging.debug(f"Checking existence of: {os.path.join(directory, unique_filename)}")

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}({counter}){extension}"
        logging.debug(f"File exists. Trying new name: {unique_filename}")
        counter += 1

    logging.info(f"Final unique filename: {unique_filename}")
    return unique_filename


def run_inference(
    input_dir,
    output_dir,
    output_file,
    temp_file_storage,
    resampled_audio_dir,
    labels_path,
    checkpoint_path,
    model_version,
    update_progress,
    enable_run_button,
):
    """
    Runs the complete inference process for detecting Rana Draytonii calls in audio files.

    This function performs the following steps:
    1. Extracts metadata from the input audio files.
    2. Preprocesses the audio files (splitting, resampling, converting to mono).
    3. Runs the inference process using the AST model.
    4. Displays success or error messages to the user.
    5. Updates the progress and logs the process.

    Parameters:
    input_dir (str): Path to the directory containing the input audio files.
    output_dir (str): Path to the directory where the output files will be saved.
    output_file (str): Name of the output file.
    temp_file_storage (str): Path to the temporary file storage directory.
    resampled_audio_dir (str): Path to the directory containing resampled audio files.
    labels_path (str): Path to the labels.csv file.
    checkpoint_path (str): Path to the model weights file.
    model_version (str): Version number of the model.
    update_progress (function): Callback function for updating progress.
    enable_run_button (function): Callback function to re-enable the Run button.

    Returns:
    None
    """
    try:
        import inference

        logging.info(f"Starting inference with output file: {output_file}")
        update_progress("Inference started", 0, "Inference Started.")
        metadata_dict = AudioPreprocessing.extract_metadata_from_files_in_directory(
            input_dir, update_progress
        )
        AudioPreprocessing.Preprocess_audio(
            input_dir, temp_file_storage, resampled_audio_dir, update_progress
        )

        # Generate a unique output filename
        output_file = generate_unique_filename(output_dir, output_file)
        logging.info(f"Unique output filename generated: {output_file}")

        inference.run_inference(
            labels_path,
            checkpoint_path,
            resampled_audio_dir,
            model_version,
            output_dir,
            output_file,
            metadata_dict,
            update_progress,
        )

        messagebox.showinfo(
            "Success",
            f"Inference completed successfully. View your results at {os.path.join(output_dir, output_file)}",
        )
        update_progress(
            "Inference completed.", 100, "Inference completed successfully."
        )

    except Exception as e:
        update_progress(None, None, f"Error: {str(e)}")
        logging.exception(f"Error in inference: {str(e)}")
        messagebox.showerror(
            "Error", f"An error occurred while running inference: {str(e)}"
        )
        enable_run_button()


def main():
    """
    Initializes and runs the RibbitRadar application.

    This function performs the following steps:
    1. Initializes the Tkinter root window and splash screen.
    2. Sets up the FFmpeg environment.
    3. Defines paths for model, audio processing, and labels.
    4. Initializes the RibbitRadar GUI.
    5. Sets the inference callback for the GUI.
    6. Starts the Tkinter event loop.

    Returns:
    None
    """
    root = tk.Tk()
    root.title("RibbitRadar")

    # Display the splash screen
    splash = RibbitRadarGUI.create_splash_screen(root)

    # Setup FFmpeg environment
    ffmpeg_executable, _ = GetFFMPEG.get_ffmpeg_path()
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_executable)
    os.environ["FFMPEG_BINARY"] = ffmpeg_executable

    # =============== Temporarily removed since no longer using remote model ==================

    # Check and install necessary packages
    # PackageInstaller.check_and_install_packages()

    # Define paths and update model
    # base_path = os.path.dirname(os.path.abspath(__file__))
    # local_version_file = os.path.normpath(os.path.join(base_path, 'model', 'ModelVersion.txt'))
    # model_output_dir = os.path.normpath(os.path.join(base_path, 'model'))
    # remote_version_file_id = '1yfgi1RgIyu9bVbp1cZInQNfowGA6W359'
    # PackageInstaller.check_and_update_model(local_version_file, remote_version_file_id, model_output_dir)

    # =================================================================================================

    # Function to determine the base path for accessing data files
    def resource_path(relative_path):
        """Get the absolute path to the resource, works for dev and for PyInstaller."""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = resource_path("model/best_audio_model_V2.pth")
    model_version = "2"

    # Define paths for audio processing
    Resampled_audio_path = os.path.normpath(
        os.path.join(base_path, "Rana_Draytonii_ML_Model", "ResampledAudio")
    )
    temp_file_storage = os.path.normpath(
        os.path.join(base_path, "Rana_Draytonii_ML_Model", "Temp_File_Storage")
    )
    labels_path = os.path.normpath(
        os.path.join(base_path, "Rana_Draytonii_ML_Model", "labels.csv")
    )
    # checkpoint_path, model_version = PackageInstaller.get_model_info(local_version_file, model_output_dir) removed since no longer using remote model

    # Now that initial setup is done, destroy the splash screen
    splash.destroy()

    # Initialize the main GUI
    app = RibbitRadarGUI(root, Resampled_audio_path)

    app.set_inference_callback(
        lambda: run_inference(
            os.path.normpath(app.input_folder_entry.get()),
            os.path.normpath(app.output_folder_entry.get()),
            os.path.normpath(app.output_file_entry.get()),
            temp_file_storage,
            Resampled_audio_path,
            labels_path,
            model_path,
            model_version,
            update_progress=lambda message=None, value=None, log_message=None: app.update_queue.put(
                (message, value, log_message)
            ),
            enable_run_button=lambda: app.update_queue.put(
                (None, None, "Enable Run Button")
            ),
        )
    )

    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    try:
        logging.info("Application starting...")
        main()
        logging.info("Application shutdown.")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
