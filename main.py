# main.py
from PackageInstaller import check_and_install_packages
import os
import logging

# Define the log file path
log_file_path = "ribbitradar.log"

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path, mode="w"), logging.StreamHandler()],
)

#check_and_install_packages()

import tkinter as tk
from tkinter import messagebox
from gui import RibbitRadarGUI
import GetFFMPEG
import AudioPreprocessing
import sys
from ModelManager import (
    update_local_model,
    get_highest_local_model_version,
    get_latest_local_model_file,
)

logging.info("Application started")

# Google Drive folder ID and local paths for model management
MODEL_URL = "https://drive.google.com/uc?id=1lKOiBk1zrelbnQKHN8y0FzPy35cfM2Rz"
LOCAL_MODEL_DIR = "model"


def generate_unique_filename(directory, filename):
    """
    Generates a unique filename by appending a counter if a file with the same name already exists in the directory.

    Parameters:
    directory (str): The directory where the file will be saved.
    filename (str): The base name of the file (with or without extension).

    Returns:
    str: A unique filename that does not conflict with existing files in the directory.
    """
    base, extension = os.path.splitext(filename)
    if not extension:
        extension = ".xlsx"  # Default to .xlsx if no extension is provided
    counter = 1
    unique_filename = f"{base}{extension}"

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
    radr_threshold,
    raca_threshold,
    full_report,
    summary_report,
    custom_report,
    label_choice,
    prediction_mode,
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
    radr_threshold (float): Threshold value for RADR detection.
    raca_threshold (float): Threshold value for RACA detection.
    full_report (bool): Whether to generate a full report.
    summary_report (bool): Whether to generate a summary report.
    custom_report (dict): Dictionary specifying custom report options.
    label_choice (list): List of label choices for prediction.
    prediction_mode (str): Mode for making predictions, either 'Threshold' or 'Highest Score'.

    Returns:
    None
    """
    try:
        import inference

        update_progress("Inference started", 0, "Inference Started.")
        metadata_dict = AudioPreprocessing.extract_metadata_from_files_in_directory(
            input_dir, update_progress
        )
        AudioPreprocessing.Preprocess_audio(
            input_dir, temp_file_storage, resampled_audio_dir, update_progress
        )

        # Generate a unique output filename
        output_file = generate_unique_filename(output_dir, output_file)

        inference.run_inference(
            labels_path,
            checkpoint_path,
            resampled_audio_dir,
            model_version,
            output_dir,
            output_file,
            metadata_dict,
            update_progress,
            radr_threshold,
            raca_threshold,
            full_report,
            summary_report,
            custom_report,
            label_choice,
            prediction_mode,
        )

        messagebox.showinfo(
            "Success",
            f"Inference completed successfully. View your results at {output_dir}",
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
    3. Checks for and installs required packages.
    4. Updates the model weights if a new version is available.
    5. Defines paths for model, audio processing, and labels.
    6. Initializes the RibbitRadar GUI.
    7. Sets the inference callback for the GUI.
    8. Starts the Tkinter event loop.

    Returns:
    None
    """
    root = tk.Tk()
    root.title("RibbitRadar")

    # Display the splash screen
    splash = RibbitRadarGUI.create_splash_screen(root)
    splash.attributes("-topmost", True)  # Make splash screen the topmost window

    def update_splash_progress(message):
        """
        Updates the splash screen progress message.

        Parameters:
        message (str): The progress message to display on the splash screen.
        """
        splash.progress_label.config(text=message)
        splash.update_idletasks()

    # Setup FFmpeg environment
    ffmpeg_executable, _ = GetFFMPEG.get_ffmpeg_path()
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_executable)
    os.environ["FFMPEG_BINARY"] = ffmpeg_executable

    # Check and install required packages
    check_and_install_packages()

    # Update model weights if a new version is available
    update_local_model(LOCAL_MODEL_DIR, update_splash_progress)

    # Get the highest local model version
    model_version = get_highest_local_model_version(LOCAL_MODEL_DIR)

    # Determine the latest model file in the local directory
    model_path = get_latest_local_model_file(LOCAL_MODEL_DIR)

    base_path = os.path.dirname(os.path.abspath(__file__))

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
            enable_run_button=lambda: app.update_queue.put(  # FIXME
                (None, None, "Enable Run Button")
            ),
            radr_threshold=float(app.radr_threshold_entry.get()),
            raca_threshold=float(app.raca_threshold_entry.get()),
            # Pass the report options
            full_report=app.full_report_var.get(),
            summary_report=app.summary_report_var.get(),
            custom_report={
                "metadata": app.include_metadata_var.get(),
                "segment_scores": app.include_seg_scores_var.get(),
                "times_heard_radr": app.include_times_heard_radr_var.get(),
                "times_heard_raca": app.include_times_heard_raca_var.get(),
            },
            label_choice=app.label_choice_var.get().split(", "),
            prediction_mode=app.prediction_mode_var.get(),
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
    finally:
        logging.shutdown()
