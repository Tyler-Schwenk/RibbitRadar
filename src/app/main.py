import sys
import os
import subprocess
import logging

# Explicitly set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)

# Add 'src' and 'src/utilities' to the Python path
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'utilities'))

print("Current Python Path:", sys.path)


# Define the log file path
log_file_path = os.path.join(project_root, "ribbitradar.log")

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path, mode="w"), logging.StreamHandler()],
)

# Check and install required packages before any other imports
def install_requirements(requirements_file="config/requirements.txt"):
    try:
        # Check if the requirements file exists
        if os.path.exists(requirements_file):
            logging.info("Installing required packages from %s", requirements_file)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            logging.info("Package installation successful.")
        else:
            logging.error("Requirements file not found at %s", requirements_file)
            return False
    except subprocess.CalledProcessError as e:
        logging.error("Package installation failed with error: %s", e)
        return False

# Install dependencies without restarting the script
install_requirements()

# Now that packages are installed, import the rest of the modules
try:
    from src.utilities.package_installer import check_and_install_packages
    from src.app.gui_manager import initialize_gui
    from src.app.inference_runner import run_inference
    from src.utilities.setup_ffmpeg import setup_ffmpeg
    from src.preprocessing.preprocessing_manager import preprocess_audio_pipeline
    from src.app.model_manager import (
        update_local_model,
        get_highest_local_model_version,
        get_latest_local_model_file,
    )
    logging.info("Application started")
except ImportError as e:
    logging.error(f"Import failed: {e}")
    sys.exit(1)


# Google Drive folder ID and local paths for model management
MODEL_URL = "https://drive.google.com/uc?id=1lKOiBk1zrelbnQKHN8y0FzPy35cfM2Rz"
LOCAL_MODEL_DIR = os.path.join(project_root, "src", "models")


def generate_unique_filename(directory, filename):
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


def main():
    """
    Initializes and runs the RibbitRadar application.
    """
    # Initialize the GUI and splash screen
    root, splash, app = initialize_gui()

    def update_splash_progress(message):
        splash.progress_label.config(text=message)
        splash.update_idletasks()

    # Setup FFmpeg environment
    setup_ffmpeg()

    # Update model weights if a new version is available
    update_local_model(LOCAL_MODEL_DIR, update_splash_progress)

    # Get the highest local model version
    model_version = get_highest_local_model_version(LOCAL_MODEL_DIR)

    # Determine the latest model file in the local directory
    model_path = get_latest_local_model_file(LOCAL_MODEL_DIR)

    base_path = os.path.dirname(os.path.abspath(__file__))

    # Define paths for audio processing
    temp_file_storage = os.path.normpath(
        os.path.join(base_path, "processing", "temp_file_storage")
    )
    resampled_audio_path = os.path.normpath(
        os.path.join(base_path, "processing", "resampled_audio")
    )

    # Now that initial setup is done, destroy the splash screen
    splash.destroy()

    # Set the inference callback for when the user hits 'Run'
    app.set_inference_callback(
        lambda: run_inference(
            temp_file_storage=temp_file_storage,
            resampled_audio_path=resampled_audio_path,
            model_path=model_path,
            model_version=model_version,
            metadata_dict=preprocess_audio_pipeline(  # Preprocessing only after user input
                app.input_folder_entry.get(),
                temp_file_storage,
                resampled_audio_path,
                app.update_progress
            ),
            update_progress=app.update_progress,
            radr_threshold=float(app.radr_threshold_entry.get()), 
            raca_threshold=float(app.raca_threshold_entry.get()),  
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
            output_file=generate_unique_filename(app.output_folder_entry.get(), app.output_file_entry.get())
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
