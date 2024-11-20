import sys
import os
import logging

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)

# Add 'src' to the Python path
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.utilities.package_installer import check_and_install_packages
from config.paths import REQUIREMENTS_FILE

# Install required packages before importing anything else
check_and_install_packages(REQUIREMENTS_FILE)

from src.utilities.util import generate_unique_filename
from src.utilities.logging_setup import setup_logging
from config.paths import LOCAL_MODEL_DIR
from src.utilities.setup_ffmpeg import setup_ffmpeg
from src.app.gui_manager import initialize_gui
from src.inference.inference_pipeline import start_inference_pipeline
from src.app.model_manager import (
    update_local_model,
    get_latest_local_model_version,
    get_latest_model_file_path,
)

def main():
    """
    Initializes and runs the RibbitRadar application.
    """
    # Initialize logging
    log_file_path = setup_logging()
    logging.info("Application starting...")
    logging.info(f"Log file location: {log_file_path}")

    try:
        # Initialize the GUI and splash screen
        root, splash, app = initialize_gui(log_file_path)

        def update_splash_progress(message):
            splash.progress_label.config(text=message)
            splash.update_idletasks()

        # Setup FFmpeg environment
        setup_ffmpeg()

        # Update model weights if a new version is available
        update_local_model(LOCAL_MODEL_DIR, update_splash_progress)

        # Get the highest local model version
        model_version = get_latest_local_model_version(LOCAL_MODEL_DIR)

        # Determine the latest model file in the local directory
        model_path = get_latest_model_file_path(LOCAL_MODEL_DIR)

        # Now that initial setup is done, destroy the splash screen
        splash.destroy()

        # Set the inference callback for when the user hits 'Run'
        app.set_inference_callback(
            lambda: start_inference_pipeline(
                model_path=model_path,
                model_version=model_version,
                output_dir=app.output_folder_entry.get(),
                output_file=generate_unique_filename(app.output_file_entry.get()),
                input_dir=app.input_folder_entry.get(),
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
            )
        )

        # Start the GUI event loop
        root.mainloop()

    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        # Optional: Display error dialog in the GUI
        from tkinter import messagebox
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}")
    finally:
        logging.info("Application shutdown.")
        logging.shutdown()


if __name__ == "__main__":
    try:
        main()
        logging.info("Application shutdown.")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        logging.shutdown()
