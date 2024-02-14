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
logging.basicConfig(level=logging.DEBUG,
                    filename='ribbitradar.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# You can also add a handler to log to console at a different level, if desired
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler)

logging.info("Application started")


def run_inference(input_dir, output_dir, output_file, temp_file_storage, resampled_audio_dir, labels_path, checkpoint_path, model_version, update_progress, enable_run_button):
    try:
        import inference
        update_progress("Inference started", 0, "Inference Started.")
        metadata_dict = AudioPreprocessing.extract_metadata_from_files_in_directory(input_dir, update_progress)
        AudioPreprocessing.Preprocess_audio(input_dir, temp_file_storage, resampled_audio_dir, update_progress)
        inference.run_inference(labels_path, checkpoint_path, resampled_audio_dir, model_version, output_dir, output_file, metadata_dict, update_progress)

        messagebox.showinfo('Success', f'Inference completed successfully. View your results at {output_dir}')
        update_progress("Inference completed.", 100, "Inference completed successfully.")
    
    except Exception as e:
        update_progress(None, None, f"Error: {str(e)}")
        logging.exception(f"Error in inference: {str(e)}")
        messagebox.showerror('Error', f'An error occurred while running inference: {str(e)}')
        enable_run_button()

        
def main():
    root = tk.Tk()
    root.title('RibbitRadar')

    # Display the splash screen
    splash = RibbitRadarGUI.create_splash_screen(root)

    # Setup FFmpeg environment
    ffmpeg_executable, _ = GetFFMPEG.get_ffmpeg_path()
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_executable)
    os.environ["FFMPEG_BINARY"] = ffmpeg_executable

    # =============== Temporarily removed since no longer using remote model ==================

    # Check and install necessary packages
    #PackageInstaller.check_and_install_packages()

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
    model_path = resource_path('model/best_audio_model_V2.pth')
    model_version = "2"  


    # Define paths for audio processing
    Resampled_audio_path = os.path.normpath(os.path.join(base_path, 'Rana_Draytonii_ML_Model', 'ResampledAudio'))
    temp_file_storage = os.path.normpath(os.path.join(base_path, 'Rana_Draytonii_ML_Model', 'Temp_File_Storage'))
    labels_path = os.path.normpath(os.path.join(base_path, 'Rana_Draytonii_ML_Model', 'labels.csv'))
    #checkpoint_path, model_version = PackageInstaller.get_model_info(local_version_file, model_output_dir) removed since no longer using remote model

    # Now that initial setup is done, destroy the splash screen
    splash.destroy()

    # Initialize the main GUI
    app = RibbitRadarGUI(root, Resampled_audio_path)

    app.set_inference_callback(lambda: run_inference(
        os.path.normpath(app.input_folder_entry.get()),
        os.path.normpath(app.output_folder_entry.get()),
        os.path.normpath(app.output_file_entry.get()),
        temp_file_storage,
        Resampled_audio_path,
        labels_path,
        model_path,
        model_version,
        update_progress=lambda message=None, value=None, log_message=None: app.update_queue.put((message, value, log_message)),
        enable_run_button=lambda: app.update_queue.put((None, None, "Enable Run Button"))
    ))

    # Start the GUI event loop
    root.mainloop()

if __name__ == '__main__':
    try:
        logging.info("Application starting...")
        main()
        logging.info("Application shutdown.")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)

