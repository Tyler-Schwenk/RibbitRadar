from .gui import RibbitRadarGUI
import tkinter as tk
from config.paths import RESAMPLED_AUDIO_PATH

def initialize_gui(log_file_path):
    """
    Initializes the main GUI window and splash screen for the RibbitRadar application.

    Returns:
        tuple: A tuple containing the root Tkinter window, the splash screen object, and the app (RibbitRadarGUI instance).
    """
    root = tk.Tk()
    root.title("RibbitRadar")
    
    # Create splash screen
    splash = RibbitRadarGUI.create_splash_screen(root)
    splash.attributes("-topmost", True)  # Make splash screen the topmost window
    
    # Create the main application GUI
    app = RibbitRadarGUI(root, resampled_audio_path=RESAMPLED_AUDIO_PATH, log_file_path=log_file_path)  # Provide the actual resampled audio path here
    
    return root, splash, app
