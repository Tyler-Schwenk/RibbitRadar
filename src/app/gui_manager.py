from .gui import RibbitRadarGUI
import tkinter as tk

def initialize_gui():
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
    app = RibbitRadarGUI(root, resampled_audio_path="processing/resampled_audio")  # Provide the actual resampled audio path here
    
    return root, splash, app
