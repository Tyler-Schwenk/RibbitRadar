import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import os
import threading
import queue


class RibbitRadarGUI:
    """
    A class to represent the GUI of RibbitRadar application.

    Attributes:
    root (tk.Tk): The main window of the application.
    preprocess_audio_callback (function): A callback function for preprocessing audio.
    """

    def __init__(self, root, resampled_audio_path):
        """
        Constructs all the necessary attributes for the RibbitRadarGUI object.

        Parameters:
        root (tk.Tk): The main window of the application.
        resampled_audio_path (str): Path to the directory where resampled audio files will be saved.
        """
        self.root = root
        self.resampled_audio_path = resampled_audio_path
        self.inference_callback = None  # Initialize as None
        self.create_widgets()
        self.update_queue = queue.Queue()
        self.check_queue()

    def set_inference_callback(self, callback):
        """
        Sets the callback function for running inference.

        Parameters:
        callback (function): The callback function to set, which will be called when "Run Inference" is clicked
        """
        self.inference_callback = callback

    def check_queue(self):
        try:
            while not self.update_queue.empty():
                message, value, log_message = self.update_queue.get_nowait()
                # Handle GUI updates based on the message content
                if message == "Enable Run Button":
                    self.enable_run_button()
                else:
                    self.update_progress(message, value, log_message)
        finally:
            self.root.after(100, self.check_queue)

    @staticmethod
    def create_splash_screen(root):
        """
        Creates a splash screen window.

        Parameters:
        root (tk.Tk): The root window of the application.

        Returns:
        tk.Toplevel: The splash screen window.
        """
        splash = tk.Toplevel(root)
        splash.title("Loading")
        splash.geometry("300x100")
        tk.Label(splash, text="Starting RibbitRadar, please wait...").pack(pady=20)
        splash.update()
        return splash

    # In gui.py, inside the RibbitRadarGUI class
    def enable_run_button(self):
        self.run_button.config(state="normal")

    def validate_paths(self):
        # Normalize and trim paths
        input_folder = os.path.normpath(self.input_folder_entry.get().strip())
        output_folder = os.path.normpath(self.output_folder_entry.get().strip())
        output_file = self.output_file_entry.get().strip()

        # Validate Input Folder Path
        if not os.path.isdir(input_folder):
            messagebox.showerror("Invalid Path", "The input folder path is invalid.")
            return False

        # Validate Output Folder Path
        if not os.path.isdir(output_folder):
            messagebox.showerror("Invalid Path", "The output folder path is invalid.")
            return False

        # Validate Output File Name (ensure it's not empty and doesn't contain invalid characters)
        if not output_file or any(char in output_file for char in r'\/:*?"<>|'):
            messagebox.showerror(
                "Invalid Name", "Please enter a valid output file name."
            )
            return False

        return True

    def create_widgets(self):
        """
        Creates and arranges the widgets in the GUI.
        """
        # Main buttons and progress updates
        main_frame = ttk.Frame(self.root)
        main_frame.grid(column=0, row=0, sticky="nwse")

        # Instructions for using the application
        instruction_frame = ttk.Frame(self.root)
        instruction_frame.grid(column=1, row=0, sticky="n", padx=10)

        # Input Folder
        ttk.Label(main_frame, text="Input Folder:").grid(
            column=0, row=4, padx=10, pady=10
        )
        self.input_folder_entry = ttk.Entry(main_frame)
        self.input_folder_entry.grid(column=1, row=4, padx=10, pady=10)
        input_folder_button = ttk.Button(
            main_frame,
            text="Browse",
            command=lambda: self.select_directory(self.input_folder_entry),
        )
        input_folder_button.grid(column=2, row=4, padx=10, pady=10)

        # Output File Name
        ttk.Label(main_frame, text="Unique Output File Name:").grid(
            column=0, row=2, padx=10, pady=10
        )
        self.output_file_entry = ttk.Entry(main_frame)
        self.output_file_entry.grid(column=1, row=2, padx=10, pady=10)

        # Output Folder
        ttk.Label(main_frame, text="Output Location:").grid(
            column=0, row=3, padx=10, pady=10
        )
        self.output_folder_entry = ttk.Entry(main_frame)
        self.output_folder_entry.grid(column=1, row=3, padx=10, pady=10)
        base_path_button = ttk.Button(
            main_frame,
            text="Browse",
            command=lambda: self.select_directory(self.output_folder_entry),
        )
        base_path_button.grid(column=2, row=3, padx=10, pady=10)

        # Run Button
        self.run_button = ttk.Button(
            self.root, text="Run Inference", command=self.run_inference
        )
        self.run_button.grid(column=0, row=6, columnspan=2, pady=20)

        # Add a log text area with scrollbar
        self.log_area = scrolledtext.ScrolledText(
            main_frame, height=10, state="disabled"
        )
        self.log_area.grid(column=0, row=9, columnspan=3, padx=10, pady=10)

        # Add a status label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(column=0, row=7, columnspan=3)

        # Add a progress bar
        self.progress_bar = ttk.Progressbar(
            main_frame, orient="horizontal", length=300, mode="determinate"
        )
        self.progress_bar.grid(column=0, row=8, columnspan=3, pady=10)

        # Instruction Panel
        instruction_label = ttk.Label(
            instruction_frame, text="Instructions", font=("Helvetica", 16, "bold")
        )
        instruction_label.grid(column=0, row=0, sticky="nw", pady=(0, 10))

        instruction_text = tk.Text(
            instruction_frame,
            height=25,
            width=50,
            wrap="word",
            state="disabled",
            bg=self.root.cget("bg"),
            relief="flat",
        )
        instruction_text.grid(column=0, row=1, sticky="nw")

        # Adding instructions
        instructions = """
        How to Use RibbitRadar:
        ------------------------
        1. Select the Input Folder containing your audio files.
        2. Enter a unique name for the Output File.
        3. Choose the Output Location to save the results.
        4. Click 'Run Inference' to start processing.
        """
        instruction_text.config(state="normal")
        instruction_text.insert("end", instructions)
        instruction_text.config(state="disabled")

        # Making the instruction text read-only and scrollable
        instruction_text_scrollbar = ttk.Scrollbar(
            instruction_frame, orient="vertical", command=instruction_text.yview
        )
        instruction_text_scrollbar.grid(column=1, row=1, sticky="ns")
        instruction_text["yscrollcommand"] = instruction_text_scrollbar.set

    def update_log(self, message):
        """Update the log area with new messages."""
        self.log_area.config(state="normal")  # Enable editing of the text area
        self.log_area.insert(tk.END, message + "\n")  # Append message
        self.log_area.yview(tk.END)  # Auto-scroll to the bottom
        self.log_area.config(state="disabled")  # Disable editing of the text area

    def update_progress(self, message=None, value=None, log_message=None):
        """Update the progress bar and optionally the status label and log message."""
        if message is not None:
            self.status_label.config(text=message)
        if value is not None:
            self.progress_bar["value"] = value
        if log_message is not None:
            self.update_log(log_message)
        self.root.update_idletasks()  # Update the GUI

    def select_directory(self, entry_widget):
        """
        Opens a dialog to select a directory and updates the entry widget with the chosen path.

        Parameters:
        entry_widget (ttk.Entry): The entry widget to update with the selected directory.
        """
        selected_directory = filedialog.askdirectory()
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, selected_directory)

    def run_inference(self):
        """
        Triggers the inference process using the provided callback in a separate thread.
        """
        if not self.validate_paths():
            return  # Stop the process if paths are invalid
        if self.inference_callback is not None:
            try:
                # Disable the Run button to prevent multiple clicks
                self.run_button.config(state="disabled")
                # Run the inference in a separate thread
                inference_thread = threading.Thread(
                    target=lambda: self.inference_callback()
                )
                inference_thread.start()
            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred while running inference: {str(e)}"
                )
                # Ensure the run button is enabled in case of error
                self.run_button.config(state="normal")
        else:
            messagebox.showerror("Error", "Inference callback is not set.")
            self.run_button.config(state="normal")  # Ensure the button is re-enabled

    def set_inference_callback(self, callback):
        """
        Sets the callback function for running inference.

        Parameters:
        callback (function): The callback function to set.
        """
        self.inference_callback = callback
