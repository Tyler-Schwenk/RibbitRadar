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
    resampled_audio_path (str): Path to the directory where resampled audio files will be saved.
    inference_callback (function): A callback function for running inference, initialized as None.
    update_queue (queue.Queue): A queue to manage the update of GUI elements.
    """

    def __init__(self, root, resampled_audio_path, log_file_path):
        """
        Initialize the GUI and log file path.

        Parameters:
        root (tk.Tk): The main window of the application.
        resampled_audio_path (str): Path to the directory where resampled audio files will be saved.
        log_file_path (str): Path to the log file.
        """
        self.root = root
        self.resampled_audio_path = resampled_audio_path
        self.log_file_path = log_file_path  # Store the log file path
        self.inference_callback = None
        self.create_widgets()
        self.update_queue = queue.Queue()
        self.check_queue()

    def view_log_file(self):
        """
        Opens the log file using the default application for .log files.
        If the file does not exist, display an error message.
        """
        if not os.path.exists(self.log_file_path):
            messagebox.showerror("Error", "Log file not found.")
            return

        try:
            if os.name == "nt":  # Windows
                os.startfile(self.log_file_path)
            elif os.name == "posix":  # macOS and Linux
                subprocess.run(["open", self.log_file_path], check=True)
            else:
                messagebox.showerror(
                    "Error", f"Unsupported OS: Unable to open log file."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open log file: {str(e)}")

    def set_inference_callback(self, callback):
        """
        Sets the callback function for running inference.

        Parameters:
        callback (function): The callback function to set, which will be called when "Run Inference" is clicked.
        """
        self.inference_callback = callback

    def check_queue(self):
        """
        Periodically checks the update queue for messages and handles the GUI updates.
        """
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
        Creates a splash screen to display while the application is loading.

        Parameters:
        root (tk.Tk): The root window of the application.

        Returns:
        tk.Toplevel: The splash screen window.
        """
        splash = tk.Toplevel(root)
        splash.geometry("300x200")
        splash.title("RibbitRadar Loading...")
        splash_label = tk.Label(splash, text="Loading...", font=("Helvetica", 16))
        splash_label.pack(expand=True)

        # Add a progress label to display updates
        progress_label = tk.Label(splash, text="", font=("Helvetica", 12))
        progress_label.pack()

        splash.progress_label = progress_label  # Store the label for later updates

        return splash

    # In gui.py, inside the RibbitRadarGUI class
    def enable_run_button(self):
        """
        Enables the 'Run Inference' button, allowing users to run the inference process again.
        """
        self.run_button.config(state="normal")

    def validate_paths(self):
        """
        Validates the input and output folder paths and the output file name.

        Returns:
        bool: True if all paths and file names are valid, otherwise False.
        """
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

        # Define the label choices as strings
        label_choices = ["RADR, Negative", "RACA, Negative", "RADR, RACA, Negative"]

        # Setup the combobox with these string values
        self.label_choice_var = tk.StringVar()
        self.label_choice_combobox = ttk.Combobox(
            main_frame,
            textvariable=self.label_choice_var,
            values=label_choices,  # Display the choices as strings
        )
        self.label_choice_combobox.grid(column=1, row=1, padx=10, pady=10)
        self.label_choice_combobox.current(0)  # Default to 'RADR, Negative'

        # Prediction Mode Dropdown
        ttk.Label(main_frame, text="Prediction Mode:").grid(
            column=0, row=12, padx=10, pady=10
        )
        self.prediction_mode_var = tk.StringVar()
        prediction_modes = ["Threshold", "Highest Score"]
        self.prediction_mode_combobox = ttk.Combobox(
            main_frame, textvariable=self.prediction_mode_var, values=prediction_modes
        )
        self.prediction_mode_combobox.grid(column=1, row=12, padx=10, pady=10)
        self.prediction_mode_combobox.current(0)  # Default to 'Threshold'

        # Bind the combobox selection event to toggle the threshold inputs
        self.prediction_mode_combobox.bind(
            "<<ComboboxSelected>>", self.toggle_threshold_options
        )

        # RADR Threshold
        self.radr_threshold_label = ttk.Label(main_frame, text="RADR Threshold (0-1):")
        self.radr_threshold_label.grid(column=0, row=11, padx=10, pady=10)
        self.radr_threshold_entry = ttk.Entry(main_frame)
        self.radr_threshold_entry.grid(column=1, row=11, padx=10, pady=10)
        self.radr_threshold_entry.insert(0, "0.90")  # Default value

        # RACA Threshold
        self.raca_threshold_label = ttk.Label(main_frame, text="RACA Threshold (0-1):")
        self.raca_threshold_label.grid(column=0, row=10, padx=10, pady=10)
        self.raca_threshold_entry = ttk.Entry(main_frame)
        self.raca_threshold_entry.grid(column=1, row=10, padx=10, pady=10)
        self.raca_threshold_entry.insert(0, "0.85")  # Default value

        # Add report selection section
        report_frame = ttk.LabelFrame(self.root, text="Report Options", padding=10)
        report_frame.grid(column=0, row=10, padx=10, pady=10)

        # Report checkboxes
        self.full_report_var = tk.BooleanVar(value=True)  # Full report checkbox
        self.summary_report_var = tk.BooleanVar(value=False)  # Summary report checkbox
        self.custom_report_var = tk.BooleanVar(value=False)  # Custom report checkbox

        full_report_cb = ttk.Checkbutton(
            report_frame, text="Full Report", variable=self.full_report_var
        )
        summary_report_cb = ttk.Checkbutton(
            report_frame, text="Summary Report", variable=self.summary_report_var
        )
        custom_report_cb = ttk.Checkbutton(
            report_frame,
            text="Custom Report",
            variable=self.custom_report_var,
            command=self.toggle_custom_options,
        )

        full_report_cb.grid(column=0, row=0, sticky="w")
        summary_report_cb.grid(column=0, row=1, sticky="w")
        custom_report_cb.grid(column=0, row=2, sticky="w")

        # Custom report options (initially hidden)
        self.custom_options_frame = ttk.Frame(report_frame)
        self.custom_options_frame.grid(column=1, row=2, sticky="w", padx=10)
        self.custom_options_frame.grid_remove()  # Hide by default

        self.include_metadata_var = tk.BooleanVar(value=True)
        self.include_seg_scores_var = tk.BooleanVar(value=True)
        self.include_times_heard_radr_var = tk.BooleanVar(value=True)
        self.include_times_heard_raca_var = tk.BooleanVar(value=True)

        include_metadata_cb = ttk.Checkbutton(
            self.custom_options_frame,
            text="Include Metadata",
            variable=self.include_metadata_var,
        )
        include_seg_scores_cb = ttk.Checkbutton(
            self.custom_options_frame,
            text="Include all Segment Scores",
            variable=self.include_seg_scores_var,
        )
        include_times_heard_radr_cb = ttk.Checkbutton(
            self.custom_options_frame,
            text="Include Times Heard RADR",
            variable=self.include_times_heard_radr_var,
        )
        include_times_heard_raca_cb = ttk.Checkbutton(
            self.custom_options_frame,
            text="Include Times Heard RACA",
            variable=self.include_times_heard_raca_var,
        )

        include_metadata_cb.grid(column=0, row=0, sticky="w")
        include_seg_scores_cb.grid(column=0, row=1, sticky="w")
        include_times_heard_radr_cb.grid(column=0, row=3, sticky="w")
        include_times_heard_raca_cb.grid(column=0, row=4, sticky="w")

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

        # Add a "View Log File" button
        view_log_button = ttk.Button(
            self.root,
            text="View Log File",
            command=self.view_log_file  # Call the method to open the log file
        )
        view_log_button.grid(column=0, row=9, columnspan=2, pady=10)

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
        1. This will be updated before release
        2. Save the Frogs!!
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

    def toggle_custom_options(self):
        """
        Show or hide the custom report options depending on whether 'Custom Report' is selected.
        """
        if self.custom_report_var.get():
            self.custom_options_frame.grid()  # Show
        else:
            self.custom_options_frame.grid_remove()  # Hide

    def toggle_threshold_options(self, event=None):
        """
        Show or hide the threshold options based on the selected prediction mode.
        Only shows thresholds if 'Threshold' mode is selected.
        """
        if self.prediction_mode_var.get() == "Threshold":
            self.radr_threshold_label.grid()  # Show RADR Threshold
            self.radr_threshold_entry.grid()
            self.raca_threshold_label.grid()  # Show RACA Threshold
            self.raca_threshold_entry.grid()
        else:
            self.radr_threshold_label.grid_remove()  # Hide RADR Threshold
            self.radr_threshold_entry.grid_remove()
            self.raca_threshold_label.grid_remove()  # Hide RACA Threshold
            self.raca_threshold_entry.grid_remove()

    def update_log(self, message):
        """
        Updates the log area with a new message.

        Parameters:
        message (str): The message to be displayed in the log area.
        """
        self.log_area.config(state="normal")  # Enable editing of the text area
        self.log_area.insert(tk.END, message + "\n")  # Append message
        self.log_area.yview(tk.END)  # Auto-scroll to the bottom
        self.log_area.config(state="disabled")  # Disable editing of the text area

    def update_progress(self, message=None, value=None, log_message=None):
        """
        Updates the progress bar and optionally the status label and log.

        Parameters:
        message (str): The message to display in the status label (optional).
        value (int or float): The progress bar value (optional).
        log_message (str): The message to log in the log area (optional).
        """
        if message is not None:
            self.status_label.config(text=message)
        if value is not None:
            self.progress_bar["value"] = value
        if log_message is not None:
            self.update_log(log_message)
        self.root.update_idletasks()  # Update the GUI

    def select_directory(self, entry_widget):
        """
        Opens a dialog to select a directory and updates the corresponding entry widget.

        Parameters:
        entry_widget (ttk.Entry): The entry widget to be updated with the selected directory path.
        """
        selected_directory = filedialog.askdirectory()
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, selected_directory)

    def run_inference(self):
        """
        Triggers the inference process in a separate thread after validating paths.
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
