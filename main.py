import PackageInstaller
import AudioProcessing


def select_directory(entry_widget):
    # Open a directory selection dialog and update the entry widget with the selected path
    selected_directory = filedialog.askdirectory()  # Open the dialog
    entry_widget.delete(0, tk.END)  # Clear any existing text in the entry widget
    entry_widget.insert(0, selected_directory)  # Insert the selected path

# Function to run the inference calling other functions
def run_inference():
  try:
    input_dir = input_folder_entry.get()
    output_dir = os.path.join(base_path_entry.get(), "output")
    resampled_audio_dir = os.path.join(base_path_entry.get(), "ResampledAudio")
    AudioProcessing.Preprocess_audio(input_dir, output_dir, resampled_audio_dir)
    messagebox.showinfo('Success', 'Inference completed successfully.')
  except Exception as e:
    messagebox.showerror('Error', f'An error occurred while running inference: {str(e)}')


root = tk.Tk()
root.title('RibbitRadar')

check_and_install_packages() # remove?

# Add the ffmpeg binary path to the environment
ffmpeg_executable, _ = get_ffmpeg_path()
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_executable)
os.environ["FFMPEG_BINARY"] = ffmpeg_executable

# Model Name
ttk.Label(root, text='Model Name:').grid(column=0, row=0, padx=10, pady=10)
model_name_entry = ttk.Entry(root)
model_name_entry.grid(column=1, row=0, padx=10, pady=10)

# Model Version
ttk.Label(root, text='Model Version:').grid(column=0, row=1, padx=10, pady=10)
model_version_entry = ttk.Entry(root)
model_version_entry.grid(column=1, row=1, padx=10, pady=10)

# Output File Name
ttk.Label(root, text='Output File Name:').grid(column=0, row=2, padx=10, pady=10)
output_file_entry = ttk.Entry(root)
output_file_entry.grid(column=1, row=2, padx=10, pady=10)

# Base Path
ttk.Label(root, text='Base Path:').grid(column=0, row=3, padx=10, pady=10)
base_path_entry = ttk.Entry(root)
base_path_entry.grid(column=1, row=3, padx=10, pady=10)
base_path_button = ttk.Button(root, text="Browse", command=lambda: select_directory(base_path_entry))  
base_path_button.grid(column=2, row=3, padx=10, pady=10)  

# Input Folder
ttk.Label(root, text='Input Folder:').grid(column=0, row=4, padx=10, pady=10)
input_folder_entry = ttk.Entry(root)
input_folder_entry.grid(column=1, row=4, padx=10, pady=10)
input_folder_button = ttk.Button(root, text="Browse", command=lambda: select_directory(input_folder_entry))  
input_folder_button.grid(column=2, row=4, padx=10, pady=10)  

# Run Button
run_button = ttk.Button(root, text='Run Inference', command=run_inference)
run_button.grid(column=0, row=6, columnspan=2, pady=20)

root.mainloop()
