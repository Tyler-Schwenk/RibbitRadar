import torchaudio
import math
import os
import re
import shutil
import json
import csv
import GetFFMPEG
from pydub import AudioSegment
from pydub.utils import mediainfo # for retrieving metadata

def split_all_audio_files(input_dir, output_dir, segment_length_sec=10, file_types=('.wav', '.WAV')):
  """
    Splits each audio file in the input directory into segments of specified length and saves them in the output directory.

    Parameters:
    input_dir (str): The directory containing the original audio files.
    output_dir (str): The directory where the segmented audio files will be saved.
    segment_length_sec (int, optional): Length of each audio segment in seconds. Defaults to 10 seconds.
    file_types (tuple, optional): Tuple of file extensions to consider for processing. Defaults to ('.wav', '.WAV').

    Returns:
    None
    """
  os.makedirs(output_dir, exist_ok=True)

  for filename in os.listdir(input_dir):
    if filename.endswith(file_types):
      try:
        # Full path to the original audio file
        audio_path = os.path.join(input_dir, filename)
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Calculate number of samples in segment_length_sec
        num_samples_segment = segment_length_sec * sample_rate
        total_segments = math.ceil(waveform.shape[1] / num_samples_segment)

        # Split waveform into segments and save each segment to a new .wav file
        for i in range(total_segments):
          start = i * num_samples_segment
          end = start + num_samples_segment
          segment = waveform[:, start:end]

          # Prepare filename for the segment
          segment_filename = f"{filename.rstrip('.wav')}_segment{i}.wav"
          segment_path = os.path.join(output_dir, segment_filename)

          # Save segment as a .wav file
          segment = (segment * 32767).short()  # Convert to 16-bit PCM format
          torchaudio.save(segment_path, segment, sample_rate)
      except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
  print("Finished Splitting files into segments.")


def extract_metadata_from_files_in_directory(directory, file_types=('.wav', '.WAV')):
  """
    Extracts metadata from audio files in the specified directory.

    Parameters:
    directory (str): The directory containing audio files from which metadata is to be extracted.
    file_types (tuple, optional): Tuple of file extensions to consider for metadata extraction. Defaults to ('.wav', '.WAV').

    Returns:
    dict: A dictionary where each key is a filename, and the value is another dictionary of extracted metadata.
    """
  metadata_dict = {}
  ffmpeg_executable, _ = GetFFMPEG.get_ffmpeg_path()  # Retrieve the path to the local ffmpeg binary


  for file in os.listdir(directory):
    if file.endswith(file_types):
      filepath = os.path.join(directory, file)
      metadata = mediainfo(filepath, ffmpeg=ffmpeg_executable)

      # Extract desired information from the comment
      comment = metadata.get('TAG', {}).get('comment', '')
      match = re.search(r"Recorded at (.*) by (AudioMoth .*) at .* while .* and temperature was (.*C)\.", comment)

      # Additional check for no match found
      if match:
        recorded_at, device_id, temperature = match.groups()

        # Create a new dictionary with only the desired information
        simplified_metadata = {
          'filename': file,
          'recorded_at': recorded_at,
          'device_id': device_id,
          'temperature': temperature,
        }
        metadata_dict[file] = simplified_metadata  # Store metadata using filename as key
      else:
        print(f"No metadata match for file {file}")

  return metadata_dict


def clear_directory(directory):
    """
    Clears all files and subdirectories in the given directory.

    Parameters:
    directory (str): The directory to be cleared.

    Returns:
    None
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def define_paths():
    #THIS IS BAD NEEDS CHANGING
    model_name = "AST_Rana_Draytonii"
    model_version = "2.0"

    # output will be stored in this excel file. Definitly change this one
    output_file_name = "results.xlsx"
    # !! If there is already a file with this name at this location it will be overwritten with the new file !!

    # Define the base path to where your Rana7 folder is stored in google drive
    # If youre not sure you can find it in the file explorer to the left. /content will always be the base
    base_path = '/content/drive/MyDrive/Rana_Draytonii/Rana_Draytonii_ML_Model'

    # This is the name of the folder within Rana_Draytionii_ML_Model where you will place the files you want reviewed
    input_folder = 'Wav_Files_Input'

    """**That was all you need to do for inputting data**, now just run the rest of the cells top to bottom"""

    # Defines paths for other files/directories by concatenating with the base path
    labels_path = os.path.join(base_path, 'labels.csv')
    checkpoint_path = os.path.join(base_path, 'best_audio_model_V2.pth')
    input_dir = os.path.join(base_path, input_folder)
    output_dir = os.path.join(base_path, 'Temp_File_Storage')
    results_path = os.path.join(base_path, 'results.xlsx')
    resampled_audio_dir = os.path.join(base_path, 'ResampledAudio')
    metadata_dict = extract_metadata_from_files_in_directory(input_dir)

def resampler(audio_path, save_dir):
    """
    Resamples the given audio file to 16kHz and saves the output to the specified directory.

    Parameters:
    audio_path (str): The path to the original audio file.
    save_dir (str): The directory where the resampled audio file will be saved.

    Returns:
    str: The path to the resampled audio file.
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)

    waveform_resampled = resampler(waveform)

    # create new path with original filename
    base_filename = os.path.basename(audio_path)
    new_path = os.path.join(save_dir, base_filename)

    # save the resampled audio
    torchaudio.save(new_path, waveform_resampled, sample_rate=16000)

    return new_path

# Function to convert strero files to mono
def stereo_to_mono(directory_path):
  """
    Converts all stereo audio files in the given directory to mono.

    Parameters:
    directory_path (str): The directory containing audio files to be converted.

    Returns:
    None
    """
  # Loop through all files in the directory
  for filename in os.listdir(directory_path):
    # Check if the file is a .wav file
    if filename.endswith('.wav'):
      # Get the full path of the file
      file_path = os.path.join(directory_path, filename)

      # Load audio file
      audio = AudioSegment.from_wav(file_path)

      # If the audio file is stereo
      if audio.channels == 2:
        print(f"Converting stereo file: {filename}")

        # Convert to mono
        mono_audio = audio.set_channels(1)

        # Replace the original file with the mono version
        mono_audio.export(file_path, format='wav')

def Preprocess_audio(input_dir, output_dir, resampled_audio_dir):
    """
    Preprocesses audio files by splitting, clearing directories, resampling, and converting to mono.

    This function performs several preprocessing steps on audio files:
    - Splits audio files into smaller segments.
    - Clears the output directory.
    - Resamples audio files to a specified sample rate.
    - Converts stereo audio files to mono.

    Parameters:
    input_dir (str): The directory containing the original audio files to be processed.
    output_dir (str): The directory where the processed audio segments will be saved.
    resampled_audio_dir (str): The directory where resampled audio files will be saved.

    Returns:
    None
    """
    try:
        # Clears the output directory
        clear_directory(output_dir)
        print("Output directory cleared...")
        
        # Splits and processes the audio files
        split_all_audio_files(input_dir, output_dir)
        print("Audio files split into segments...")
        
        # Clear the ResampledAudio directory and resample the split audio files
        if not os.path.exists(resampled_audio_dir):
            os.makedirs(resampled_audio_dir)
            print(f"Directory {resampled_audio_dir} created...")
        
        clear_directory(resampled_audio_dir)
        print("ResampledAudio directory cleared...")
        
        # Convert to mono and resample the audio files
        stereo_to_mono(output_dir)
        print("Files converted to Mono...")
        
        for filename in os.listdir(output_dir):
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filepath = os.path.join(output_dir, filename)
                resampler(filepath, resampled_audio_dir)  # Resample and get new file path
        
        print("Finished Preprocessing...")
    except Exception as e:
        messagebox.showerror('Error', f'An error occurred while preprocessing audio: {str(e)}')
