import torchaudio
import math
import os
import re
import shutil
from src.utilities import get_ffmpeg
from tkinter import messagebox
from pydub import AudioSegment
from pydub.utils import mediainfo


def split_all_audio_files(
    input_dir,
    output_dir,
    progress_callback,
    segment_length_sec=10,
    file_types=(".wav", ".WAV"),
):
    """
    Splits each audio file in the input directory into segments of specified length and saves them in the output directory.

    Parameters:
    input_dir (str): The directory containing the original audio files.
    output_dir (str): The directory where the segmented audio files will be saved.
    segment_length_sec (int, optional): Length of each audio segment in seconds. Defaults to 10 seconds.
    file_types (tuple, optional): Tuple of file extensions to consider for processing. Defaults to ('.wav', '.WAV').

    This function processes each audio file in the input directory that matches the specified file types. It splits the audio
    files into segments of a specified length and saves each segment as a new audio file in the output directory.
    Errors during processing are caught and displayed in a messagebox.

    Returns:
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(file_types):
            audio_path = os.path.join(input_dir, filename)
            audio_path = os.path.normpath(audio_path)

            try:
                waveform, sample_rate = torchaudio.load(audio_path, format="wav")
                num_samples_segment = segment_length_sec * sample_rate
                total_segments = math.ceil(waveform.shape[1] / num_samples_segment)

                for i in range(total_segments):
                    start = i * num_samples_segment
                    end = start + num_samples_segment
                    segment = waveform[:, start:end]
                    segment_filename = (
                        f"{filename.rstrip('.wav').rstrip('.WAV')}_segment{i}.wav"
                    )
                    segment_path = os.path.join(output_dir, segment_filename)
                    segment = (segment * 32767).short()  # Convert to 16-bit PCM format
                    torchaudio.save(segment_path, segment, sample_rate)

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error processing file {filename}: {str(e)}"
                )


def extract_metadata_from_files_in_directory(
    directory, progress_callback, file_types=(".wav", ".WAV")
):
    """
    Extracts metadata from audio files in the specified directory.

    Parameters:
    directory (str): The directory containing audio files from which metadata is to be extracted.
    file_types (tuple, optional): Tuple of file extensions to consider for metadata extraction. Defaults to ('.wav', '.WAV').

    Returns:
    dict: A dictionary where each key is a filename, and the value is another dictionary of extracted metadata.
    """
    metadata_dict = {}
    ffmpeg_executable, _ = (
        get_ffmpeg.get_ffmpeg_path()
    )  # Retrieve the path to the local ffmpeg binary

    for file in os.listdir(directory):
        if file.endswith(file_types):
            filepath = os.path.join(directory, file)
            filepath = os.path.normpath(filepath)
            metadata = mediainfo(filepath)

            # Extract desired information from the comment
            comment = metadata.get("TAG", {}).get("comment", "")
            match = re.search(
                r"Recorded at (.*) by (AudioMoth .*) at .* while .* and temperature was (.*C)\.",
                comment,
            )

            # Additional check for no match found
            if match:
                recorded_at, device_id, temperature = match.groups()

                # Create a new dictionary with only the desired information
                simplified_metadata = {
                    "filename": file,
                    "recorded_at": recorded_at,
                    "device_id": device_id,
                    "temperature": temperature,
                }
                metadata_dict[file] = (
                    simplified_metadata  # Store metadata using filename as key
                )
            else:
                progress_callback(log_message=f"No metadata match for file {file}")

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
        file_path = os.path.normpath(file_path)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


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
    new_path = os.path.normpath(new_path)

    # save the resampled audio
    torchaudio.save(new_path, waveform_resampled, sample_rate=16000)

    return new_path

def resample_audio_files(input_dir, output_dir, progress_callback):
    """
    Resamples all audio files in the input directory and saves them to the output directory.

    Parameters:
    input_dir (str): The directory containing the original audio files.
    output_dir (str): The directory where resampled audio files will be saved.
    progress_callback (function): A callback function for updating the progress of the resampling.

    Returns:
    None
    """
    try:
        audio_files = [
            f for f in os.listdir(input_dir) if f.endswith((".wav", ".WAV"))
        ]
        total_files = len(audio_files)

        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for i, filename in enumerate(audio_files, start=1):
            filepath = os.path.join(input_dir, filename)
            filepath = os.path.normpath(filepath)
            resampler(filepath, output_dir)

            # Update progress based on number of files processed
            progress_percentage = (i / total_files) * 100
            progress_callback(
                f"Resampling audio files... ({i}/{total_files})", progress_percentage
            )

    except Exception as e:
        messagebox.showerror(
            "Error", f"An error occurred while resampling audio files: {str(e)}"
        )
        progress_callback(
            log_message=f"An error occurred while resampling audio files: {str(e)}"
        )

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
        if filename.endswith(".wav"):
            # Get the full path of the file
            file_path = os.path.join(directory_path, filename)
            file_path = os.path.normpath(file_path)

            # Load audio file
            audio = AudioSegment.from_wav(file_path)

            # If the audio file is stereo
            if audio.channels == 2:
                print(f"Converting stereo file: {filename}")

                # Convert to mono
                mono_audio = audio.set_channels(1)

                # Replace the original file with the mono version
                mono_audio.export(file_path, format="wav")


