import sys
import os
import warnings
import logging

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pydub import AudioSegment


def get_ffmpeg_path():
    logging.info("Checking FFmpeg path...")
    logging.debug(
        f"Operating System: {os.name}, Python Version: {sys.version}, _MEIPASS: {getattr(sys, '_MEIPASS', 'Not set')}"
    )

    # Get the directory of the current script
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    # Define the path to the ffmpeg executable inside the bin folder
    ffmpeg_bin_dir = os.path.join(base_path, "ffmpeg", "bin")

    # Choose the correct executable based on the operating system
    if os.name == "nt":  # Windows
        ffmpeg_executable = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
        ffprobe_executable = os.path.join(ffmpeg_bin_dir, "ffprobe.exe")
    else:  # macOS, Linux, or other *nix systems
        ffmpeg_executable = os.path.join(ffmpeg_bin_dir, "ffmpeg")
        ffprobe_executable = os.path.join(ffmpeg_bin_dir, "ffprobe")

    # Verify that the FFmpeg binaries exist
    if not os.path.isfile(ffmpeg_executable) or not os.path.isfile(ffprobe_executable):
        logging.error("FFmpeg binaries not found at expected locations:")
        logging.error("ffmpeg path: ", ffmpeg_executable)
        logging.error("ffprobe path: ", ffprobe_executable)
    else:
        logging.info("FFmpeg binaries found successfully.")
        logging.info("ffmpeg path: ", ffmpeg_executable)
        logging.info("ffprobe path: ", ffprobe_executable)

    # Set the converter path for pydub
    AudioSegment.converter = ffmpeg_executable
    AudioSegment.ffprobe = ffprobe_executable

    return ffmpeg_executable, ffprobe_executable
