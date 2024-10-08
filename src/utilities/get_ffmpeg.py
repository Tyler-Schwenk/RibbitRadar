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

    # Adjust the ffmpeg path to point to the main RibbitRadar folder
    ribbitradar_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ffmpeg_bin_dir = os.path.join(ribbitradar_root, 'ffmpeg', 'bin')

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
        logging.error(f"ffmpeg path:  {ffmpeg_executable}")
        logging.error(f"ffprobe path: {ffprobe_executable}")
    else:
        logging.info("FFmpeg binaries found successfully.")
        logging.error(f"ffmpeg path:  {ffmpeg_executable}")
        logging.error(f"ffprobe path: {ffprobe_executable}")

    # Set the converter path for pydub
    AudioSegment.converter = ffmpeg_executable
    AudioSegment.ffprobe = ffprobe_executable

    return ffmpeg_executable, ffprobe_executable
