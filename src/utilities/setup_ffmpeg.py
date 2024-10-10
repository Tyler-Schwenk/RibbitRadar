import os
from .get_ffmpeg import get_ffmpeg_path

def setup_ffmpeg():
    """
    Configures the environment with the correct FFmpeg paths.
    """
    ffmpeg_executable, _ = get_ffmpeg_path()
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_executable)
    os.environ["FFMPEG_BINARY"] = ffmpeg_executable
