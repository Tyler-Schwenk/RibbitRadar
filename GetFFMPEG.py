def get_ffmpeg_path():
    # Get the directory of the current script
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    # Define the path to the ffmpeg executable
    ffmpeg_dir = os.path.join(base_path, 'ffmpeg')
    # Choose the correct executable based on the operating system
    if os.name == 'nt':  # Windows
        ffmpeg_executable = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
        ffprobe_executable = os.path.join(ffmpeg_dir, 'ffprobe.exe')
    else:  # macOS, Linux, or other *nix systems
        ffmpeg_executable = os.path.join(ffmpeg_dir, 'ffmpeg')
        ffprobe_executable = os.path.join(ffmpeg_dir, 'ffprobe')
    return ffmpeg_executable, ffprobe_executable
