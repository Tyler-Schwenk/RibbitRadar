# src/metadata_extraction/metadata_extractor.py

import os
import re
from pydub.utils import mediainfo

def extract_metadata_from_audiomoth_files(directory, progress_callback, file_types=(".wav", ".WAV")):
    """
    Extracts metadata from audio files frecorded by audiomoth recorders in the specified directory.
    More info: https://www.openacousticdevices.info/support/device-support/how-to-extract-temperature-data-from-any-audiomoth-s-file

    Parameters:
    directory (str): The directory containing audio files from which metadata is to be extracted.
    file_types (tuple, optional): Tuple of file extensions to consider for metadata extraction. Defaults to ('.wav', '.WAV').

    Returns:
    dict: A dictionary where each key is a filename, and the value is another dictionary of extracted metadata.
    """
    metadata_dict = {}

    for file in os.listdir(directory):
        if file.endswith(file_types):
            filepath = os.path.join(directory, file)
            filepath = os.path.normpath(filepath)
            metadata = mediainfo(filepath)

            comment = metadata.get("TAG", {}).get("comment", "")
            match = re.search(
                r"Recorded at (.*) by (AudioMoth .*) at .* while .* and temperature was (.*C)\.",
                comment,
            )

            if match:
                recorded_at, device_id, temperature = match.groups()
                metadata_dict[file] = {
                    "filename": file,
                    "recorded_at": recorded_at,
                    "device_id": device_id,
                    "temperature": temperature,
                }
            else:
                progress_callback(log_message=f"No metadata match for file {file}")

    return metadata_dict
