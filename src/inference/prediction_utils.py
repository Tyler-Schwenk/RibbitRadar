# src/inference/prediction_utils.py
import torch
import logging
from collections import defaultdict
import os
from src.preprocessing.dataset import RibbitDataset
from src.preprocessing.dataset import get_data_loader
import itertools
import torch, torchaudio
from config.parameters import BATCH_SIZE

def perform_inference(
    audio_model, resampled_audio_dir, label_choice, radr_threshold,
    raca_threshold, prediction_mode, progress_callback
):
    """
    Performs the inference on the dataset.

    Args:
        audio_model (torch.nn.Module): The initialized model.
        resampled_audio_dir (str): Directory of resampled audio files.
        label_choice (list): The labels to use for predictions.
        radr_threshold (float): RADR threshold.
        raca_threshold (float): RACA threshold.
        prediction_mode (str): Prediction mode ('Highest Score' or 'Threshold').
        progress_callback (function): Function to update the GUI progress.

    Returns:
        dict: Predictions for each file.
    """
    progress_callback("Loading dataset...", 20, "Dataset loading.")
    
    audio_files_dataset = RibbitDataset(resampled_audio_dir, transform=make_features_fixed)
    data_loader = get_data_loader(resampled_audio_dir, batch_size=BATCH_SIZE, shuffle=False)

    return make_predictions(
        data_loader, audio_model, audio_files_dataset, progress_callback,
        radr_threshold, raca_threshold, label_choice, prediction_mode
    )

def make_predictions(
    data_loader, audio_model, audio_files_dataset, progress_callback, radr_threshold,
    raca_threshold, label_choice, prediction_mode
):
    """
    Generate predictions for audio files using the AST model.
    """
    file_predictions = defaultdict(list)
    total_processed_files = 0

    progress_callback(
        "Running Inference: Creating predictions...", 15,
        "Running Inference: Creating predictions..."
    )
    for i, batch in enumerate(data_loader):
        logging.debug(f"Batch {i+1}: Type - {type(batch)}, Length - {len(batch)}")

        # Check if batch is a list and convert it to tensor if needed
        if isinstance(batch, list):
            logging.warning("Batch is a list, converting to tensor")
            batch = torch.stack([make_features_fixed(file) for file in batch])

        # Unsqueeze batch if necessary (this was part of the original code)
        if len(batch.shape) == 2:
            logging.debug("Unsqueezing the batch")
            batch = batch.unsqueeze(1)

        feats_data = batch.to(torch.device("cpu"))
        with torch.no_grad():
            output = audio_model.forward(feats_data)
            output = torch.sigmoid(output)
        result_output = output.data.cpu().numpy()

        for j, file_result in enumerate(result_output):
            file_index = total_processed_files + j
            audio_file = audio_files_dataset.files[file_index]
            file_name = os.path.basename(audio_file)

            # Extract base file name without segment info
            base_file_name, _ = os.path.splitext(file_name.split("_segment")[0])

            prediction = determine_prediction(
                file_result, radr_threshold, raca_threshold,
                prediction_mode, label_choice
            )
            file_predictions[base_file_name].append((prediction, *file_result))

        total_processed_files += len(batch)
        percent_processed = total_processed_files / len(audio_files_dataset.files)
        progress_callback(
            f"Running Inference: Creating predictions... {total_processed_files}/{len(audio_files_dataset.files)}",
            int(percent_processed * 100)
        )
    return file_predictions


def determine_prediction(scores, radr_threshold, raca_threshold, prediction_mode, label_choice):
    """
    Determines the prediction based on the scores for RADR, RACA, and Negative.
    """
    radr_score, raca_score, negative_score = None, None, None

    if len(label_choice) == 2:
        if "RADR" in label_choice:
            radr_score, negative_score = scores[0], scores[1]
        elif "RACA" in label_choice:
            raca_score, negative_score = scores[0], scores[1]
    else:
        radr_score, raca_score, negative_score = scores

    if prediction_mode == "Highest Score":
        max_score = max(filter(None, [radr_score, raca_score, negative_score]))
        if max_score == radr_score:
            return "RADR"
        elif max_score == raca_score:
            return "RACA"
        else:
            return "Negative"
    elif prediction_mode == "Threshold":
        predictions = []
        if radr_score and radr_score >= radr_threshold:
            predictions.append("RADR")
        if raca_score and raca_score >= raca_threshold:
            predictions.append("RACA")
        if not predictions:
            predictions.append("Negative")
        return ", ".join(predictions)

# Group the segments into consecutive ranges
def group_consecutive_elements(data):
    """
    Group consecutive numbers in the input list.

    Args:
        data (list): A list of integers representing indices.

    Returns:
        list: A list of tuples, where each tuple represents a start and end of consecutive numbers.
    """
    ranges = []
    for k, g in itertools.groupby(enumerate(data), lambda ix: ix[0] - ix[1]):
        consecutive_elements = list(map(lambda x: x[1], g))
        ranges.append((consecutive_elements[0], consecutive_elements[-1]))
    return ranges

def make_features_fixed(wav_name):
    """
    Extract Mel-frequency features from a waveform file.

    Args:
        wav_name (str): The path to the audio file (wav format).

    Returns:
        torch.Tensor: The extracted Mel-frequency features.
    """
    waveform, sr = torchaudio.load(wav_name)
    return make_features(waveform, mel_bins=128)

def make_features(waveform, mel_bins, target_length=1000):
    """
    Create Mel-frequency filterbank (fbank) features from the input waveform.

    Args:
        waveform (torch.Tensor): The audio waveform tensor.
        mel_bins (int): The number of Mel bins to use in the filterbank.
        target_length (int, optional): The target number of frames in the output. Defaults to 1000.

    Returns:
        torch.Tensor: The computed Mel-frequency filterbank features, padded or truncated to the target length.
    """
    sr = 16000  # the sample rate is always 16kHz

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank