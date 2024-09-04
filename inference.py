from src.models import ASTModel
from datetime import datetime
from collections import defaultdict
import DataSet
import itertools
import torch, torchaudio, timm
import numpy
import pandas as pd
import os

import logging


def make_features(waveform, mel_bins, target_length=1000):
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


def make_features_fixed(wav_name):
    waveform, sr = torchaudio.load(wav_name)
    return make_features(waveform, mel_bins=128)


# Function to initialize and load the model
def initialize_model(checkpoint_path):
    model = ASTModel(
        label_dim=3,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1000,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    audio_model = torch.nn.DataParallel(model, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    return audio_model.to(torch.device("cpu")).eval()


def make_predictions(data_loader, audio_model, audio_files_dataset, progress_callback, radr_threshold, raca_threshold):
    logging.debug("Starting make_predictions function")
    file_predictions = defaultdict(list)
    total_processed_files = 0

    progress_callback(
        "Inference Step 2/3 (the big one): Creating predictions...",
        15,
        "Inference Step 2/3 (the big one): Creating predictions...",
    )
    for i, batch in enumerate(data_loader):
        logging.debug(f"Batch {i+1}: Type - {type(batch)}, Length - {len(batch)}")

        # Check if batch is a list or a tensor
        if isinstance(batch, list):
            logging.warning("Batch is a list, converting to tensor")
            # Assuming batch is a list of file paths
            batch = torch.stack([make_features_fixed(file) for file in batch])

        if len(batch.shape) == 2:
            logging.debug("Unsqueezing the batch")
            batch = batch.unsqueeze(1)

        feats_data = batch.to(torch.device("cpu"))
        with torch.no_grad():
            logging.debug("Making predictions")
            output = audio_model.forward(feats_data)
            output = torch.sigmoid(output)
        result_output = output.data.cpu().numpy()

        for j, file_result in enumerate(result_output):
            file_index = total_processed_files + j
            if file_index >= len(audio_files_dataset.files):
                logging.error(f"Index out of range: {file_index}")
                continue

            audio_file = audio_files_dataset.files[file_index]
            file_name = os.path.basename(audio_file)
            logging.debug(f"Processing file: {file_name}")

            base_file_name, _ = os.path.splitext(file_name.split("_segment")[0])

            # Extract scores
            radr_score = file_result[0]
            raca_score = file_result[1]
            negative_score = file_result[2]

            # Determine prediction
            prediction = determine_prediction((radr_score, raca_score, negative_score), radr_threshold, raca_threshold)
            file_predictions[base_file_name].append(
                (prediction, radr_score, raca_score, negative_score)
            )

        total_processed_files += len(batch)
        percent_processed = total_processed_files / len(audio_files_dataset.files)
        progress_callback(
            f"Inference Step 2/3 (the big one): Creating predictions...{total_processed_files}/{len(audio_files_dataset.files)}",
            int(percent_processed),
            f"Inference Step 2/3 (the big one): Creating predictions...{total_processed_files}/{len(audio_files_dataset.files)}",
        )
        logging.debug(
            f"Processed {total_processed_files}/{len(audio_files_dataset.files)} files"
        )

    return file_predictions


# Group the segments into consecutive ranges
def group_consecutive_elements(data):
    ranges = []
    for k, g in itertools.groupby(enumerate(data), lambda ix: ix[0] - ix[1]):
        consecutive_elements = list(map(lambda x: x[1], g))
        ranges.append((consecutive_elements[0], consecutive_elements[-1]))
    return ranges


def determine_prediction(scores, radr_threshold, raca_threshold):
    """
    Determines the prediction based on the scores for RADR and RACA.

    Args:
        scores (tuple): A tuple containing the scores for RADR, RACA, and Negative.
        radr_threshold (float): The threshold above which a RADR score is considered positive.
        raca_threshold (float): The threshold above which a RACA score is considered positive.

    Returns:
        str: The prediction based on the scores.
    """
    radr_score, raca_score, negative_score = scores

    predictions = []
    if radr_score >= radr_threshold:
        predictions.append("RADR")
    if raca_score >= raca_threshold:
        predictions.append("RACA")
    if not predictions:
        predictions.append("Negative")

    return ", ".join(predictions)


def aggregate_results(
    file_predictions, metadata_dict, progress_callback
):
    progress_callback(
        "Inference Step 3/3: aggregating results...",
        95,
        "Inference Step 3/3: aggregating results...",
    )
    results_df = pd.DataFrame(
        columns=[
            "File Name",
            "Prediction",
            "Avg RADR Score",
            "Avg RACA Score",
            "Avg Negative Score",
            "Times Heard RACA",
            "Times Heard RADR",
            "Device ID",
            "Timestamp",
            "Temperature",
        ]
    )

    for base_file_name, predictions in file_predictions.items():
        # Extract predictions and scores for each segment
        heard_segments_radr = [
            i for i, (pred, radr_score, _, _) in enumerate(predictions) if "RADR" in pred
        ]
        heard_segments_raca = [
            i for i, (pred, _, raca_score, _) in enumerate(predictions) if "RACA" in pred
        ]
        
        # Scores for aggregating
        radr_scores = [score for _, score, _, _ in predictions]
        raca_scores = [score for _, _, score, _ in predictions]
        negative_scores = [score for _, _, _, score in predictions]

        # Compute average scores
        avg_radr_score = sum(radr_scores) / len(radr_scores)
        avg_raca_score = sum(raca_scores) / len(raca_scores)
        avg_negative_score = sum(negative_scores) / len(negative_scores)

        # Determine the prediction for the whole file based on heard segments
        if heard_segments_radr and heard_segments_raca:
            prediction = "RADR, RACA"
        elif heard_segments_radr:
            prediction = "RADR"
        elif heard_segments_raca:
            prediction = "RACA"
        else:
            prediction = "Negative"

        # Group consecutive segments where RACA and RADR were heard
        if len(heard_segments_raca) == 0:
            times_heard_raca = "N/A"
        else:
            heard_ranges_raca = group_consecutive_elements(heard_segments_raca)
            times_heard_raca = ", ".join(f"{s*10}-{(e+1)*10}" for s, e in heard_ranges_raca)

        if len(heard_segments_radr) == 0:
            times_heard_radr = "N/A"
        else:
            heard_ranges_radr = group_consecutive_elements(heard_segments_radr)
            times_heard_radr = ", ".join(f"{s*10}-{(e+1)*10}" for s, e in heard_ranges_radr)

        # Try to get metadata for both .WAV and .wav versions of the file
        metadata = metadata_dict.get(
            base_file_name + ".WAV", metadata_dict.get(base_file_name + ".wav", {})
        )
        device_id = metadata.get("device_id")
        recorded_at = metadata.get("recorded_at")
        temperature = metadata.get("temperature")

        # Append results to the dataframe
        new_row = pd.DataFrame(
            {
                "File Name": [base_file_name],
                "Prediction": [prediction],
                "Avg RADR Score": [avg_radr_score],
                "Avg RACA Score": [avg_raca_score],
                "Avg Negative Score": [avg_negative_score],
                "Times Heard RACA": [times_heard_raca],
                "Times Heard RADR": [times_heard_radr],
                "Device ID": [device_id],
                "Timestamp": [recorded_at],
                "Temperature": [temperature],
            }
        )
        if not new_row.empty:
            results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df



# Function to save results
def save_results(
    results_df, 
    results_path, 
    model_version, 
    raca_threshold, 
    radr_threshold
):
    if not results_path.endswith(".xlsx"):
        results_path += ".xlsx"

    try:
        with pd.ExcelWriter(results_path, engine='xlsxwriter') as writer:
            # Create a summary DataFrame with global information
            global_info_df = pd.DataFrame(
                {
                    "Model Version": [model_version],
                    "Review Date": [datetime.now().strftime("%Y-%m-%d")],
                    "RACA Threshold": [raca_threshold],
                    "RADR Threshold": [radr_threshold],
                }
            )

            # Write the global information at the top
            global_info_df.to_excel(writer, sheet_name="Results", index=False, startrow=0)
            
            # Write the results starting after the global information (row 3)
            results_df.to_excel(writer, sheet_name="Results", index=False, startrow=4)
            
        logging.info(f"Results successfully saved to {results_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")



def run_inference(
    labels_path,
    checkpoint_path,
    resampled_audio_dir,
    model_version,
    output_dir,
    output_file,
    metadata_dict,
    progress_callback,
    radr_threshold, 
    raca_threshold
):
    """
    Runs the inference process on preprocessed audio files to detect Rana Draytonii calls.

    This function performs the following steps:
    1. Initializes the AST model with the specified checkpoint.
    2. Loads and preprocesses audio files from the resampled audio directory.
    3. Runs predictions on the audio segments using the model.
    4. Aggregates results and metadata.
    5. Generates an Excel report with the inference results.

    Parameters:
    labels_path (str): The path to the labels.csv file.
    checkpoint_path (str): The path to the model weights file (e.g., best_audio_model_V2.pth).
    resampled_audio_dir (str): Path to the directory containing resampled audio files.
    model_version (str): Number indicating the model version.
    output_dir (str): Directory where the results will be saved.
    output_file (str): Name of the output Excel file.
    metadata_dict (dict): Dictionary containing metadata for the audio files.
    progress_callback (function): A callback function for updating the progress of the inference process.
        This function should accept three parameters: a message (str), a progress value (float or int), and a log message (str).

    Returns:
    None
    """
    logging.debug(f"resampled_audio_dir: {resampled_audio_dir}")
    # Check if resampled_audio_dir is a string and points to a directory
    if not isinstance(resampled_audio_dir, str) or not os.path.isdir(
        resampled_audio_dir
    ):
        logging.error("resampled_audio_dir is not a valid directory path")
        return

    progress_callback(
        "Inference Step 1/3: Initializing model...",
        10,
        "Inference Step 1/3: Initializing model...",
    )

    audio_model = initialize_model(checkpoint_path)

    audio_files_dataset = DataSet.RanaDraytoniiDataset(
        resampled_audio_dir, transform=make_features_fixed
    )
    data_loader = DataSet.get_data_loader(
        resampled_audio_dir, batch_size=32, shuffle=False
    )

    file_predictions = make_predictions(
        data_loader, audio_model, audio_files_dataset, progress_callback, radr_threshold, raca_threshold
    )
    metadata_dict = {md["filename"]: md for md in metadata_dict.values()}
    results_df = aggregate_results(
        file_predictions, metadata_dict, progress_callback
    )

    results_path = os.path.join(output_dir, output_file)
    logging.debug(f"Results path: {results_path}")
    save_results(results_df, results_path, model_version, raca_threshold, radr_threshold)
