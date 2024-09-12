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


def make_predictions(
    data_loader,
    audio_model,
    audio_files_dataset,
    progress_callback,
    radr_threshold,
    raca_threshold,
):
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
            prediction = determine_prediction(
                (radr_score, raca_score, negative_score), radr_threshold, raca_threshold
            )
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


def aggregate_results(file_predictions, metadata_dict, progress_callback):
    progress_callback(
        "Inference Step 3/3: aggregating results...",
        95,
        "Inference Step 3/3: aggregating results...",
    )

    results = []  # List to store both the summary and detailed segment information

    for base_file_name, predictions in file_predictions.items():
        # Extract predictions and scores for each segment
        heard_segments_radr = [
            i
            for i, (pred, radr_score, _, _) in enumerate(predictions)
            if "RADR" in pred
        ]
        heard_segments_raca = [
            i
            for i, (pred, _, raca_score, _) in enumerate(predictions)
            if "RACA" in pred
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
        times_heard_raca = "N/A"
        if len(heard_segments_raca) > 0:
            heard_ranges_raca = group_consecutive_elements(heard_segments_raca)
            times_heard_raca = ", ".join(
                f"{s*10}-{(e+1)*10}" for s, e in heard_ranges_raca
            )

        times_heard_radr = "N/A"
        if len(heard_segments_radr) > 0:
            heard_ranges_radr = group_consecutive_elements(heard_segments_radr)
            times_heard_radr = ", ".join(
                f"{s*10}-{(e+1)*10}" for s, e in heard_ranges_radr
            )

        # Try to get metadata for both .WAV and .wav versions of the file
        metadata = metadata_dict.get(
            base_file_name + ".WAV", metadata_dict.get(base_file_name + ".wav", {})
        )
        device_id = metadata.get("device_id")
        recorded_at = metadata.get("recorded_at")
        temperature = metadata.get("temperature")

        # Append the file-level summary first
        results.append(
            {
                "File Name": base_file_name,
                "Prediction": prediction,
                "Times Heard RACA": times_heard_raca,
                "Times Heard RADR": times_heard_radr,
                "Device ID": device_id,
                "Timestamp": recorded_at,
                "Temperature": temperature,
                "Segment": "N/A",  # Summary row
            }
        )

        # Append the detailed information for each segment
        for idx, (
            segment_prediction,
            radr_score,
            raca_score,
            negative_score,
        ) in enumerate(predictions):
            segment_start = idx * 10
            segment_end = (idx + 1) * 10
            segment_range = f"{segment_start}-{segment_end}"

            # Append segment-level information
            results.append(
                {
                    "File Name": f"{base_file_name} (Segment {segment_range})",
                    "Prediction": segment_prediction,
                    "RADR Score": radr_score,
                    "RACA Score": raca_score,
                    "Negative Score": negative_score,
                    "Times Heard RACA": "N/A",
                    "Times Heard RADR": "N/A",
                    "Device ID": "^",
                    "Timestamp": "^",
                    "Temperature": "^",
                    "Segment": segment_range,  # Segment range for detailed info
                }
            )

    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def save_results(
    results_df,
    results_path,
    model_version,
    raca_threshold,
    radr_threshold,
    full_report,
    summary_report,
    custom_report,
):
    if not results_path.endswith(".xlsx"):
        results_path += ".xlsx"

    try:
        with pd.ExcelWriter(results_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            # Define center alignment format
            center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
            # Define thicker border format
            thick_border_format = workbook.add_format({'bottom': 2, 'align': 'center', 'valign': 'vcenter'})

            # Function to adjust column width automatically
            def adjust_column_width(worksheet, df):
                for i, col in enumerate(df.columns):
                    # Find the length of the longest value in the column
                    max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_length)

            # Write global information
            global_info_df = pd.DataFrame(
                {
                    "Model Version": [model_version],
                    "Review Date": [datetime.now().strftime("%Y-%m-%d")],
                    "RACA Threshold": [raca_threshold],
                    "RADR Threshold": [radr_threshold],
                }
            )
            global_info_df.to_excel(
                writer, sheet_name="Results", index=False, startrow=0
            )

            worksheet = writer.sheets["Results"]
            worksheet.set_column(0, len(global_info_df.columns) - 1, None, center_format)
            worksheet.freeze_panes(1, 2)  # Freezes the first row
            adjust_column_width(worksheet, global_info_df)

            # Full report: Include all details (file summaries and segment details)
            if full_report:
                full_report_df = results_df.copy()
                full_report_df.to_excel(
                    writer, sheet_name="Full Report", index=False
                )
                full_report_worksheet = writer.sheets["Full Report"]
                full_report_worksheet.set_column(0, len(full_report_df.columns) - 1, None, center_format)
                full_report_worksheet.freeze_panes(1, 2)  # Freezes the first row
                adjust_column_width(full_report_worksheet, full_report_df)

                # Apply center alignment and thicker borders before each file-level summary row
                for idx, row in full_report_df.iterrows():
                    full_report_worksheet.set_row(idx + 1, None, center_format)
                    if row["Segment"] == "N/A":  # File-level summary
                        full_report_worksheet.set_row(idx, None, thick_border_format)  # Apply thick border to previous row

            # Summary report: Only include file summaries, skipping segment details
            if summary_report:
                summary_report_df = results_df[results_df["Segment"] == "N/A"]
                summary_report_df.to_excel(
                    writer, sheet_name="Summary Report", index=False
                )
                summary_worksheet = writer.sheets["Summary Report"]
                summary_worksheet.set_column(0, len(summary_report_df.columns) - 1, None, center_format)
                summary_worksheet.freeze_panes(1, 2)  # Freezes the first row
                adjust_column_width(summary_worksheet, summary_report_df)

                # Apply center alignment and thicker borders before each file-level summary row
                for idx, row in summary_report_df.iterrows():
                    summary_worksheet.set_row(idx + 1, None, center_format)
                    summary_worksheet.set_row(idx, None, thick_border_format)

            # Custom report: Adjust based on user selection
            if custom_report:
                custom_df = results_df.copy()

                # Remove metadata columns based on user preferences
                if not custom_report["metadata"]:
                    custom_df.drop(
                        ["Device ID", "Timestamp", "Temperature"], axis=1, inplace=True
                    )

                # Remove segment scores and behave like summary if segment_scores is deselected
                if not custom_report["segment_scores"]:
                    custom_df = custom_df[custom_df["Segment"] == "N/A"]
                    custom_df.drop(
                        ["RADR Score", "RACA Score", "Negative Score"],
                        axis=1,
                        inplace=True,
                    )

                if not custom_report["times_heard_radr"]:
                    custom_df.drop(["Times Heard RADR"], axis=1, inplace=True)
                if not custom_report["times_heard_raca"]:
                    custom_df.drop(["Times Heard RACA"], axis=1, inplace=True)

                custom_df.to_excel(
                    writer, sheet_name="Custom Report", index=False
                )
                custom_worksheet = writer.sheets["Custom Report"]
                custom_worksheet.set_column(0, len(custom_df.columns) - 1, None, center_format)
                custom_worksheet.freeze_panes(1, 2)
                adjust_column_width(custom_worksheet, custom_df)

                # Apply center alignment and thicker borders before each file-level summary row
                for idx, row in custom_df.iterrows():
                    custom_worksheet.set_row(idx + 1, None, center_format)
                    if row["Segment"] == "N/A":
                        custom_worksheet.set_row(idx, None, thick_border_format)

            # Apply conditional formatting to highlight specific keywords for all reports
            for sheet_name in ["Full Report", "Summary Report", "Custom Report"]:
                if sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]

                    # Highlight "RACA" cells in yellow
                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'RACA',
                        'format': workbook.add_format({'bg_color': '#FFFF00'})
                    })

                    # Highlight "RADR" cells in light green
                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'RADR',
                        'format': workbook.add_format({'bg_color': '#C6EFCE'})
                    })

                    # Highlight "Negative", "negative", "no", "No", or "NO" cells in rose
                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'Negative',
                        'format': workbook.add_format({'bg_color': '#FFC7CE'})
                    })

                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'negative',
                        'format': workbook.add_format({'bg_color': '#FFC7CE'})
                    })

                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'no',
                        'format': workbook.add_format({'bg_color': '#FFC7CE'})
                    })

                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'No',
                        'format': workbook.add_format({'bg_color': '#FFC7CE'})
                    })

                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'NO',
                        'format': workbook.add_format({'bg_color': '#FFC7CE'})
                    })

                    # Highlight "RADR, RACA" cells in light blue
                    worksheet.conditional_format('A1:Z1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'RADR, RACA',
                        'format': workbook.add_format({'bg_color': '#ADD8E6'})
                    })

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
    raca_threshold,
    full_report,
    summary_report,
    custom_report,
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
        data_loader,
        audio_model,
        audio_files_dataset,
        progress_callback,
        radr_threshold,
        raca_threshold,
    )
    metadata_dict = {md["filename"]: md for md in metadata_dict.values()}
    logging.info(f"Aggregating Results...")
    results_df = aggregate_results(file_predictions, metadata_dict, progress_callback)

    results_path = os.path.join(output_dir, output_file)
    logging.debug(f"Results path: {results_path}")
    save_results(
        results_df,
        results_path,
        model_version,
        raca_threshold,
        radr_threshold,
        full_report,
        summary_report,
        custom_report,
    )
