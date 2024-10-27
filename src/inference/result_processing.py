# src/inference/result_processing.py
import os
import pandas as pd
from datetime import datetime
from src.inference.prediction_utils import group_consecutive_elements
import logging

def aggregate_results(file_predictions, metadata_dict, progress_callback, label_choice):
    """
    Aggregates the segment-level predictions into a file-level summary.

    Args:
        file_predictions (dict): The predictions for each segment of the files.
        metadata_dict (dict): Metadata information for each file (e.g., device ID, timestamp, etc.).
        progress_callback (function): Callback function to update progress.
        label_choice (list): List of labels used in prediction (e.g., ['RADR', 'Negative']).

    Returns:
        pd.DataFrame: A DataFrame containing both file-level and segment-level details of the predictions.
    """
    progress_callback(
        "Finishing Inference: aggregating results...",
        95,
        "Finishing Inference: aggregating results...",
    )

    results = []  # List to store both the summary and detailed segment information

    for base_file_name, predictions in file_predictions.items():
        # Handle 2-label or 3-label cases dynamically
        num_labels = len(predictions[0]) - 1  # Exclude the prediction text itself
        heard_segments_radr = []
        heard_segments_raca = []

        # Case using binary search for one frog or negative
        if num_labels == 2:
            # Case searching for RADR/Negative
            if "RADR" in predictions[0][0]:
                heard_segments_radr = [
                    i
                    for i, (pred, radr_score, negative_score) in enumerate(predictions)
                    if "RADR" in pred
                ]
            # Case searching for RACA/Negative
            elif "RACA" in predictions[0][0]:
                heard_segments_raca = [
                    i
                    for i, (pred, raca_score, negative_score) in enumerate(predictions)
                    if "RACA" in pred
                ]
            radr_scores = [score for _, score, _ in predictions]
            raca_scores = []  # No RACA scores in this case
            negative_scores = [score for _, _, score in predictions]

        elif num_labels == 3:  # Handle the three-label system (RADR, RACA, Negative)
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
            radr_scores = [score for _, score, _, _ in predictions]
            raca_scores = [score for _, _, score, _ in predictions]
            negative_scores = [score for _, _, _, score in predictions]

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
            *scores,  # Variable length to handle both 2-label and 3-label cases
        ) in enumerate(predictions):
            segment_start = idx * 10
            segment_end = (idx + 1) * 10
            segment_range = f"{segment_start}-{segment_end}"

            # Handle segment-level information based on the number of labels
            segment_info = {
                "File Name": f"{base_file_name} (Segment {segment_range})",
                "Prediction": segment_prediction,
                "Times Heard RACA": "N/A",
                "Times Heard RADR": "N/A",
                "Device ID": "^",
                "Timestamp": "^",
                "Temperature": "^",
                "Segment": segment_range,  # Segment range for detailed info
            }

            if num_labels == 2:
                # For 2-label, add RADR or RACA with Negative
                if "RADR" in label_choice:
                    segment_info["RADR Score"] = scores[0]
                elif "RACA" in label_choice:
                    segment_info["RACA Score"] = scores[0]
                segment_info["Negative Score"] = scores[1]
            elif num_labels == 3:
                # For 3-label, add RADR, RACA, and Negative
                segment_info["RADR Score"] = scores[0]
                segment_info["RACA Score"] = scores[1]
                segment_info["Negative Score"] = scores[2]

            results.append(segment_info)

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
    label_choice,
    prediction_mode,
):
    """
    Save the prediction results to an Excel file with optional formatting and conditional formatting.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results.
        results_path (str): Path to save the Excel file.
        model_version (str): The version of the model used.
        raca_threshold (float): The threshold for RACA classification.
        radr_threshold (float): The threshold for RADR classification.
        full_report (bool): Whether to include the full report.
        summary_report (bool): Whether to include the summary report.
        custom_report (dict): Custom report settings (include metadata, segment scores, etc.).

    Returns:
        None
    """
    if not results_path.endswith(".xlsx"):
        results_path += ".xlsx"

    try:
        with pd.ExcelWriter(results_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            # Define center alignment format
            center_format = workbook.add_format(
                {"align": "center", "valign": "vcenter"}
            )
            # Define thicker border format
            thick_border_format = workbook.add_format(
                {"bottom": 2, "align": "center", "valign": "vcenter"}
            )

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
                    "Label Choice": [label_choice],
                    "Prediction Mode": [prediction_mode],
                }
            )
            global_info_df.to_excel(
                writer, sheet_name="Results", index=False, startrow=0
            )

            worksheet = writer.sheets["Results"]
            worksheet.set_column(
                0, len(global_info_df.columns) - 1, None, center_format
            )
            worksheet.freeze_panes(1, 2)  # Freezes the first row
            adjust_column_width(worksheet, global_info_df)

            # Full report: Include all details (file summaries and segment details)
            if full_report:
                full_report_df = results_df.copy()
                full_report_df.to_excel(writer, sheet_name="Full Report", index=False)
                full_report_worksheet = writer.sheets["Full Report"]
                full_report_worksheet.set_column(
                    0, len(full_report_df.columns) - 1, None, center_format
                )
                full_report_worksheet.freeze_panes(1, 2)  # Freezes the first row
                adjust_column_width(full_report_worksheet, full_report_df)

                # Apply center alignment and thicker borders before each file-level summary row
                for idx, row in full_report_df.iterrows():
                    full_report_worksheet.set_row(idx + 1, None, center_format)
                    if row["Segment"] == "N/A":  # File-level summary
                        full_report_worksheet.set_row(
                            idx, None, thick_border_format
                        )  # Apply thick border to previous row

            # Summary report: Only include file summaries, skipping segment details
            if summary_report:
                summary_report_df = results_df[results_df["Segment"] == "N/A"]
                summary_report_df.to_excel(
                    writer, sheet_name="Summary Report", index=False
                )
                summary_worksheet = writer.sheets["Summary Report"]
                summary_worksheet.set_column(
                    0, len(summary_report_df.columns) - 1, None, center_format
                )
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

                custom_df.to_excel(writer, sheet_name="Custom Report", index=False)
                custom_worksheet = writer.sheets["Custom Report"]
                custom_worksheet.set_column(
                    0, len(custom_df.columns) - 1, None, center_format
                )
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
                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "RACA",
                            "format": workbook.add_format({"bg_color": "#FFFF00"}),
                        },
                    )

                    # Highlight "RADR" cells in light green
                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "RADR",
                            "format": workbook.add_format({"bg_color": "#C6EFCE"}),
                        },
                    )

                    # Highlight "Negative", "negative", "no", "No", or "NO" cells in rose
                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "Negative",
                            "format": workbook.add_format({"bg_color": "#FFC7CE"}),
                        },
                    )

                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "negative",
                            "format": workbook.add_format({"bg_color": "#FFC7CE"}),
                        },
                    )

                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "no",
                            "format": workbook.add_format({"bg_color": "#FFC7CE"}),
                        },
                    )

                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "No",
                            "format": workbook.add_format({"bg_color": "#FFC7CE"}),
                        },
                    )

                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "NO",
                            "format": workbook.add_format({"bg_color": "#FFC7CE"}),
                        },
                    )

                    # Highlight "RADR, RACA" cells in light blue
                    worksheet.conditional_format(
                        "A1:Z1000",
                        {
                            "type": "text",
                            "criteria": "containing",
                            "value": "RADR, RACA",
                            "format": workbook.add_format({"bg_color": "#ADD8E6"}),
                        },
                    )

        logging.info(f"Results successfully saved to {results_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")

def save_inference_results(
    predictions, metadata_dict, output_dir, output_file, model_version, radr_threshold,
    raca_threshold, full_report, summary_report, custom_report, label_choice, progress_callback, prediction_mode
):
    logging.info("Aggregating results...")
    results_df = aggregate_results(predictions, metadata_dict, progress_callback, label_choice)
    results_path = os.path.join(output_dir, output_file)
    save_results(results_df, results_path, model_version, raca_threshold, radr_threshold, full_report, summary_report, custom_report, label_choice, prediction_mode)
    progress_callback("Results saved.", 90, "Results saved.")
