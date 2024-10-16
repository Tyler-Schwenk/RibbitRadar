# src/inference/inference_logic.py
import logging
from src.inference.model_utils import initialize_and_load_model
from src.inference.prediction_utils import perform_inference
from src.inference.result_processing import save_inference_results
from src.inference.cleanup import clean_up

def run_inference_logic(
    checkpoint_path, temp_file_storage, resampled_audio_dir, model_version, output_dir,
    output_file, metadata_dict, progress_callback, radr_threshold, raca_threshold,
    full_report, summary_report, custom_report, label_choice, prediction_mode
):
    """
    Runs the full inference process to predict frog calls in audio files and generates a report.
    """
    try:
        logging.info("Initializing the inference process...")
        # Initialize and load the model
        audio_model = initialize_and_load_model(checkpoint_path, label_choice)
        progress_callback("Model initialized successfully.", 10, "Model initialized.")

        # Load the dataset and run predictions
        predictions = perform_inference(
            audio_model, resampled_audio_dir, label_choice, radr_threshold,
            raca_threshold, prediction_mode, progress_callback
        )
        if predictions is None:
            raise Exception("Prediction process failed.")

        # Aggregate results and save them
        save_inference_results(
            predictions, metadata_dict, output_dir, output_file, model_version,
            radr_threshold, raca_threshold, full_report, summary_report,
            custom_report, label_choice, progress_callback, prediction_mode
        )

        # Clean up temporary directories
        clean_up(temp_file_storage, resampled_audio_dir)

        logging.info("Inference process completed successfully.")
        progress_callback("Inference process completed.", 100, "Completed.")

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        progress_callback(f"Error: {str(e)}", 100, "Error")
