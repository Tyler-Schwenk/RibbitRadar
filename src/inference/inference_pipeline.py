from src.preprocessing.preprocessing_manager import preprocess_audio_pipeline
from src.metadata_extraction.audio_file_metadata_extractor import extract_metadata_from_audiomoth_files
from src.inference.inference_logic import run_inference_logic
from config import paths

def start_inference_pipeline(
    model_path, model_version, output_dir, output_file, input_dir,
    update_progress, radr_threshold, raca_threshold, full_report,
    summary_report, custom_report, label_choice, prediction_mode
):
    """
    Manages the complete inference pipeline, including preprocessing, metadata extraction, and inference.
    """
    try:
        # Step 1: Preprocessing
        update_progress("Starting preprocessing...")
        preprocess_audio_pipeline(input_dir, update_progress)

        # Step 2: Metadata Extraction
        update_progress("Extracting metadata...")
        metadata_dict = extract_metadata_from_audiomoth_files(input_dir, update_progress)

        # Step 3: Inference
        run_inference_logic(
            checkpoint_path=model_path,
            model_version=model_version,
            resampled_audio_dir=paths.RESAMPLED_AUDIO_PATH,
            temp_file_storage=paths.TEMP_FILE_STORAGE,
            output_dir=output_dir,
            output_file=output_file,
            metadata_dict=metadata_dict,
            progress_callback=update_progress,
            radr_threshold=radr_threshold,
            raca_threshold=raca_threshold,
            full_report=full_report,
            summary_report=summary_report,
            custom_report=custom_report,
            label_choice=label_choice,
            prediction_mode=prediction_mode
        )

    except Exception as e:
        update_progress(f"Error during inference: {str(e)}", 100, "Error")
