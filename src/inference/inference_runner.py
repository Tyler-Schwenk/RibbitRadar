# src/inference/inference_runner.py
from .inference_logic import run_inference_logic
from config import paths

def run_inference( model_path, model_version, output_dir,
    metadata_dict, update_progress, radr_threshold, raca_threshold,
    full_report, summary_report, custom_report, label_choice, prediction_mode,
    output_file
):
    """
    Delegates the inference logic to the core inference script while handling progress updates for the GUI.
    """
    run_inference_logic(
        checkpoint_path=model_path,  # Path to model checkpoint
        temp_file_storage=paths.TEMP_FILE_STORAGE,  # Temporary storage path
        resampled_audio_dir=paths.RESAMPLED_AUDIO_PATH,  # Directory for resampled audio
        model_version=model_version,  # Model version
        output_dir=output_dir,  # Directory for output, passed directly
        output_file=output_file,  # Already a unique filename, passed directly
        metadata_dict=metadata_dict,  # Metadata generated during preprocessing
        progress_callback=update_progress,  # Progress callback from the GUI
        radr_threshold=radr_threshold,  # RADR threshold as float
        raca_threshold=raca_threshold,  # RACA threshold as float
        full_report=full_report,  # Full report option
        summary_report=summary_report,  # Summary report option
        custom_report=custom_report,  # Custom report options
        label_choice=label_choice,  # Label choices
        prediction_mode=prediction_mode  # Prediction mode
    )
