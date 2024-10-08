from .inference import run_inference as inference_logic
import os

def run_inference(
    temp_file_storage, resampled_audio_path, model_path, model_version,
    metadata_dict, update_progress, radr_threshold, raca_threshold,
    full_report, summary_report, custom_report, label_choice, prediction_mode,
    output_file
):
    """
    Delegates the inference logic to the core inference script while handling progress updates for the GUI.
    """
    inference_logic(
        checkpoint_path=model_path,  # Path to model checkpoint
        temp_file_storage=temp_file_storage,  # Temporary storage path
        resampled_audio_dir=resampled_audio_path,  # Directory for resampled audio
        model_version=model_version,  # Model version
        output_dir=os.path.dirname(output_file),  # Directory for output
        output_file=os.path.basename(output_file),  # File name for output
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
