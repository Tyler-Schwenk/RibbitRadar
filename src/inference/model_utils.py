# src/inference/model_utils.py
import torch
from src.models.ast_models import ASTModel
import logging

def initialize_model(checkpoint_path, label_dim):
    """
    Initialize and load the AST model with the given label dimension and checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint (state dict).
        label_dim (int): Number of labels the model will predict.

    Returns:
        torch.nn.Module: The initialized and loaded AST model in evaluation mode.
    """
    model = ASTModel(
        label_dim=label_dim,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1000,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    audio_model = torch.nn.DataParallel(model, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    return audio_model.to(torch.device("cpu")).eval()

def initialize_and_load_model(checkpoint_path, label_choice):
    """
    Initializes the AST model and loads the checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        label_choice (list): The labels to use for predictions.

    Returns:
        torch.nn.Module: The initialized model.
    """
    label_dim = len(label_choice)
    model = initialize_model(checkpoint_path, label_dim)
    logging.info("Model initialized.")
    return model
