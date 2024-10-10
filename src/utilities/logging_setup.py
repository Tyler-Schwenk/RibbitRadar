import logging

def initialize_logging(log_file_path):
    """
    Initializes the logging setup for the application.
    
    Args:
        log_file_path (str): Path to the log file.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),  # Log to a file
            logging.StreamHandler(),  # Log to the console
        ],
    )
    logging.info("Logging setup complete.")
