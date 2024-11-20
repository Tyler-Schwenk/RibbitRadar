import os
import logging
import sys

def setup_logging():
    """
    Sets up logging for the application, creating a log file dynamically.
    Returns the path to the log file.
    """
    # Determine the log directory
    if getattr(sys, 'frozen', False):  # Check if running as a packaged executable
        # Use user-specific directory for packaged version
        log_dir = os.path.join(os.path.expanduser("~"), "RibbitRadarLogs")
    else:
        # Use project root during development
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')

    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

    # Log file name
    log_file = os.path.join(log_dir, "ribbitradar.log")

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),  # Append to log file
            logging.StreamHandler()  # Output to console
        ],
        force=True,
    )
    logging.info(f"Logging initialized. Log file located at: {log_file}")
    return log_file
