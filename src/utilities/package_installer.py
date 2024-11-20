import subprocess
import sys
import logging
import os

def check_and_install_packages(requirements_file="config/requirements.txt"):
    """Install required Python packages from requirements.txt with error handling."""
    try:
        # Check if the requirements file exists
        if not os.path.exists(requirements_file):
            logging.error("Requirements file not found at %s", requirements_file)
            return False

        logging.info("Checking if required packages are installed from %s", requirements_file)
        # Install packages from requirements file
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.error("Package installation failed: %s", result.stderr)
            return False
        logging.info(result.stdout)
        return True
    except Exception as e:
        logging.error("Error during package installation: %s", e)
        return False


if __name__ == "__main__":
    check_and_install_packages()
