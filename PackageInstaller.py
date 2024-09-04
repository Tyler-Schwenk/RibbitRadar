import subprocess
import sys
import logging
import os


def check_and_install_packages(requirements_file="requirements.txt"):
    """Install required Python packages from requirements.txt with error handling."""
    if os.path.exists(requirements_file):
        try:
            # Attempt to install packages from requirements.txt
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", requirements_file]
            )
            logging.info("Successfully installed packages from %s", requirements_file)
        except subprocess.CalledProcessError as e:
            logging.error(
                "Failed to install packages from %s. Error: %s", requirements_file, e
            )
        except Exception as e:
            logging.error(
                "An unexpected error occurred while installing packages. Error: %s", e
            )
    else:
        logging.error("Requirements file %s not found", requirements_file)


if __name__ == "__main__":
    check_and_install_packages()
