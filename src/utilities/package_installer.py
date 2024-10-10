import subprocess
import sys
import logging
import os

def check_and_install_packages(requirements_file="config/requirements.txt"):
    """Install required Python packages from requirements.txt with error handling."""
    try:
        # Check if the requirements file exists
        if os.path.exists(requirements_file):
            logging.info("Installing required packages from %s", requirements_file)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            logging.info("Package installation successful.")

            # Relaunch the script with the --restart flag
            logging.info("Restarting application to apply newly installed dependencies...")
            subprocess.run([sys.executable, *sys.argv, "--restart"], check=True)

        else:
            logging.error("Requirements file not found at %s", requirements_file)
            return False
    except subprocess.CalledProcessError as e:
        logging.error("Package installation failed with error: %s", e)
        return False


if __name__ == "__main__":
    check_and_install_packages()
