import subprocess
import sys
import logging
import os

def check_and_install_packages(requirements_file="config/requirements.txt"):
    """Install required Python packages from requirements.txt with error handling."""
    try:
        # Check if the requirements file exists
        if os.path.exists(requirements_file):
            logging.info("Checking if required packages are installed from %s", requirements_file)
            # Check if any packages need installation or updates
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--disable-pip-version-check", "-r", requirements_file],
                capture_output=True,
                text=True
            )

            # If no package was installed or updated, continue execution
            if "Requirement already satisfied" in result.stdout and "upgraded" not in result.stdout:
                logging.info("All packages are already installed and up to date, skipping restart.")
                return

            # Log package installation or upgrade details
            logging.info(result.stdout)

            logging.info("Some packages were installed or updated, restarting application to apply changes...")

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
