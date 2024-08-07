# PackageInstaller.py
import subprocess
import sys
import logging


def check_and_install_packages():
    """Install required Python packages with error handling."""
    packages_to_install = [
        "torchaudio",
        "timm==0.4.5",
        "wget",
        "pydub",
        "requests",
        "soundfile",
        "pandas",
        "openpyxl",
        "gdown",
        "google-auth",
        "google-auth-oauthlib",
        "google-auth-httplib2",
        "google-api-python-client",
        "python-dotenv",
    ]

    for package in packages_to_install:
        try:
            # Check if package is already installed
            __import__(package.split("==")[0])
        except ImportError:
            try:
                # Attempt to install the package
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logging.info("Successfully installed %s", package)
            except subprocess.CalledProcessError as e:
                logging.error("Failed to install %s. Error: %s", package, e)
            except Exception as e:
                logging.error(
                    "An unexpected error occurred while installing %s. Error: %s",
                    package,
                    e,
                )


if __name__ == "__main__":
    check_and_install_packages()
