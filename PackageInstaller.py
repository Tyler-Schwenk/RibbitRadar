import subprocess
import sys
import os

def check_and_install_packages():
    packages_to_install = ['torchaudio', 'timm==0.4.5', 'wget', 'pydub']

    for package in packages_to_install:
        try:
            __import__(package.split('==')[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Check if the GitHub repository is cloned
    if not os.path.exists('ast-Rana-Draytonii'):
        subprocess.check_call(["git", "clone", "https://github.com/Tyler-Schwenk/ast-Rana-Draytonii.git"])

if __name__ == '__main__':
    check_and_install_packages()
