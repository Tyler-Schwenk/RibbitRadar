# RibbitRadar: Rana Draytonii Detection

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Running RibbitRadar](#running-ribbitradar)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Ideas for Improvement](#ideas-for-improvement)
- [Citing](#citing)
- [Contact](#contact)

## Introduction

RibbitRadar is a software tool designed to accurately identify the presence of the endangered frog species, *Rana Draytonii*, within audio recordings. Leveraging my fine-tuned version of the Audio Spectrogram Transformer (AST), RibbitRadar processes raw audio data to recognize the unique calls of *Rana Draytonii* amidst various background noises. More details on the underlying machine-learning model can be found [here](https://github.com/tyler-schwenk/ast-rana-draytonii)

## Getting Started

To use RibbitRadar, download the latest release from the [Releases](https://github.com/Tyler-Schwenk/ribbitradar/releases) page. The release includes a packaged application for macOS and Windows, making it straightforward to run without needing to install Python or any dependencies. If time is a constraint, my [Google Colab](https://github.com/tyler-schwenk/ast-rana-draytonii) version can provide increased speed if you pay for the use of Google's powerful GPUs or TPUs.

### Prerequisites

- macOS or Windows operating system.
- Audio recordings in WAV format to analyze.

## Running RibbitRadar

### macOS

1. Extract the RibbitRadar.zip file.
2. Open the Terminal and navigate to the RibbitRadar directory.
3. Run the application using the command:

    ```bash
    ./main
    ```

### Windows

1. Extract the RibbitRadar.zip file.
2. Navigate to the RibbitRadar directory.
3. Double-click on `main.exe` to run the application.

## Project Structure

The core of RibbitRadar lies within its ability to preprocess audio files, run them through the AST model, and output detailed analysis including detection times and environmental metadata in an Excel file.

## Examples

- [Solo Rana Draytonii Call](https://example.com)
- [Rana Draytonii Amongst Noise](https://example.com)

## Ideas for Improvement

- Enhance frequency range filtering to focus on Rana Draytonii's vocalization frequencies.
- Expand the model to detect additional species relevant to Rana Draytonii's ecosystem.
- Integrate geographical data to visualize call distributions over time or space.

## Citing

If you utilize RibbitRadar in your research, please consider citing the original AST paper and any subsequent works that this project builds upon.

```bibtex
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
