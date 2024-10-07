# RibbitRadar: Rana Draytonii Detection

<p align="center">
    <a href="https://opensource.org/license/bsd-3-clause"><img alt="License: BSD 3-Clause" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://github.com/Tyler-Schwenk/RibbitRadar/issues"><img alt="Issue Badge" src="https://img.shields.io/github/issues/Tyler-Schwenk/RibbitRadar"></a>
    <a href="https://github.com/Tyler-Schwenk/RibbitRadar/pulls"><img alt="Pull requests Badge" src="https://img.shields.io/github/issues-pr/Tyler-Schwenk/RibbitRadar"></a>
</p>


# This README is out of date. Will be updated by 10/17


## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running RibbitRadar](#running-ribbitradar)
- [Examples](#examples)
- [Ideas for Improvement](#ideas-for-improvement)
- [Citing](#citing)
- [Contact](#contact)

## Introduction

RibbitRadar is a hands-off software tool designed to accurately identify the presence of the endangered frog species, *Rana Draytonii*, within audio recordings. Leveraging my fine-tuned version of the Audio Spectrogram Transformer (AST), RibbitRadar processes raw audio data to recognize the unique calls of *Rana Draytonii* amidst various background noises. More details on the underlying machine-learning model can be found [here](https://github.com/tyler-schwenk/ast-rana-draytonii)

## Project Structure

RibbitRadar will take the directory on your machine containing audio recordings from the field, as well as the name and location for your output file to be saved. After inputting this, the user only needs to click "Run Inference" and watch the application update them on the model locating all instances of *Rana Draytonii* vocalizations in their files. RibbitRadar will preprocess audio files, run them through the AST model, and output detailed analysis including detection times and environmental metadata in an Excel file at the location specified. The output file will contain information as below:

| Model Name : Version | File Name     | Prediction | Times Heard (sec) | Device ID               | Timestamp                  | Temperature | Review Date |
|----------------------|---------------|------------|-------------|-------------------------|----------------------------|-------------|-------------|
| AST_Rana_Draytonii:2.0 | POND_19000 | Positive!   | 0-20, 40-60     | AudioMoth 249BC30461CBB1E6 | 19:00:00 01/12/2022 (UTC-8) | 9.3C        | 2023-07-22  |
| AST_Rana_Draytonii:2.0 | POND_20500 | Negative   | N/A         | AudioMoth 249BC30461CBB1E6 | 20:50:00 01/12/2022 (UTC-8) | 9.1C        | 2023-07-22  |



## Getting Started

To use RibbitRadar, download the latest release from the [Releases](https://github.com/Tyler-Schwenk/ribbitradar/releases) page. The release includes a packaged application for macOS and Windows, making it straightforward to run without needing to install Python or any dependencies. If time is a constraint, my [Google Colab](https://github.com/tyler-schwenk/ast-rana-draytonii) version can provide increased speed if you pay for the use of Google's powerful GPUs or TPUs.

### Prerequisites

- macOS or Windows operating system.
- Audio recordings in WAV format to analyze.

Before setting up the project, ensure that the following dependencies are installed:

1. Python 3.x
2. pip (Python package manager)
3. Microsoft Visual C++ Redistributable (required for PyTorch and other packages)
   - Download and install from [Microsoft's official site](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160)
4. FFmpeg - (add details)


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


## Future Development

- Expand the model to detect additional species relevant to Rana Draytonii's ecosystem.
    - Given quality training data, this can be achieved with the current training pipeline    
- Integrate geographical data to visualize call distributions over time and space.

## Citing  

If you utilize RibbitRadar in your research, please consider citing the original AST paper and any subsequent works that this project builds upon.

The first paper proposes the Audio Spectrogram Transformer while the second paper describes the training pipeline that they applied on AST to achieve the new state-of-the-art on AudioSet.   
```  
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
```  
```  
@ARTICLE{gong_psla, 
    author={Gong, Yuan and Chung, Yu-An and Glass, James},  
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
    title={PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation},   
    year={2021}, 
    doi={10.1109/TASLP.2021.3120633}
}
```  


 ## Contact
If you have a question, would like to develop something similar for another species, or just want to share how you have used this, send me an email at tylerschwenk1@yahoo.com.
