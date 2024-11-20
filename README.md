<p align="center">
    <a href="https://app.gitbook.com/o/aNWUVTNAlFvz3xjN3vJ9/s/AyJT2U3IR01QuVCUVjpK/"><img alt="Docs" src="https://img.shields.io/badge/docs-GitBook-blue"></a>
    <a href="https://opensource.org/license/bsd-3-clause"><img alt="License: BSD 3-Clause" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://github.com/Tyler-Schwenk/RibbitRadar/issues"><img alt="Issue Badge" src="https://img.shields.io/github/issues/Tyler-Schwenk/RibbitRadar"></a>
    <a href="https://github.com/Tyler-Schwenk/RibbitRadar/pulls"><img alt="Pull requests Badge" src="https://img.shields.io/github/issues-pr/Tyler-Schwenk/RibbitRadar"></a>
</p>

# RibbitRadar

RibbitRadar is a python-based application designed to accurately identify the presence of specific frog species within audio recordings. Leveraging my fine-tuned version of the Audio Spectrogram Transformer ([AST](https://github.com/YuanGongND/ast)), RibbitRadar processes audio data, preforms inference, and generates reports with detailed information on detection.

<!--- More details on the underlying machine-learning model can be found [here](https://github.com/tyler-schwenk/ast-training) --->




## Table of Contents
- [Overview](#overview)
- [Functionality](#functionality)
- [Getting Started](#getting-started)
- [Running RibbitRadar](#running-ribbitradar)
- [Citing](#citing)
- [Contact](#contact)


## Overview

Ribbit Radar is part of a broader project focused on automated frog call recognition. The application performs the following key tasks: 

- Preprocessing: Converts audio files into a format suitable for model inference.
- Inference: Uses pre-trained models to identify frog species in the recordings.
- Reporting: Generates results in various report formats, providing both detailed and summary-level information.
- Features: Adjustable prediction mode, thresholds, and report formatting.

A more detailed flowchart of the application logic is below

![flowchart](Ribbit_Radar.png?raw=true "Title")

## Functionality

### Performance 
 - Rana draytonii: Accuracy: 96.52% - Precision: 96.09% - Recall: 91.87%
 - Rana catesbeiana: Accuracy: 94.60% - Precision: 95.61% - Recall: 82.43%

Based on a test set of 10-second audio files with a split of 455 rana draytonii, 370 Rana catesbeiana, and 1111 Negative.

## Getting Started

To use RibbitRadar, download the latest release from the [Releases](https://github.com/Tyler-Schwenk/ribbitradar/releases) page. The release includes a packaged application for macOS and Windows.

### Prerequisites

- macOS or Windows operating system.
- Audio recordings in WAV format to analyze.


## Running RibbitRadar

1. Extract the RibbitRadar.zip file.
2. Navigate to the RibbitRadar directory.
3. Double-click on `main.exe` to run the application.


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
