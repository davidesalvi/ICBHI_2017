# Amplifier Health - AI Engineer/Researcher Take-Home Project

## Overview
This repository contains the implementation of the Take-Home project for the AI Engineer/Researcher role at Amplifier Health.

To tackle this task, I classified audio data using both binary and multi-class classification approaches.
This was done by implementing two different models: ResNet-18 and LCNN.
These models process various audio features (MelSpectrogram, LogSpectrogram, MFCC, and LFCC), with an analysis conducted to identify the feature set that yields the best classification performance for the task at hand.

Here are the main components of this repository:

- `main.py`: The core script that handles model setup, dataset loading, training, and evaluation.
- `visualize_results.py`: A script to visualize the evaluation results of the trained models.
- `src/`: A folder containing the source code for the models, training, and feature extraction.
- `config/`: A folder containing configuration files for the models and training.
- `requirements.txt`: A txt file listing all the dependencies needed to run the project.
- `run_experiments.sh`: A shell script to run the experiments across different configurations.
- `notebooks/analyze_data.ipynb`: A Jupyter notebook for dataset analysis and visualization.
- `report.pdf`: The final report of the assignment.

## Running the Code

After creating the environment, run the `main.py` script, choosing the experiment configuration using the following command-line arguments:

- `--feature_set`: Choose from ['MelSpec', 'LogSpec', 'MFCC', 'LFCC'] for the feature set.
- `--model_arch`: Choose from ['ResNet', 'LCNN'] for the model architecture.
- `--train_model`: Set to `True` to train the model, or `False` to skip training.
- `--eval_model`: Set to `True` to evaluate the model after training, or `False` to skip evaluation.
- `--classification_type`: Set the classification type to either 'binary' or 'multi'.
- `--win_len`: The length (in seconds) of the audio window for analysis.

Example:

```bash
python main.py --feature_set MelSpec --model_arch ResNet --train_model True --eval_model True --classification_type binary --win_len 5.0
```
