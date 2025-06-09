#!/usr/bin/env python
"""
Evaluation Script
--------------------------------------

This script evaluates a TrackNet model using a provided test dataset and pre-trained model.
It computes various metrics (accuracy, precision, recall, F1 score, and inference speed) and saves
the results to a JSON file.

Usage:
    python src/eval.py --model_path <path_to_model> --dataset <dataset_name> \
        [--batch_size <batch_size>] [--tol <tolerance>]

Example:
    python src/eval.py --model_path ./models/model_final.keras \
        --dataset tennis_game_level_split --batch_size 1 --tol 4

Arguments:
    --model_name   : Name of the model to use.
                     Allowed values: Baseline_TrackNetV2, TrackNetV4_TypeA, TrackNetV4_TypeB.
    --model_path   : Path to the model file to load before evaluation.
    --dataset      : Name of the dataset to use.
                     Allowed values: tennis_game_level_split, tennis_clip_level_split, badminton, new_tennis.
    --batch_size   : Batch size for evaluation (default: 1).
    --tol          : Tolerance value for evaluation metric calculation (default: 4).
    --result_dir   : Directory to save the evaluation results JSON file (default: "./").

Note:
    The script will output evaluation metrics to the console and save them in a JSON file
    named after the provided weights file.
"""

import os
import json
import time
import argparse
from tensorflow.keras.models import load_model

from util import custom_loss, outcome, get_dataset, get_model
from constants import HEIGHT, WIDTH
from models.TrackNetV4 import (
    MotionPromptLayer,
    FusionLayerTypeA,
    FusionLayerTypeB
)


def evaluate_model(model_name, model_path, dataset, batch_size, tol, result_dir):
    """
    Evaluates the specified TrackNet model on a test dataset.

    Parameters:
        model_name (str): Name of the model to use.
        model_path (str): Path to a pretrained model to load before training.
        dataset (str): Name of the dataset to evaluate.
        batch_size (int): Batch size used during evaluation.
        tol (int): Tolerance value for outcome calculation.
        result_dir (str): Directory to save the evaluation results JSON file.

    Returns:
        None. Prints the evaluation metrics and saves them to a JSON file.
    """
    # Print experiment configurations
    evaluation_config = {
        "model_path": model_path,
        "dataset": dataset,
        "batch_size": batch_size,
        "tol": tol
    }
    print("Evaluation Configurations:")
    for key, value in evaluation_config.items():
        print(f"  {key}: {value}")

    # Ensure the result directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Load the test dataset
    test_dataset = get_dataset(dataset, "test")

    # Load model
    model = get_model(model_name, HEIGHT, WIDTH, model_path)

    # Initialize counters for timing and evaluation metrics
    total_time_taken = 0
    total_frames = 0
    TP = TN = FP1 = FP2 = FN = 0

    # Evaluate the model over each batch in the dataset
    for i, (x_data, y_data) in enumerate(test_dataset):
        start_time = time.time()
        # Run inference on the current batch
        y_pred = model.predict(x_data, batch_size=batch_size)
        elapsed_time = time.time() - start_time
        total_time_taken += elapsed_time

        # Assuming each sample contains 3 frames; update total frames count
        total_frames += len(x_data) * 3

        # Convert predictions to binary values using a threshold of 0.5
        y_pred = (y_pred > 0.5).astype('float32')

        # Compute outcomes: true positives, true negatives, and various false positives/negatives
        tp, tn, fp1, fp2, fn = outcome(y_pred, y_data, tol)
        print(f"Finished evaluating batch {i}: (TP, TN, FP1, FP2, FN) = {(tp, tn, fp1, fp2, fn)}")

        # Aggregate the results
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn

    # Calculate evaluation metrics while safely handling division by zero
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except ZeroDivisionError:
        accuracy = 0

    try:
        precision = TP / (TP + FP1 + FP2)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0

    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    # Calculate inference speed (frames processed per second)
    inference_speed = total_frames / total_time_taken if total_time_taken > 0 else 0

    # Output evaluation metrics
    print("Number of True Positives:", TP)
    print("Number of True Negatives:", TN)
    print("Number of False Positives FP1:", FP1)
    print("Number of False Positives FP2:", FP2)
    print("Number of False Negatives:", FN)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("Inference Speed (frames per second):", inference_speed)

    # Prepare results dictionary for JSON output
    results = {
        "True Positives": TP,
        "True Negatives": TN,
        "False Positives FP1": FP1,
        "False Positives FP2": FP2,
        "False Negatives": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Inference Speed (frames per second)": inference_speed,
        "Total Frames Processed": total_frames,
        "Total Time Taken for Inference (seconds)": total_time_taken
    }

    # Create a JSON file name based on the model weights file name
    model_file_name = os.path.basename(model_path).replace('.keras', '')
    json_file_path = os.path.join(result_dir, f"{model_file_name}.json")

    # Save the results to a JSON file with proper indentation
    with open(json_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {json_file_path}")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Evaluate a TrackNet model with provided weights and dataset."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB'],
        help="Name of the model to use."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the model weights file (.keras) to load before evaluation."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis'],
        help="Name of the dataset to use."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Batch size for evaluation. Default is 1."
    )
    parser.add_argument(
        '--tol',
        type=int,
        default=4,
        help="Tolerance value for evaluation metric calculation. Default is 4."
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default="./",
        help="Directory to save the evaluation results JSON file (default: './')."
    )

    args = parser.parse_args()

    # Call the evaluation function with the provided arguments
    evaluate_model(
        args.model_path,
        args.dataset,
        args.batch_size,
        args.tol,
        args.result_dir
    )
