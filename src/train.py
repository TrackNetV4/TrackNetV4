#!/usr/bin/env python
"""
Training Script
-------------------------

This script trains a TrackNet model on a specified dataset using configurable parameters.
It supports loading a pretrained model, saving checkpoints, and evaluating performance during training.

Usage:
    python src/train.py --model_name <MODEL> --dataset <DATASET> [options]

Example:
    python src/train.py --model_name Baseline_TrackNetV2 --dataset tennis_game_level_split \
        --batch_size 3 --learning_rate 1.0 --height 288 --width 512 --epochs 30 --tol 4 \
        --work_dir ./models --save_freq 1

Arguments:
    --model_name   : Name of the model to use.
                     Allowed values: Baseline_TrackNetV2, TrackNetV4_TypeA, TrackNetV4_TypeB.
    --dataset      : Name of the dataset to use.
                     Allowed values: tennis_game_level_split, tennis_clip_level_split, badminton, new_tennis.
    --batch_size   : Batch size for training (default: 3).
    --learning_rate: Learning rate for the optimizer (default: 1.0).
    --height       : Target image height (default: 288).
    --width        : Target image width (default: 512).
    --epochs       : Number of epochs for training (default: 30).
    --tol          : Tolerance for the outcome evaluation (default: 4).
    --model_path   : Path to a pretrained model (.keras) to load before training (optional).
    --work_dir     : Directory to save the trained models (default: "./models").
    --save_freq    : Frequency (in epochs) to save model checkpoints (default: 1).

Note:
    If the default work directory is used, a timestamp will be appended to create a unique directory.
"""

import argparse
import datetime
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adadelta

from util import custom_loss, get_dataset, get_model, outcome
from models.TrackNetV4 import (
    MotionPromptLayer,
    FusionLayerTypeA,
    FusionLayerTypeB
)


def main(args):
    """
    Train the TrackNet model using specified configurations.
    """
    # Unpack arguments
    model_name = args.model_name
    dataset_name = args.dataset
    batch_size = args.batch_size
    height = args.height
    width = args.width
    learning_rate = args.learning_rate
    epochs = args.epochs
    tol = args.tol
    save_freq = args.save_freq
    model_path = args.model_path
    work_dir = args.work_dir

    # If using default work directory, append a timestamp for uniqueness
    if work_dir == "./models":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(work_dir, timestamp)

    # Print all experiment configurations before starting the training
    experiment_config = {
        "model_name": model_name,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "height": height,
        "width": width,
        "epochs": epochs,
        "tol": tol,
        "model_path": model_path,
        "work_dir": work_dir,
        "save_freq": save_freq,
    }
    print("Training Configurations:")
    for key, value in experiment_config.items():
        print(f"  {key}: {value}")

    # Create the work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)

    # Load model
    model = get_model(model_name, height, width)

    # Load training dataset
    dataset_train = get_dataset(dataset_name, "train")

    # Compile the model with Adadelta optimizer and custom loss
    model.compile(
        loss=custom_loss,
        optimizer=Adadelta(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    # Main training loop
    for epoch in range(epochs):
        print(f"======== Epoch {epoch + 1} ========")

        # Train on each batch in the training dataset
        for x_train, y_train in dataset_train:
            model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
            del x_train, y_train

        # Evaluate model performance on the training set (if enabled)
        if not args.disable_eval_on_train:
            TP = TN = FP1 = FP2 = FN = 0
            for x_train, y_train in dataset_train:
                y_pred = model.predict(x_train, batch_size=batch_size)
                y_pred = (y_pred > 0.5).astype('float32')

                tp, tn, fp1, fp2, fn = outcome(y_pred, y_train, tol)
                TP += tp
                TN += tn
                FP1 += fp1
                FP2 += fp2
                FN += fn

                del x_train, y_train, y_pred

            print(f"Epoch {epoch + 1} results: TP={TP}, TN={TN}, FP1={FP1}, FP2={FP2}, FN={FN}")

        # Save model checkpoint based on frequency
        if (epoch + 1) % save_freq == 0:
            model_path = os.path.join(work_dir, f"model_{epoch + 1}.keras")
            model.save(model_path)
            print(f"Saved model to {model_path}")

    # Save the final model after training
    final_model_path = os.path.join(work_dir, "model_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TrackNet model with configurable parameters."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB'],
        help="Name of the model to use."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis'],
        help="Name of the dataset to use."
    )
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate for the optimizer")
    parser.add_argument("--height", type=int, default=288, help="Target height of the images")
    parser.add_argument("--width", type=int, default=512, help="Target width of the images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--tol", type=int, default=4, help="Tolerance for the outcome evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model (.keras) to load before training"
    )
    parser.add_argument("--work_dir", type=str, default="./models", help="Directory to save the trained models")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency (in epochs) to save model checkpoints")
    parser.add_argument(
        "--disable_eval_on_train",
        action="store_true",
        help="If set, disables evaluations on the model on the training set after each epoch"
    )
    
    args = parser.parse_args()
    main(args)
