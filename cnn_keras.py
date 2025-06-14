#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional Neural Networks with Keras - Extended Version
==========================================================

This script demonstrates how to build and train CNNs using Keras with the MNIST dataset.
It includes models with 1, 2, and 3 sets of convolutional and pooling layers (with both
decreasing and increasing filter patterns), tested with different batch sizes,
and includes comprehensive plotting and comparison.

Uses proper Train/Validation/Test methodology:
- Train: 48,000 samples (80% of original training data)
- Validation: 12,000 samples (20% of original training data)
- Test: 10,000 samples (separate test set, used only for final evaluation)

Total Models: 12 (4 architectures × 3 batch sizes)
- CNN 1 Layer
- CNN 2 Layers
- CNN 3 Layers (Decreasing filters: 32→16→8)
- CNN 3 Layers (Increasing filters: 8→16→32)

Batch Sizes: 50, 150, 250

Installation Requirements:
--------------------------
Run these commands to install required libraries:

pip install numpy==2.0.2
pip install pandas==2.2.2
pip install tensorflow-cpu==2.18.0
pip install matplotlib==3.9.2
pip install scikit-learn
"""

import os
import warnings

# Suppress TensorFlow warning messages (for CPU architecture)
# Comment out these lines if using GPU architecture
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Import required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def create_results_directory():
    """
    Create a directory to store all result plots.

    Returns:
        str: Path to the created directory
    """
    dir_name = "CNN_Results"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f"Results will be saved to: {dir_name}")
    return dir_name


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset with proper Train/Validation/Test split.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, num_classes)
    """
    print("Loading MNIST dataset...")

    # Load data
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # Split training data into train and validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"Original training data: {X_train_full.shape[0]} samples")
    print(f"After split - Training: {X_train.shape[0]} samples")
    print(f"After split - Validation: {X_val.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")

    # Reshape to be [samples][height][width][channels]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    # Normalize pixel values to be between 0 and 1
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # Convert target variable into binary categories (one-hot encoding)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    num_classes = y_test.shape[1]  # number of categories

    print(f"Final data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Number of classes: {num_classes}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


def create_cnn_model_1_layer(num_classes):
    """
    Create a CNN model with one set of convolutional and pooling layers.

    Args:
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (5, 5), strides=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_cnn_model_2_layers(num_classes):
    """
    Create a CNN model with two sets of convolutional and pooling layers.

    Args:
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(8, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_cnn_model_3_layers_decreasing(num_classes):
    """
    Create a CNN model with three sets of convolutional and pooling layers (Decreasing filters).

    Args:
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(8, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_cnn_model_3_layers_increasing(num_classes):
    """
    Create a CNN model with three sets of convolutional and pooling layers (Increasing filters).

    Args:
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(8, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(32, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_training_history(history, model_name, batch_size, save_dir):
    """
    Create and save a plot showing training history with both accuracy and loss.
    Uses dynamic scaling for better visualization.

    Args:
        history: Keras training history object
        model_name (str): Name of the model
        batch_size (int): Batch size used for training
        save_dir (str): Directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy\n(Batch Size: {batch_size})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Dynamic Y-axis scaling for accuracy
    all_acc = history.history['accuracy'] + history.history['val_accuracy']
    min_acc = min(all_acc)
    max_acc = max(all_acc)
    margin = (max_acc - min_acc) * 0.1  # 10% margin
    ax1.set_ylim([max(0, min_acc - margin), min(1, max_acc + margin)])

    # Plot training & validation loss
    ax2.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title(f'{model_name} - Loss\n(Batch Size: {batch_size})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Dynamic Y-axis scaling for loss
    all_loss = history.history['loss'] + history.history['val_loss']
    min_loss = min(all_loss)
    max_loss = max(all_loss)
    margin = (max_loss - min_loss) * 0.1  # 10% margin
    ax2.set_ylim([max(0, min_loss - margin), max_loss + margin])

    plt.tight_layout()

    # Save the plot
    filename = f"{model_name.replace(' ', '_')}_batch_{batch_size}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {filename}")


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                             epochs=10, batch_size=200, model_name="CNN", save_dir=None):
    """
    Train and evaluate a CNN model with proper Train/Validation/Test methodology.

    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data (used only for final evaluation)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        model_name (str): Name for logging purposes
        save_dir (str): Directory to save plots

    Returns:
        tuple: (trained_model, val_accuracy, test_accuracy, history)
    """
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    print(f"{'=' * 60}")

    # Train the model using validation data for monitoring
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

    # Plot and save training history
    if save_dir:
        plot_training_history(history, model_name, batch_size, save_dir)

    # Evaluate on validation set
    val_scores = model.evaluate(X_val, y_val, verbose=0)
    val_accuracy = val_scores[1]

    # Final evaluation on test set (only once!)
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    test_accuracy = test_scores[1]

    print(f"\nFinal Results for {model_name} (Batch Size: {batch_size}):")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Validation Error: {100 - val_accuracy * 100:.2f}%")
    print(f"Test Error: {100 - test_accuracy * 100:.2f}%")

    return model, val_accuracy, test_accuracy, history


def plot_all_test_accuracies(results, save_dir):
    """
    Create a summary plot showing all test accuracies.

    Args:
        results (list): List of tuples containing (model_name, batch_size, val_accuracy, test_accuracy, history)
        save_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(16, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

    # Group results by model architecture
    model_types = {}
    for model_name, batch_size, val_accuracy, test_accuracy, history in results:
        arch = model_name.split('(')[0].strip()  # Extract architecture name
        if arch not in model_types:
            model_types[arch] = []
        model_types[arch].append((batch_size, test_accuracy, history))

    # Plot each model type
    for i, (arch_name, arch_results) in enumerate(model_types.items()):
        batch_sizes = []
        final_test_accuracies = []

        for batch_size, test_accuracy, history in arch_results:
            batch_sizes.append(batch_size)
            final_test_accuracies.append(test_accuracy * 100)  # Convert to percentage

        plt.plot(batch_sizes, final_test_accuracies,
                 color=colors[i], marker=markers[i], markersize=10, linewidth=3,
                 label=f'{arch_name}', alpha=0.8)

        # Add value labels on points
        for j, (bs, acc) in enumerate(zip(batch_sizes, final_test_accuracies)):
            plt.annotate(f'{acc:.1f}%',
                         (bs, acc),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=10,
                         fontweight='bold')

    plt.title('Test Accuracy Comparison\nAcross All Models and Batch Sizes',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Batch Size', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xticks([50, 150, 250], fontsize=12)
    plt.yticks(fontsize=12)

    # Set y-axis to show a good range
    all_test_accuracies = [test_acc for _, _, _, test_acc, _ in results]
    min_acc = min(all_test_accuracies) * 100
    max_acc = max(all_test_accuracies) * 100
    plt.ylim([min_acc - 1, max_acc + 1])

    plt.tight_layout()

    # Save the plot
    filename = "All_Models_Test_Accuracy_Comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved summary plot: {filename}")


def main():
    """
    Main function to run the complete CNN training and evaluation process.
    """
    print("Convolutional Neural Networks with Keras - Extended Version")
    print("=" * 70)
    print("Training 4 CNN architectures with 3 different batch sizes each")
    print("Total models to train: 12")
    print("=" * 70)

    # Create results directory
    save_dir = create_results_directory()

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_and_preprocess_data()

    # Define model architectures and batch sizes
    model_configs = [
        ("CNN 1 Layer", create_cnn_model_1_layer),
        ("CNN 2 Layers", create_cnn_model_2_layers),
        ("CNN 3 Layers (Decreasing)", create_cnn_model_3_layers_decreasing),
        ("CNN 3 Layers (Increasing)", create_cnn_model_3_layers_increasing)
    ]

    batch_sizes = [50, 150, 250]
    epochs = 10

    # Store all results for final comparison
    all_results = []

    # Train all model combinations
    total_models = len(model_configs) * len(batch_sizes)
    current_model = 0

    for model_name, model_creator in model_configs:
        for batch_size in batch_sizes:
            current_model += 1

            print(f"\n{'=' * 70}")
            print(f"MODEL {current_model}/{total_models}: {model_name} with Batch Size {batch_size}")
            print(f"{'=' * 70}")

            # Create and train model
            model = model_creator(num_classes)

            # Show model architecture for first batch size only
            if batch_size == batch_sizes[0]:
                print(f"\nModel Architecture ({model_name}):")
                model.summary()

            # Train and evaluate
            full_model_name = f"{model_name} (Batch Size: {batch_size})"
            trained_model, val_accuracy, test_accuracy, history = train_and_evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test,
                epochs=epochs, batch_size=batch_size,
                model_name=model_name, save_dir=save_dir
            )

            # Store results
            all_results.append((full_model_name, batch_size, val_accuracy, test_accuracy, history))

    # Create summary plot
    print(f"\n{'=' * 70}")
    print("CREATING SUMMARY PLOT")
    print(f"{'=' * 70}")

    plot_all_test_accuracies(all_results, save_dir)

    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY OF ALL RESULTS")
    print(f"{'=' * 80}")

    print(f"{'Model Configuration':<45} {'Batch Size':<12} {'Val Accuracy':<12} {'Test Accuracy':<12} {'Test Error'}")
    print("-" * 95)

    for model_name, batch_size, val_accuracy, test_accuracy, history in all_results:
        arch_name = model_name.split('(')[0].strip()
        print(
            f"{arch_name:<45} {batch_size:<12} {val_accuracy * 100:>8.2f}%  {test_accuracy * 100:>9.2f}%  {100 - test_accuracy * 100:>8.2f}%")

    # Find best model based on test accuracy
    best_model = max(all_results, key=lambda x: x[3])  # x[3] is test_accuracy
    print(f"\nBest performing model (Test Accuracy): {best_model[0]}")
    print(f"Best test accuracy: {best_model[3] * 100:.2f}%")

    print(f"\nAll results saved to: {save_dir}")
    print(f"Total files created: {len(all_results) + 1} PNG files")
    print(f"- {len(all_results)} individual model plots (Train/Validation Accuracy + Loss)")
    print(f"- 1 summary comparison plot (Test Accuracy)")

    print(f"\nMethodology used:")
    print(f"- Training: {X_train.shape[0]} samples")
    print(f"- Validation: {X_val.shape[0]} samples (for monitoring during training)")
    print(f"- Test: {X_test.shape[0]} samples (for final evaluation only)")


if __name__ == "__main__":
    main()