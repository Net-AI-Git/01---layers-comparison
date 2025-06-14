# 01---layers-comparison
# CNN Layers Comparison: Architectural Impact Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## Description

A comprehensive empirical study comparing the performance impact of different CNN architectures on the MNIST handwritten digit classification task. This project systematically evaluates how the number of convolutional layers and their filter configurations affect model accuracy and training dynamics across various batch sizes.

The study tests 4 distinct CNN architectures (1-3 layers with different filter patterns) against 3 batch sizes (50, 150, 250), resulting in 12 total model configurations with detailed performance analysis.

**Key Value**: Provides data-driven insights for CNN architecture selection and hyperparameter tuning decisions in computer vision projects.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Experimental Design](#experimental-design)
- [Results Summary](#results-summary)
- [Visual Results](#visual-results)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## Technologies Used

- **Python**: 3.8+
- **TensorFlow**: 2.18.0 (CPU optimized)
- **Keras**: High-level neural networks API
- **NumPy**: 2.0.2 - Numerical computing
- **Matplotlib**: 3.9.2 - Visualization and plotting
- **Scikit-learn**: Model evaluation utilities
- **Pandas**: 2.2.2 - Data manipulation and analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Net-AI-Git/01---layers-comparison.git
   cd 01---layers-comparison
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python cnn_keras.py --help
   ```

## Usage

### Running the Complete Analysis

Execute the full experiment suite (trains all 12 model configurations):

```bash
python cnn_keras.py
```

**Expected Runtime**: Approximately 15-20 minutes on standard CPU hardware.

### Output

The script will generate:
- **Results Directory**: `CNN_Results/` containing all visualization plots
- **Individual Model Plots**: Training/validation accuracy and loss curves for each configuration
- **Summary Plot**: Comparative test accuracy analysis across all models
- **Console Output**: Detailed performance metrics and final rankings

### Customization

To modify experimental parameters, edit the configuration variables in `main()`:

```python
batch_sizes = [50, 150, 250]  # Adjust batch sizes
epochs = 10                   # Modify training duration
```

## Project Structure

```
01---layers-comparison/
├── cnn_keras.py              # Main experiment script
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── results/                 # 📁 CREATE THIS FOLDER - Add your images here
│   ├── All_Models_Test_Accuracy_Comparison.png
│   ├── CNN_1_Layer_batch_50.png
│   ├── CNN_2_Layers_batch_150.png
│   └── [additional model plots...]
└── CNN_Results/             # Generated results (created after execution)
    ├── CNN_1_Layer_batch_50.png
    ├── CNN_2_Layers_batch_150.png
    ├── [additional model plots...]
    └── All_Models_Test_Accuracy_Comparison.png
```

## Experimental Design

### Model Architectures

1. **CNN 1 Layer**: Single Conv2D(16) + MaxPooling + Dense layers
2. **CNN 2 Layers**: Conv2D(16) → Conv2D(8) with pooling between layers  
3. **CNN 3 Layers (Decreasing)**: 32→16→8 filter progression
4. **CNN 3 Layers (Increasing)**: 8→16→32 filter progression

### Training Methodology

- **Dataset Split**: 80% training (48K), 20% validation (12K), separate test set (10K)
- **Optimizer**: Adam with categorical crossentropy loss
- **Epochs**: 10 per model
- **Evaluation**: Validation monitoring during training, final test evaluation

### Batch Size Analysis

Tests model sensitivity to batch sizes: 50, 150, and 250 samples per batch to understand convergence behavior and final performance impact.

## Results Summary

### Key Findings

- **Best Performing Model**: CNN 2 Layers with batch size 50 achieved **98.88% test accuracy**
- **Architecture Impact**: 2-layer CNN consistently outperformed both simpler and more complex architectures
- **Batch Size Trends**: Smaller batch sizes (50) generally yielded better final accuracy across all architectures
- **Performance Range**: All models achieved >98% accuracy, with differences in the 98.0-98.9% range

### Performance Insights

- **Layer Count**: Adding layers beyond 2 showed diminishing returns, suggesting potential overfitting for this dataset size
- **Filter Patterns**: In 3-layer models, decreasing filter patterns (32→16→8) slightly outperformed increasing patterns
- **Training Stability**: All models showed stable convergence within 10 epochs with proper train/validation split methodology

## Visual Results

![All_Models_Test_Accuracy_Comparison](https://github.com/user-attachments/assets/e793d0c7-42a0-48e9-909c-4a721a98cc39)


![Test Accuracy Comparison](results/All_Models_Test_Accuracy_Comparison.png)

*Comparative analysis showing test accuracy across all CNN architectures and batch sizes. The plot clearly demonstrates that CNN 2 Layers with batch size 50 achieves the highest performance.*

### Additional Training Visualizations

<img width="589" alt="summery" src="https://github.com/user-attachments/assets/005e1cd7-534d-4453-a7c3-ad0d89e8926c" />


**📁 ADD THESE IMAGES TO YOUR `results/` FOLDER:**

![CNN_3_Layers_(Increasing)_batch_250](https://github.com/user-attachments/assets/d3a8bb3b-132b-40fb-9e5f-6d9ffc1154e2)
![CNN_3_Layers_(Increasing)_batch_150](https://github.com/user-attachments/assets/0320d4bd-bde7-4c66-9ead-975643c73996)
![CNN_3_Layers_(Increasing)_batch_50](https://github.com/user-attachments/assets/d1b75913-e347-462f-885b-bb67b6443c51)
![CNN_3_Layers_(Decreasing)_batch_250](https://github.com/user-attachments/assets/a585cb16-fa7c-4862-8d3c-298570f3b1ee)
![CNN_3_Layers_(Decreasing)_batch_150](https://github.com/user-attachments/assets/4e5a3b60-6ee8-4536-9969-42b0e53b8a8e)
![CNN_3_Layers_(Decreasing)_batch_50](https://github.com/user-attachments/assets/fd4a7685-51a6-47bf-bf7e-9a5c880888a0)
![CNN_2_Layers_batch_250](https://github.com/user-attachments/assets/79b6e821-8f4a-4b2e-a460-59d937a09073)
![CNN_2_Layers_batch_150](https://github.com/user-attachments/assets/11170c19-1b20-41f7-8701-def49de6ba92)
![CNN_2_Layers_batch_50](https://github.com/user-attachments/assets/7c4177b5-2093-4da6-9375-4f627be2caa6)
![CNN_1_Layer_batch_250](https://github.com/user-attachments/assets/fbef4bc9-eed0-40e3-8ba5-9597bb7e4915)
![CNN_1_Layer_batch_150](https://github.com/user-attachments/assets/5644e740-c4ce-4a59-9b79-b38ac379348f)
![CNN_1_Layer_batch_50](https://github.com/user-attachments/assets/15c688d4-0707-4d65-a03d-abc2fbbc8b2b)


Each plot shows training and validation accuracy/loss progression over 10 epochs, providing insights into model convergence behavior and potential overfitting patterns.

</details>

## Future Work

### Potential Enhancements

- **Extended Architectures**: Test ResNet, DenseNet, or attention-based architectures
- **Regularization Analysis**: Compare dropout, batch normalization, and L1/L2 regularization effects
- **Dataset Expansion**: Evaluate on CIFAR-10, Fashion-MNIST, or custom datasets
- **Hyperparameter Optimization**: Implement grid search for learning rates, optimizers, and architectural parameters
- **Efficiency Metrics**: Add training time, memory usage, and inference speed comparisons

### Implementation Improvements

- **Configuration Management**: Add YAML/JSON config files for experiment parameters
- **Logging Integration**: Implement structured logging with TensorBoard integration
- **Cross-Validation**: Add k-fold validation for more robust performance estimates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Netanel Itzhak**  
- **LinkedIn**: [linkedin.com/in/netanelitzhak](https://www.linkedin.com/in/netanelitzhak)
- **Email**: ntitz19@gmail.com
- **GitHub**: [github.com/Net-AI-Git](https://github.com/Net-AI-Git)

---

*This project demonstrates systematic approach to neural network architecture evaluation and empirical machine learning research methodology.*
