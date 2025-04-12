# HyperForensics++: A Framework for Hyperspectral Image Forgery Detection and Adversarial Attacks

## Overview

HyperForensics++ is a comprehensive framework for hyperspectral image (HSI) forgery detection and adversarial attack research. This project provides a systematic environment for evaluating the robustness of forgery detection models and developing advanced adversarial attack methods tailored for hyperspectral imagery.

The framework enables researchers and practitioners to:
- Train and evaluate state-of-the-art hyperspectral forgery detection models
- Generate adversarial examples using multiple attack strategies
- Test model robustness against various attack domains (spatial, spectral, and hybrid)
- Evaluate both segmentation performance and attack effectiveness using comprehensive metrics

## Key Features

- **Complete Training Pipeline**: End-to-end training and evaluation workflow for HSI forgery detection models
- **Advanced Attack Methods**: Implementation of multiple adversarial attack algorithms including FGSM, PGD, CW, and DeepFool
- **Domain-Specific Attacks**: Specialized attacks targeting spatial domain, spectral domain, or hybrid approaches
- **Hyperspectral Data Processing**: Specialized data loading and preprocessing for hyperspectral imagery
- **Comprehensive Evaluation**: Detailed segmentation and adversarial attack evaluation metrics
- **Visualization Tools**: Rich visualization capabilities for understanding detection results and attack effects
- **Configurable Framework**: Highly customizable through configuration files

## Directory Structure

```
HyperForensics++/
├── config/             # Configuration files
├── models/             # Model definitions
├── datasets/           # Dataset handling and loading
├── attacks/            # Adversarial attack implementations
│   ├── basic_attacks.py      # Basic attack methods (FGSM, PGD)
│   ├── spatial_attacks.py    # Spatial domain attacks
│   ├── spectral_attacks.py   # Spectral domain attacks
│   ├── hybrid_attacks.py     # Combined spatial and spectral attacks
│   ├── attack_utils.py       # Attack utility functions
│   └── utils.py              # General utility functions for attacks
├── utils/              # Utility functions
├── metrics/            # Evaluation metrics
├── scripts/            # Script tools
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── generate_adversarial.py # Adversarial sample generation
│   └── visualization.py      # Visualization utilities
├── checkpoints/        # Model checkpoints
├── logs/               # Training logs
└── results/            # Output results directory
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (for GPU acceleration)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/HSI_Adv_Atk.git
cd HSI_Adv_Atk/HyperForensics++

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Instructions

### Training Models

```bash
# Training with all configurations
python scripts/train.py --config_path ./config/train_config.yaml

# Training with specific configuration(s)
python scripts/train.py --config_path ./config/train_config.yaml --configs config1

# Training with multiple specific configurations
python scripts/train.py --config_path ./config/train_config.yaml --configs config1 config2

# Resuming training from a checkpoint
python scripts/train.py --config_path ./config/train_config.yaml --configs config1 --checkpoint ./logs/weights/exp_20240320_120000/checkpoint_epoch_50.pth
```

Notes:
- When using the `--configs` parameter, the program automatically includes the "Origin" configuration for validation
- You can also directly modify the `CONFIGS` list in `train_config.yaml` to select configurations to use
- The training process automatically saves the following weights:
  - `best_miou.pth`: Model weights with the best mIoU
  - `best_acc.pth`: Model weights with the best accuracy
  - `last.pth`: Model weights from the last epoch
  - `checkpoint_epoch_X.pth`: Checkpoints every 5 epochs

### Remote Training

To prevent interruption due to connection issues during remote training, you can use one of the following methods:

1. Using `screen` (recommended):
```bash
# Create a new screen session
screen -S training

# Run the training script in the screen session
python scripts/train.py --config_path ./config/train_config.yaml --configs config1

# Press Ctrl+A, then D to detach the screen session
# Use screen -ls to view all sessions
# Use screen -r training to reconnect to the session
```

2. Using `tmux`:
```bash
# Create a new tmux session
tmux new -s training

# Run the training script in the tmux session
python scripts/train.py --config_path ./config/train_config.yaml --configs config1

# Press Ctrl+B, then D to detach the tmux session
# Use tmux ls to view all sessions
# Use tmux attach -t training to reconnect to the session
```

3. Using `nohup`:
```bash
nohup python scripts/train.py --config_path ./config/train_config.yaml --configs config1 > training.log 2>&1 &
```

All training logs are saved in the `/ssd1/dannyliu/HSI_Adv_Atk/Log` directory, with filenames in the format `exp_{timestamp}_lr{learning_rate}_bs{batch_size}.log`.

### Generating Adversarial Examples

The `generate_adversarial.py` script provides a comprehensive interface for creating adversarial examples using different attack methods and domains:

```bash
# Basic usage
python scripts/generate_adversarial.py --attack_method pgd --attack_domain spatial

# Detailed usage with parameters
python scripts/generate_adversarial.py --attack_method fgsm --attack_domain spectral --eps 0.1 --batch_size 4 --save_dir results/adversarial

# Example of a DeepFool attack in the spectral domain
python scripts/generate_adversarial.py --attack_method deepfool --attack_domain spectral --gpu_id 0 --eps 0.3

# Example of a CW attack in the hybrid domain
python scripts/generate_adversarial.py --attack_method cw --attack_domain hybrid --steps 100 --spatial_weight 0.7
```

Key parameters:
- `--attack_method`: Choose from `fgsm`, `pgd`, `deepfool`, or `cw`
- `--attack_domain`: Choose from `spatial`, `spectral`, or `hybrid`
- `--eps`: Maximum perturbation size (default: 0.03)
- `--steps`: Number of iterations for iterative attacks (default: 40)
- `--adaptive_eps`: Use adaptive perturbation size (enabled by default)
- `--perceptual_constraint`: Apply perceptual constraints to the adversarial examples (enabled by default)

Advanced options:
- `--spatial_weight`: Weight of spatial component in hybrid attacks (0-1, default: 0.5)
- `--target_bands`: Specific bands to attack (comma-separated, e.g., "10,20,30")
- `--spectral_importance`: Use spectral importance analysis to select bands (enabled by default)
- `--spectral_threshold`: Cumulative importance threshold for band selection (default: 0.7)

### Recent Improvements to Adversarial Attacks

Recent updates have significantly enhanced the effectiveness and stability of adversarial attacks:

1. **Data Range Adaptive Amplification Factor**:
   - Dynamically adjusts perturbation magnitude based on the hyperspectral data range
   - Automatically scales from 20x to 100x based on the actual data range (typically in thousands)
   - Significantly improves attack effectiveness for high-dynamic-range hyperspectral data

2. **Unified Domain Processing**:
   - All computations now occur in the normalized domain
   - Eliminates precision loss from domain transformations
   - Reports perturbation magnitude in both normalized and original domains

3. **Segmentation Model Support**:
   - Added full support for segmentation models with output dimensions [B, C, H, W]
   - Implemented spatial averaging for confidence computation in segmentation models
   - Modified gradient calculations for accurate attack targeting

4. **Numerical Stability Enhancements**:
   - Added zero-division protection mechanisms
   - Improved handling of gradients with near-zero magnitudes
   - Enhanced stability in DeepFool's decision boundary calculation

5. **Enhanced Diagnostic Output**:
   - Added detailed perturbation statistics including L∞ and L2 norms
   - Reports both normalized and original domain perturbation sizes
   - Displays attack success rate and band/pixel selection ratios

### Evaluation

```bash
# Evaluating the original dataset
python scripts/evaluate.py --config_path ./config/test_config.yaml --mode original

# Evaluating adversarial examples
python scripts/evaluate.py --config_path ./config/test_config.yaml --mode adversarial --adv_dir ./results/adversarial/pgd_spatial_20240425_120000/images --adv_flist ./results/adversarial/pgd_spatial_20240425_120000/filelist.txt
```

## Supported Attack Methods

### Basic Methods
- **FGSM** (Fast Gradient Sign Method): Creates adversarial examples by taking a single step in the direction of the gradient sign
- **PGD** (Projected Gradient Descent): An iterative variant of FGSM that projects perturbations onto the ε-ball
- **CW** (Carlini-Wagner): Optimization-based attack that creates adversarial examples with minimal perturbation
- **DeepFool**: Iteratively finds the minimal perturbation needed to cross the decision boundary

### Attack Domains
- **Spatial Domain Attacks**: Focus on modifying the spatial structure of the image, applying consistent spatial perturbation patterns across all spectral bands
- **Spectral Domain Attacks**: Focus on modifying spectral features of specific bands while maintaining relative spatial structure
- **Hybrid Domain Attacks**: Combine spatial and spectral domain attacks, applying perturbations in both dimensions

## Implementation Details

### FGSM Implementation

The FGSM attack is implemented with several enhancements:
- Data range adaptive scaling factor (10-50x)
- Direct operation in the normalized domain
- Support for spatial, spectral, and hybrid domains
- Comprehensive perturbation statistics

```python
# Example of FGSM attack in the spectral domain
adv_images = fgsm_spectral_attack(
    model, images, labels, eps=0.1, criterion=F.cross_entropy, 
    device='cuda', mean=mean, std=std, target_bands=important_bands
)
```

### PGD Implementation

The PGD attack includes:
- Multi-step iterative optimization
- Configurable step size and iteration count
- Projection onto perturbation constraints
- Domain-specific implementations

```python
# Example of PGD attack in the spatial domain
adv_images = pgd_spatial_attack(
    model, images, labels, eps=0.1, alpha=0.01, steps=40,
    criterion=F.cross_entropy, device='cuda', mean=mean, std=std
)
```

### DeepFool Implementation

DeepFool has been enhanced with:
- Improved gradient computation for segmentation models
- Data range adaptive scaling factor (20-100x)
- Selective perturbation based on importance masks
- Numerical stability safeguards

```python
# Example of DeepFool attack in the hybrid domain
adv_images = deepfool_hybrid_attack(
    model, images, num_classes=2, max_iter=50, overshoot=0.02,
    device='cuda', mean=mean, std=std, spatial_weight=0.6
)
```

### CW Implementation

The CW attack features:
- Confidence parameter (κ) for controlling attack strength
- Learning rate and iteration count configuration
- Domain-specific implementations
- Perceptual constraint options

```python
# Example of CW attack
adv_images = cw_hybrid_attack(
    model, images, labels, c=0.001, kappa=10.0, steps=200, lr=0.01,
    device='cuda', mean=mean, std=std, spatial_weight=0.5
)
```

## Evaluation Metrics

### Segmentation Evaluation Metrics
- Pixel Accuracy: Percentage of correctly classified pixels
- Mean IoU (Intersection over Union): Average IoU across all classes
- Class F1 Score: Harmonic mean of precision and recall

### Adversarial Attack Evaluation Metrics
- Attack Success Rate: Percentage of successful misclassifications
- Perturbation Norms: L0, L1, L2, L∞ norms to measure perturbation magnitude
- Signal-to-Noise Ratio: Measure of perturbation visibility
- Structural Similarity: Measure of structural preservation

## Comparison of Attack Methods

| Attack Method | Spatial Domain | Spectral Domain | Hybrid Domain | Computational Complexity | Success Rate |
|--------------|----------------|-----------------|---------------|--------------------------|--------------|
| FGSM         | ✓              | ✓               | ✓             | Low                      | Medium       |
| PGD          | ✓              | ✓               | ✓             | Medium                   | High         |
| CW           | ✓              | ✓               | ✓             | High                     | High         |
| DeepFool     | ✓              | ✓               | ✓             | Medium                   | Medium       |

### Advantages of Hybrid Domain Attacks

Hybrid domain attacks combine the strengths of spatial and spectral domain attacks by perturbing in both dimensions simultaneously, offering:

1. **Higher Success Rate**: Exploits vulnerabilities in both spatial and spectral dimensions
2. **Improved Stealth**: Distributes perturbation across two dimensions, making it less detectable
3. **Better Generalization**: More effective against various defense mechanisms

When using hybrid attacks, you can adjust the balance between spatial and spectral components through the `SPATIAL_WEIGHT` parameter, as well as select target bands.

## Custom Configurations

You can customize attack parameters by modifying the `config/attack_config.yaml` file:

```yaml
# config/attack_config.yaml
ATTACK:
  # Common parameters
  EPS: 0.1
  STEPS: 40
  RANDOM_START: True
  
  # Specific attack parameters
  PGD:
    ALPHA: 0.01
  
  CW:
    C: 0.001
    KAPPA: 10.0
    LR: 0.01
    
  DEEPFOOL:
    MAX_ITER: 50
    OVERSHOOT: 0.02
  
  # Domain-specific parameters
  SPECTRAL:
    TARGET_BANDS: null  # Automatically select important bands
    SPECTRAL_THRESHOLD: 0.7
  
  HYBRID:
    SPATIAL_WEIGHT: 0.5  # Weight of spatial component (0-1)
```

## Visualization

The framework includes extensive visualization capabilities:

```bash
# Visualize adversarial examples
python scripts/visualization.py --input_dir ./results/adversarial/pgd_spatial_20240425_120000/images --output_dir ./results/visualizations
```

This will generate:
- RGB visualizations of original and adversarial images
- Perturbation visualization with enhanced visibility
- Prediction difference maps
- Spectral signature plots for selected pixels

## Best Practices for Repository Management

For sustainable management of this project repository:

1. **Feature Branches**:
   - Create new branches only for complete features or significant changes
   - Use naming convention `feature/feature-name` or `fix/issue-description`
   - Merge to main only after testing

2. **Use .gitignore**:
   - Exclude generated files and large outputs
   - Example contents for .gitignore:
   ```
   results/adversarial/
   *.pth
   *.pt
   *.npy
   ```

3. **Git LFS for Large Files**:
   - For version-controlled large files, use Git Large File Storage
   - `git lfs install`
   - `git lfs track "*.pth" "*.npy"`

4. **Version Tagging**:
   - Tag important versions: `git tag -a v1.0.0 -m "Stable release 1.0.0"`
   - Push tags: `git push origin --tags`

5. **Clean Commit Messages**:
   - Format: `[Module] Brief description (under 50 chars)`
   - Example: `[DeepFool] Fix gradient computation and segmentation model support`

## Citation

If you use the HyperForensics++ framework in your research, please consider citing:

```
@article{your_paper,
  title={HyperForensics++: A Framework for Hyperspectral Image Forgery Detection and Adversarial Attacks},
  author={Your Name},
  journal={Your Journal},
  year={2023}
}
```

## License

This project is licensed under the MIT License. 