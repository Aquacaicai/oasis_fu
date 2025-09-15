# OASIS Federated Learning Project

This project implements federated learning with unlearning capabilities for the OASIS brain scan dataset, focusing on dementia classification.

## Features

- **Federated Learning (FL)**: Distributed training across multiple parties
- **Federated Unlearning**: Remove influence of specific party's data
- **Poison Attack Resistance**: Built-in backdoor detection and mitigation
- **GPU/CUDA Support**: Automatic device detection and optimization

## Dataset

The project uses the OASIS (Open Access Series of Imaging Studies) dataset with the following classes:
- Mild Dementia
- Moderate Dementia  
- Non Demented
- Very mild Dementia

## Core Components

### Models
- `model.py`: Convolutional Neural Network (FLNet) for brain scan classification

### Training
- `fl.py`: Main federated learning with FedAvg and Retrain baselines
- `local_train.py`: Local training implementation for each party
- `fusion.py`: Model aggregation strategies (FedAvg, FusionRetrain)

### Unlearning
- `unlearn.py`: Federated unlearning implementation to remove party influence
- `retrain.py`: Baseline retraining from scratch without target party

### Utilities
- `utils.py`: Evaluation metrics, backdoor injection, helper functions

## Usage

### Basic Federated Learning
```bash
python fl.py --data_dir ./imagesoasis --num_parties 5 --rounds 10 --batch_size 32
```

### Retrain Baseline (without party 0)
```bash
python retrain.py --data_dir ./imagesoasis --num_parties 5 --rounds 10
```

### Federated Unlearning
```bash
python unlearn.py --data_dir ./imagesoasis --models_path ./outputs/imagesoasis/fl/models.pth --unlearn_epochs 5
```

## Parameters

### Common Parameters
- `--data_dir`: Path to OASIS dataset directory
- `--num_parties`: Number of federated parties (default: 5)
- `--rounds`: Number of federated rounds (default: 10)
- `--batch_size`: Training batch size (default: 32)
- `--image_size`: Input image size (default: 128)
- `--device`: Computing device ('cuda' or 'cpu', auto-detected)

### Poison Attack Parameters
- `--poison_frac`: Fraction of data to poison (default: 0.1)
- `--poison_target`: Target class for backdoor (default: 0)

### Unlearning Parameters
- `--unlearn_epochs`: Number of unlearning epochs (default: 5)
- `--lr`: Learning rate for unlearning (default: 0.01)
- `--distance_threshold`: Distance constraint for unlearning (default: 2.2)

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8GB+ RAM, 4GB+ VRAM for GPU training
- **Storage**: Sufficient space for OASIS dataset (~several GB)

## Output

Results are saved in `outputs/imagesoasis/` with subdirectories:
- `fl/`: Federated learning models and logs
- `retrain/`: Retrain baseline results  
- `unlearn/`: Unlearning results

## Dependencies

- PyTorch >= 2.0
- torchvision
- numpy
- pillow

Install with:
```bash
pip install torch torchvision numpy pillow
```

## Architecture

The system implements a privacy-preserving federated learning framework where:

1. **Data Distribution**: Dataset split across multiple parties
2. **Local Training**: Each party trains on local data
3. **Secure Aggregation**: Model updates combined without sharing raw data
4. **Unlearning**: Selective removal of party influence post-training
5. **Attack Mitigation**: Backdoor detection and removal capabilities

## Research Applications

This codebase supports research in:
- Federated learning for medical imaging
- Machine unlearning in distributed settings
- Privacy-preserving healthcare AI
- Adversarial robustness in federated systems
