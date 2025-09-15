# Federated Unlearning with ImagesOasis

## Setup
```bash
pip install torch torchvision numpy matplotlib
```

## Data
Download ImagesOasis from Kaggle and extract:
```
./data/imagesoasis/class1/...
./data/imagesoasis/class2/...
```

## Run Experiments

### Federated Learning (FL with poisoned party)
```bash
python fl.py --data_dir ./data/imagesoasis --num_parties 5 --rounds 10 --poison_frac 0.1 --poison_target 0
```

### Retrain Baseline
```bash
python retrain.py --data_dir ./data/imagesoasis --num_parties 5 --rounds 10
```

### Unlearning
```bash
python unlearn.py --data_dir ./data/imagesoasis --num_parties 5 --rounds 10 --unlearn_epochs 5
```

## Outputs
- `outputs/fl/` → FedAvg models and logs
- `outputs/retrain/` → Retrain models and logs
- `outputs/unlearn/` → Unlearning models and logs

## You can modify
- **model.py** → network architecture
- **utils.py:add_backdoor** → trigger pattern
- **data split** in scripts
- Hyperparameters (lr, batch size, epochs, etc.)

## Must keep
- Fusion logic
- Utils distance and evaluation functions
- Clean vs Backdoor test evaluation
