# E-CaTCH Usage Guide

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/E-CaTCH-MisinformationDetection.git
cd E-CaTCH-MisinformationDetection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

1. Download the Fakeddit dataset and place it in the `data/raw/fakeddit` directory.
2. Run the preprocessing script:
```bash
python data/utils/preprocess.py --dataset fakeddit
```

## Training

To train the model on the Fakeddit dataset:

```bash
bash experiments/run_fakeddit.sh
```

This will:
- Train the model using the default configuration
- Save checkpoints in the `checkpoints` directory
- Log training metrics to TensorBoard

## Evaluation

To evaluate a trained model:

```bash
python evaluation/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_dir data/processed/fakeddit \
    --output_dir evaluation
```

This will generate:
- Evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Metrics visualization

## Configuration

The model and training parameters can be configured in `config/default.yaml`. Key parameters include:

- Model architecture (text encoder, image encoder, attention)
- Training hyperparameters (batch size, learning rate, epochs)
- Data processing parameters (image size, text length)

## Visualization

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs
```

## Directory Structure

- `config/`: Configuration files
- `data/`: Data processing and storage
- `datasets/`: Dataset loaders
- `models/`: Model architectures
- `training/`: Training scripts
- `evaluation/`: Evaluation scripts
- `experiments/`: Experiment scripts
- `scripts/`: Utility scripts
- `notebooks/`: Jupyter notebooks
- `tests/`: Unit tests
- `docs/`: Documentation 