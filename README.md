# E-CaTCH: Event-Centric Cross-Modal Attention with Temporal Consistency and Class-Imbalance Handling for Misinformation Detection

This repository contains the official implementation of E-CaTCH, a novel framework for multimodal misinformation detection that integrates event-centric modeling, temporal dynamics, and adaptive class imbalance handling.

## Overview

E-CaTCH addresses three key challenges in misinformation detection:
1. Cross-modal inconsistencies between text and images
2. Temporal evolution of misinformation narratives
3. Severe class imbalance in real-world datasets

The framework achieves this through:
- Event-centric modeling using pseudo-events
- Hierarchical attention-based fusion with soft gating
- Temporal trend encoding with momentum signals
- Adaptive class weighting and hard-example mining

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU acceleration)
- NVIDIA H100 GPUs (recommended for optimal performance)
- 32GB+ RAM
- 100GB+ free disk space

### Setup
```bash
# Clone the repository
git clone https://github.com/Yegi03/E-CaTCH.git
cd E-CaTCH

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Fakeddit Dataset
1. Download the Fakeddit dataset from [here](https://github.com/entitize/Fakeddit)
2. Place the data in `data/fakeddit/`
3. Run preprocessing script:
```bash
python scripts/preprocess_fakeddit.py
```

### India Elections Dataset
1. Download the dataset from [here](https://github.com/mesnico/WhatsApp-Image-Dataset)
2. Place in `data/india_elections/`
3. Run preprocessing:
```bash
python scripts/preprocess_india.py
```

### COVID-19 MISINFOGRAPH Dataset
1. Download from MediaEval 2020 challenge
2. Place in `data/covid19/`
3. Run preprocessing:
```bash
python scripts/preprocess_covid.py
```

## Training

### Basic Training
```bash
python train.py --config configs/default.yaml
```

### Training with Custom Parameters
```bash
python train.py \
    --dataset fakeddit \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --gpu_ids 0,1
```

### Key Training Parameters
- `--dataset`: Dataset to use (fakeddit/india/covid19)
- `--batch_size`: Batch size for training
- `--learning_rate`: Initial learning rate
- `--num_epochs`: Number of training epochs
- `--gpu_ids`: GPU IDs to use (comma-separated)
- `--window_size`: Size of temporal windows
- `--window_overlap`: Overlap between windows
- `--num_events`: Number of pseudo-events

## Evaluation

### Test on Single Dataset
```bash
python evaluate.py --config configs/default.yaml --test_data path/to/test_data
```

### Cross-Dataset Evaluation
```bash
python evaluate.py --config configs/default.yaml --cross_dataset
```

## Model Architecture

E-CaTCH consists of several key components:

1. **Feature Extraction**
   - Text: BERT-base transformer for contextual text encoding
   - Image: ResNet-152 for visual feature extraction
   - Event Clustering: BERT-based semantic clustering for pseudo-event formation

2. **Attention Fusion**
   - Intra-modal self-attention for feature refinement
   - Cross-modal attention with soft gating for adaptive fusion
   - Hierarchical attention mechanism for multi-level feature integration

3. **Temporal Modeling**
   - Overlapping window segmentation for continuous temporal coverage
   - Trend-aware LSTM with momentum signals
   - Semantic shift detection between windows
   - Temporal consistency regularization

4. **Classification**
   - Adaptive class weighting for imbalance handling
   - Hard-example mining for challenging cases
   - Event-level prediction with temporal aggregation

## Results

### Performance on Fakeddit Dataset
| Metric | Value |
|--------|-------|
| Accuracy | 95.50% |
| Precision | 95.70% |
| Recall | 95.30% |
| F1-Score | 95.50% |
| AUC-ROC | 97.50% |

### Performance on India Elections Dataset
| Metric | Value |
|--------|-------|
| Accuracy | 89.80% |
| Precision | 90.30% |
| Recall | 89.00% |
| F1-Score | 89.60% |
| AUC-ROC | 92.50% |

### Performance on COVID-19 Dataset
| Metric | Value |
|--------|-------|
| Accuracy | 89.50% |
| Precision | 89.90% |
| Recall | 88.50% |
| F1-Score | 89.10% |
| AUC-ROC | 93.80% |

### Cross-Dataset Generalization
| Training → Testing | Accuracy |
|-------------------|----------|
| Fakeddit → India | 88.2% |
| Fakeddit → COVID-19 | 87.8% |
| India → Fakeddit | 89.5% |
| COVID-19 → Fakeddit | 89.1% |

### Computational Efficiency
- GPU Utilization: 98%
- Training Time (Fakeddit): 14.8 hours
- Throughput: 989 TFLOPS

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{ecatch2024,
  title={E-CaTCH: Event-Centric Cross-Modal Attention with Temporal Consistency and Class-Imbalance Handling for Misinformation Detection},
  author={Mousavi, Ahmad and Abdollahinejad, Yeganeh and Corizzo, Roberto and Japkowicz, Nathalie and Boukouvalas, Zois},
  journal={Information Fusion},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- We thank the authors of the Fakeddit, India Elections, and COVID-19 MISINFOGRAPH datasets for making their data publicly available.

## Contact

For questions and issues, please open an issue in this repository or contact:
- Ahmad Mousavi (mousavi@american.edu)
- Yeganeh Abdollahinejad (yza5171@psu.edu)

## Contributing

We welcome contributions to improve E-CaTCH! Please feel free to submit pull requests or open issues for any bugs or feature requests. 
