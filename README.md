# Image2Caption-EncoderDecoder-Attention

This repository implements an **image captioning system** using **encoder–decoder architectures** with multiple improvements such as **beam search, teacher forcing, and attention mechanisms**.  
It is developed as part of coursework at the University of Tehran and is structured to provide a modular pipeline from **data preparation** to **model evaluation**.

---

## Repository Contents
- `config/`: YAML files for training configurations and logging  
- `data/`: Data loader  
- `models/`: Model architecture definitions (EncoderCNN, DecoderRNN, Attention)  
- `notebooks/`: Jupyter notebook for exploration and experiments  
- `scripts/`: Training, evaluation, and entry-point scripts  
- `utils/`: Metrics computation and visualization tools  
- `report.pdf`: Original report (in Persian)  

---

## Project Overview

### 1. **Data Preparation**
- Used **Flickr8k dataset**.
- Preprocessing includes:
  - Tokenization of captions (without pre-built tokenizers).
  - Adding `<START>` and `<END>` tokens to sequences.
  - Padding sequences for batching.
- Custom **DataLoader** implementation for efficient batching.

### 2. **Baseline Encoder–Decoder Model**
- **Encoder**: Pre-trained **ResNet-101** CNN backbone (feature extractor).
- **Decoder**: RNN-based (**LSTM / GRU**) with embedding layer to generate captions.

### 3. **Training**
- Loss: **Cross-Entropy**
- Optimizer: **Adam**
- Techniques:
  - Early Stopping
  - Hyperparameter tuning (batch size, learning rate, embedding dimension, etc.)
- Training curves for loss and validation BLEU score are logged.

### 4. **Evaluation Metrics**
- Implemented **BLEU score** for caption evaluation.
- Training stops when BLEU stops improving on validation set.

### 5. **Model Improvements**
- **Beam Search Decoding** instead of greedy approach for better captions.
- **Teacher Forcing** to stabilize RNN training.
- **Attention Mechanism**:
  - Implemented Bahdanau-style attention.
  - Improves caption accuracy and interpretability.
  - Visualizations of attention maps are included.

### 6. **Architectural Extensions**
- Alternative backbones: Replace ResNet-101 with others (e.g., ResNet-50, EfficientNet).
- Alternative decoders: Try GRU vs LSTM, compare performance.

---

## Setup
Clone the repository:
```bash
git clone https://github.com/omidnaeej/Image2Caption-EncoderDecoder-Attention.git
cd Image2Caption-EncoderDecoder-Attention
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage
Change configurations in `config/config.yaml` if you want. Then run the main script:

```bash
python -m scripts.main
```

---

## Results

* Baseline: Encoder–Decoder (ResNet + LSTM/GRU).
* Improvements: BLEU scores improved with beam search, teacher forcing, and attention.
* Visualizations: Attention heatmaps show regions of images aligned with words in captions.
