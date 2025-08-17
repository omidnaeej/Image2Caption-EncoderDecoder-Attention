# Image2Caption-EncoderDecoder-Attention

This repository implements an **image captioning system** using **encoder–decoder architectures** with multiple improvements such as **beam search, teacher forcing, and attention mechanisms**.  
It is developed as part of coursework at the University of Tehran and is structured to provide a modular pipeline from **data preparation** to **model evaluation**.

---

## Project Structure
.
├── config/                # Configuration files (hyperparameters, logging)
├── data/                  # Data loading and preprocessing
│   ├── data_loader.py
│   └── dataset/
├── models/                # Model architectures and saved checkpoints
│   ├── model.py
│   └── saved_models/
├── notebooks/             # Jupyter notebooks
│   └── ImageCaptioning_final.ipynb
├── scripts/               # Training and evaluation scripts
│   ├── main.py
│   ├── train.py
│   └── evaluate.py
├── util/                  # Utility functions (metrics, visualization, helpers)
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── report.pdf             # Full assignment report

---

## Implemented Components

### 1. **Data Preparation**
- Used **Flickr8k dataset** (or similar) with Karpathy splits (train/val/test).
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

## Installation

```bash
git clone https://github.com/omidnaeej/Image2Caption-EncoderDecoder-Attention.git
cd Image2Caption-EncoderDecoder-Attention
pip install -r requirements.txt
```

---

## Usage

### 1. **Prepare Data**

Change configurations in `config/config.yaml` if you want.

```bash
python -m scripts.main
```

---

## Results

* Baseline: Encoder–Decoder (ResNet + LSTM/GRU).
* Improvements: BLEU scores improved with beam search, teacher forcing, and attention.
* Visualizations: Attention heatmaps show regions of images aligned with words in captions.

---

## References

* Xu et al., *Show, Attend and Tell* (2015)
* Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014)
* Pennington et al., *GloVe* (2014)
* Papineni et al., *BLEU: a method for automatic evaluation of machine translation* (2002)
* Karpathy splits: [Deep Visual-Semantic Alignments](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

---

## Acknowledgements

Developed as part of the **University of Tehran – Deep Learning (Image & Speech Applications)** course.
