from models.model       import *
from scripts.train      import *
from scripts.evaluate   import *
from util.metrics       import *
from util.visualization import *

from collections import Counter
import re
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os, json

def download_flickr8k(kaggle_dataset: str,download_path: str,
                      kaggle_json_path: str = None):
    """
    Downloads and unzips the Flickr8k dataset only if it doesn't already exist.
    Checks for the presence of a known file or directory.
    """
    os.makedirs(download_path, exist_ok=True)

    expected_folder = os.path.join(download_path, "Images")
    expected_file = os.path.join(download_path, "captions.txt")

    if os.path.exists(expected_folder) and os.path.exists(expected_file):
        print("[INFO] Flickr8k dataset already exists. Skipping download.")
        return

    print("[INFO] Flickr8k dataset not found. Downloading from Kaggle...")

    # Optional: set credentials if provided
    if kaggle_json_path and os.path.exists(kaggle_json_path):
        with open(kaggle_json_path) as f:
            creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]

    # Only import KaggleApi here
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        kaggle_dataset,
        path=download_path,
        unzip=True,
        quiet=False
    )

    print("[INFO] Flickr8k download and extraction complete.")

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return text.strip().split()

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized
        ]

class Flickr8kDataset(Dataset):
    def __init__(self, captions_file, images_dir, vocabulary, transform=None, max_length=50):
        self.df = pd.read_csv(captions_file)
        self.images_dir = images_dir
        self.transform = transform
        self.vocab = vocabulary
        self.max_length = max_length

        self.vocab.build_vocab(self.df['caption'].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.iloc[idx, 1]
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.images_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_tokens = [self.vocab.stoi["<START>"]]
        caption_tokens += self.vocab.numericalize(caption)
        caption_tokens += [self.vocab.stoi["<END>"]]

        caption_tensor = torch.tensor(caption_tokens[:self.max_length], dtype=torch.long)

        return image, caption_tensor

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        captions_padded = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return torch.stack(images), captions_padded

def get_loader(dataset, vocab, batch_size, shuffle=True, num_workers=2):
    pad_idx = vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MyCollate(pad_idx)
    )
    return loader

