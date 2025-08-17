from data.data_loader   import *
from models.model       import *
from scripts.train      import *
from scripts.evaluate   import *
from util.metrics       import *
from util.visualization import *

import torch

def evaluate_test_data(encoder, decoder, test_loader, vocab, device, 
                       decode_strategy='greedy', beam_size=2):
    """
    Evaluate the model on the test dataset and compute BLEU scores.
    """
    encoder.eval()
    decoder.eval()

    references = [] 
    hypotheses = []

    print(f"Evaluating on test set using {decode_strategy} decoding...")

    with torch.no_grad():
        for imgs, caps in test_loader:
            imgs, caps = imgs.to(device), caps

            for i in range(imgs.size(0)):
                if decode_strategy == 'greedy':
                    pred_tokens = greedy_decode(encoder, decoder, imgs[i], vocab, device=device)
                elif decode_strategy == 'beam':
                    pred_tokens = beam_search_decode(encoder, decoder, imgs[i], vocab, beam_size=beam_size, device=device)
                else:
                    raise ValueError(f"Unknown decode_strategy: {decode_strategy}")

                true_caption_ids = caps[i].tolist()
                true_tokens = [
                    vocab.itos[idx] for idx in true_caption_ids
                    if idx not in {vocab.stoi['<START>'], vocab.stoi['<END>'], vocab.stoi['<PAD>']}
                ]

                references.append([true_tokens]) 
                hypotheses.append(pred_tokens)  

    bleu1, bleu2 = compute_bleu_scores(references, hypotheses)

    print(f"Test Set BLEU-1: {bleu1:.4f}")
    print(f"Test Set BLEU-2: {bleu2:.4f}")

    return bleu1, bleu2

