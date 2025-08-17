from data.data_loader   import *
from models.model       import *
from util.metrics       import *

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import random
import torch

def plot_metrics(train_loss, val_loss, bleu1_scores, bleu2_scores):
    """
    Plots the training and validation loss, and BLEU-1 and BLEU-2 scores over epochs.
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, bleu1_scores, label='BLEU-1')
    plt.plot(epochs, bleu2_scores, label='BLEU-2')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Scores')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def visualize_predictions(encoder, decoder, dataset, vocab,
                          device, decode_strategy='greedy', beam_size=3,
                          num_samples=5):
    """
    Visualize N random images with ground truth and predicted captions.
    """
    encoder.eval()
    decoder.eval()

    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(18, 6))

    for i, idx in enumerate(indices):
        img, caption_tensor = dataset[idx]
        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            if decode_strategy == 'greedy':
                pred_tokens = greedy_decode(encoder, decoder, img, vocab, device=device)
            elif decode_strategy == 'beam':
                pred_tokens = beam_search_decode(encoder, decoder, img, vocab, beam_size=beam_size, device=device)
            else:
                raise ValueError(f"Unknown decode_strategy: {decode_strategy}")

        gt_ids = caption_tensor.tolist()
        gt_tokens = [
            vocab.itos[idx] for idx in gt_ids
            if idx not in {vocab.stoi['<START>'], vocab.stoi['<END>'], vocab.stoi['<PAD>']}
        ]

        img_vis = img.cpu().permute(1, 2, 0).numpy()
        img_vis = img_vis * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_vis = img_vis.clip(0, 1)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_vis)
        plt.axis('off')

        gt_caption = ' '.join(gt_tokens)
        pred_caption = ' '.join(pred_tokens)

        plt.title(
            f"GT:\n{gt_caption}\n\nPred:\n{pred_caption}",
            fontsize=8,
            wrap=True,
            loc='left'
        )

    plt.tight_layout()
    plt.show()

def visualize_attention_map(image_tensor, encoder, decoder, vocab, device, max_length=20):
    """
    Visualizes attention maps over image for each generated word.
    image_tensor: (3, H, W) — raw image (normalized)
    """
    encoder.eval()
    decoder.eval()

    assert decoder.use_attention, "Decoder must have use_attention=True to visualize attention."

    # Original image shape before normalization for resizing the attention map
    original_image_shape = image_tensor.shape[1:] # (H, W)

    image_input = image_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    with torch.no_grad():
        encoder_out = encoder(image_input)  # (1, 2048, 14, 14)
        encoder_out = encoder_out.view(1, decoder.encoder_dim, -1).permute(0, 2, 1) # (1, 196, 2048)
        num_pixels = encoder_out.size(1)

        # Initial hidden and cell states are based on the mean of the encoder output
        h, c = decoder.init_hidden_state(encoder_out.mean(dim=1)) # mean over pixels (1, 2048)

        # Start decoding
        word_ids = [vocab.stoi["<START>"]]
        alphas_list = []
        generated_words = []

        # Initial inputs for the first step
        prev_word = torch.LongTensor([word_ids[-1]]).to(device) # <START> token
        embedding = decoder.embedding(prev_word) # (1, embed_dim)

        for t in range(max_length):
            attn_weighted, alpha = decoder.attention(encoder_out, h) # encoder_out is (1, num_pixels, 2048), h is (1, decoder_dim)

            alphas_list.append(alpha.view(14, 14).cpu().numpy()) # store attention shape (14, 14)

            gate = decoder.sigmoid(decoder.f_beta(h))
            attn_weighted = gate * attn_weighted # (1, encoder_dim)

            # Prepare LSTM input: [embedding, attention_weighted_encoding]
            lstm_input = torch.cat([embedding, attn_weighted], dim=1) # (1, embed_dim + encoder_dim)

            # Perform LSTM step
            h, c = decoder.decode_step(lstm_input, (h, c)) # h, c are (1, decoder_dim)

            scores = decoder.fc(h) 
            predicted_id = scores.argmax(dim=1).item()
            word_ids.append(predicted_id)

            word = vocab.itos[predicted_id]

            if word == "<END>":
                alphas_list.pop()
                break

            generated_words.append(word)

            embedding = decoder.embedding(torch.LongTensor([predicted_id]).to(device)) # (1, embed_dim)


    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # unnormalize
    image = np.clip(image, 0, 1)

    num_words_to_plot = len(generated_words) 
    
    fig = plt.figure(figsize=(num_words_to_plot * 3, 6)) 


    ax = fig.add_subplot(2, num_words_to_plot + 1, 1)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title("Input Image")

    # Plot attention maps for each generated word
    for t in range(num_words_to_plot):
        ax = fig.add_subplot(2, num_words_to_plot + 1, t + 2) 

        alpha = alphas_list[t] 
        
        alpha_img = Image.fromarray(alpha * 255).resize(original_image_shape[::-1], Image.BICUBIC) # Resize takes (width, height)
        alpha_resized = np.array(alpha_img) / 255.0 # Convert back to [0, 1]

        heatmap = cm.Greys(alpha_resized)[:, :, :3] # Get RGB channels

        overlay = 0.6 * image + 0.4 * heatmap
        ax.imshow(overlay)
        ax.set_title(generated_words[t])
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
    