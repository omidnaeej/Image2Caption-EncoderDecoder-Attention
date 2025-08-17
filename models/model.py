import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, fine_tune_cnn=False):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(fine_tune_cnn)

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        return features  # shape: (batch_size, 2048, 14, 14)

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Encoder saved to {path}")

    def load(self, path, device='cuda'):
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Encoder loaded from {path}")

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear layer to transform encoder output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Linear layer to transform decoder output
        self.full_att = nn.Linear(attention_dim, 1)  # Linear layer to calculate attention scores
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        attention_dim=512,
        dropout=0.5,
        use_attention=True
    ):
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.use_attention = use_attention

        # Attention layer (optional)
        if self.use_attention:
            self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Embedding + LSTM core
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + (encoder_dim if use_attention else 0), decoder_dim, bias=True
        )

        # Init hidden/cell states
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Gating scalar for attention
        if self.use_attention:
            self.f_beta = nn.Linear(decoder_dim, encoder_dim)
            self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        h = self.init_h(encoder_out)
        c = self.init_c(encoder_out)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths, teacher_forcing=False):
        batch_size = encoder_out.size(0)
        # encoder_dim = encoder_out.size(1) if not self.use_attention else encoder_out.size(-1)
        encoder_dim = self.encoder_dim
        vocab_size = self.vocab_size

        # ---------------- Sorting ----------------
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # ---------------- Attention vs. No-Attention ----------------
        if self.use_attention:
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (B, num_pixels, 2048)
            num_pixels = encoder_out.size(1)
            h, c = self.init_hidden_state(encoder_out.mean(dim=1))       # (B, 2048)
        else:
            # encoder_out shape is (B, 2048, H, W)
            mean_encoder_out = encoder_out.mean(dim=[2, 3])              # (B, 2048)
            h, c = self.init_hidden_state(mean_encoder_out)

        # ---------------- Embedding ----------------
        embeddings = self.embedding(encoded_captions)
        decode_lengths = (caption_lengths - 1).tolist()

        max_decode_len = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_decode_len, vocab_size).to(encoder_out.device)
        alphas = None
        if self.use_attention:
            alphas = torch.zeros(batch_size, max_decode_len, encoder_out.size(1)).to(encoder_out.device)

        prev_words = embeddings[:, 0, :]  # Start with <START> embedding

        # ---------------- Decoding ----------------
        for t in range(max_decode_len):
            batch_size_t = sum([l > t for l in decode_lengths])

            if teacher_forcing:
                emb_t = embeddings[:batch_size_t, t, :]
            else:
                if t == 0:
                    emb_t = prev_words[:batch_size_t]
                else:
                    prev_tokens = predictions[:batch_size_t, t - 1, :].argmax(dim=1)  # (B,)
                    emb_t = self.embedding(prev_tokens)  # (B, embed_dim)

            # emb_t = embeddings[:batch_size_t, t, :]

            if self.use_attention:
                attn_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
                attn_weighted_encoding = gate * attn_weighted_encoding
                lstm_input = torch.cat([emb_t, attn_weighted_encoding], dim=1)
            else:
                lstm_input = emb_t

            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout_layer(h))

            predictions[:batch_size_t, t, :] = preds
            if self.use_attention:
                alphas[:batch_size_t, t, :] = alpha

        if self.use_attention:
            return predictions, encoded_captions, decode_lengths, alphas, sort_ind
        else:
            return predictions, encoded_captions, decode_lengths, None, sort_ind



    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Decoder saved to {path}")

    def load(self, path, device='cuda'):
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Decoder loaded from {path}")

def greedy_decode(encoder, decoder, image, vocab, max_length=20, device='cuda'):
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        # Encode image
        encoder_out = encoder(image.unsqueeze(0).to(device))  # (1, 2048, 14, 14) if no attention, or (1, num_pixels, 2048) if attention

        # Init hidden states
        if decoder.use_attention:
             # If using attention, the encoder_out is (1, num_pixels, 2048)
             # init_hidden_state expects (B, num_pixels, 2048) but takes mean inside
             h, c = decoder.init_hidden_state(encoder_out) # Pass as is, init_hidden_state handles mean
        else:
             # If not using attention, encoder_out is (1, 2048, H, W).
             # Average spatially to get (1, 2048) before passing to init_hidden_state
             mean_encoder_out = encoder_out.mean(dim=[2, 3]) # (1, 2048)
             h, c = decoder.init_hidden_state(mean_encoder_out)


        # First input = <START>
        word = torch.tensor([vocab.stoi['<START>']]).to(device)
        embeddings = decoder.embedding(word).unsqueeze(0)  # (1, 1, embed_dim)

        seq = []

        for _ in range(max_length):
            if decoder.use_attention:
                # encoder_out here is (1, num_pixels, 2048)
                attn_weighted, _ = decoder.attention(encoder_out, h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                attn_weighted = gate * attn_weighted
                lstm_input = torch.cat([embeddings.squeeze(1), attn_weighted], dim=1)
            else:
                # No attention, LSTM input is just embeddings
                lstm_input = embeddings.squeeze(1) # (1, embed_dim)


            h, c = decoder.decode_step(lstm_input, (h, c))
            preds = decoder.fc(h)
            predicted = preds.argmax(1)
            if predicted.item() == vocab.stoi['<END>']:
                break

            seq.append(predicted.item())
            embeddings = decoder.embedding(predicted).unsqueeze(0) # Prepare embedding for next step

        return [vocab.itos[idx] for idx in seq]

def beam_search_decode(encoder, decoder, image, vocab, beam_size=2, max_length=20, device='cuda'):
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        k = beam_size
        vocab_size = len(vocab)

        # Encode image
        encoder_out = encoder(image.unsqueeze(0).to(device))  # (1, C, H, W) or (1, num_pixels, 2048)
        if decoder.use_attention:
            encoder_out = encoder_out.view(1, -1, decoder.encoder_dim)
            num_pixels = encoder_out.size(1)
            encoder_out = encoder_out.expand(k, num_pixels, decoder.encoder_dim)
        else:
            encoder_out = encoder_out.expand(k, *encoder_out.shape[1:])

        # Init hidden state
        h, c = decoder.init_hidden_state(
            encoder_out.mean(dim=1) if decoder.use_attention else encoder_out.mean(dim=[2, 3])
        )

        # Start decoding
        seqs = torch.LongTensor([[vocab.stoi['<START>']]] * k).to(device)
        top_k_scores = torch.zeros(k, 1).to(device)

        complete_seqs = []
        complete_seqs_scores = []

        step = 1
        while True:
            embeddings = decoder.embedding(seqs[:, -1])  # (k, embed_dim)

            if decoder.use_attention:
                attn_weighted, _ = decoder.attention(encoder_out, h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                attn_weighted = gate * attn_weighted
                lstm_input = torch.cat([embeddings, attn_weighted], dim=1)
            else:
                lstm_input = embeddings

            h, c = decoder.decode_step(lstm_input, (h, c))
            scores = decoder.fc(h)  # (k, vocab_size)
            scores = torch.log_softmax(scores, dim=1)  # log probabilities
            scores = top_k_scores.expand_as(scores) + scores  # add accumulated scores

            # Get top k scores and words
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = top_k_words // vocab_size  # which sequence
            next_word_inds = top_k_words % vocab_size   # next word

            # Append new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Identify completed sequences
            incomplete_inds = [ind for ind, word in enumerate(next_word_inds) if word != vocab.stoi['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Store completed sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())

            # Stop if all sequences are complete or max length is reached
            k -= len(complete_inds)
            if k == 0 or step >= max_length:
                break

            # Prepare next step
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            step += 1

        if len(complete_seqs) > 0:
            best_seq_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            best_seq = complete_seqs[best_seq_idx]
        else:
            best_seq = seqs[0].tolist()  # fallback: first incomplete sequence

        return [
            vocab.itos[idx]
            for idx in best_seq
            if idx not in {vocab.stoi["<START>"], vocab.stoi["<END>"], vocab.stoi["<PAD>"]}
        ]
    
class EarlyStopping:
    def __init__(self, patience=3, mode='loss'):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'loss' and score > self.best_score) or \
             (self.mode == 'bleu' and score < self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

