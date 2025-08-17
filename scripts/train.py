from data.data_loader   import *
from models.model       import *
from scripts.evaluate   import *
from util.metrics       import *
from util.visualization import *

import os, torch
from tqdm.auto import tqdm

def train_model(encoder, decoder, dataloaders, criterion, optimizer,
                vocab, device, max_epochs=20, teacher_forcing=False,
                early_stopping_metric='loss', patience=3,
                save_path='checkpoints/', decode_strategy='greedy', beam_size=2):
    
    os.makedirs(save_path, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience, mode=early_stopping_metric)
    best_score = float('inf') if early_stopping_metric == 'loss' else 0

    train_loss_history = []
    val_loss_history = []
    bleu1_history = []
    bleu2_history = []

    for epoch in range(max_epochs):
        encoder.train()
        decoder.train()

        total_loss = 0

        train_pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1} [Train]", leave=True)
        for imgs, captions in train_pbar:
            imgs, captions = imgs.to(device), captions.to(device)
            caption_lengths = torch.tensor([len(cap) for cap in captions]).unsqueeze(1).to(device)

            optimizer.zero_grad()
            encoder_out = encoder(imgs)
            outputs, targets, lengths, *_ = decoder(encoder_out, captions, caption_lengths, teacher_forcing=teacher_forcing)

            # Remove <START> token for targets
            targets = targets[:, 1:]

            outputs = torch.cat([outputs[i, :l, :] for i, l in enumerate(lengths)], dim=0)
            targets = torch.cat([targets[i, :l] for i, l in enumerate(lengths)], dim=0)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(dataloaders['train'])
        train_loss_history.append(avg_train_loss)

        # ----------------- Validation --------------------
        encoder.eval()
        decoder.eval()
        val_total_loss = 0
        references = []
        hypotheses = []

        val_pbar = tqdm(dataloaders['val'], desc=f"Epoch {epoch+1} [Validation]", leave=True)
        with torch.no_grad():
            for imgs, caps in val_pbar:
                imgs, caps = imgs.to(device), caps
                caption_lengths = torch.tensor([len(cap) for cap in caps]).unsqueeze(1).to(device)
                encoder_out = encoder(imgs)
                outputs, targets, lengths, *_ = decoder(encoder_out, caps.to(device), caption_lengths, teacher_forcing=False)

                targets = targets[:, 1:]
                outputs = torch.cat([outputs[i, :l, :] for i, l in enumerate(lengths)], dim=0)
                targets = torch.cat([targets[i, :l] for i, l in enumerate(lengths)], dim=0)
                val_loss = criterion(outputs, targets)
                val_total_loss += val_loss.item()
                val_pbar.set_postfix({'loss': val_loss.item()}) 

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

        avg_val_loss = val_total_loss / len(dataloaders['val'])
        bleu1, bleu2 = compute_bleu_scores(references, hypotheses)
        val_loss_history.append(avg_val_loss) 
        bleu1_history.append(bleu1)        
        bleu2_history.append(bleu2)     

        print(f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}")

        # ----------------- Early Stopping --------------------
        score = avg_val_loss if early_stopping_metric == 'loss' else bleu2
        if (early_stopping_metric == 'loss' and score < best_score) or \
           (early_stopping_metric == 'bleu' and score > best_score):
            best_score = score
            encoder.save(os.path.join(save_path, 'encoder_best.pth'))
            decoder.save(os.path.join(save_path, 'decoder_best.pth'))

        early_stopper(score)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return train_loss_history, val_loss_history, bleu1_history, bleu2_history
