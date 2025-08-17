from data.data_loader   import *
from models.model       import *
from scripts.train      import *
from scripts.evaluate   import *
from util.metrics       import *
from util.visualization import *

import yaml
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import random_split

def main(config_path="config/config.yaml"):
    # 1) load config
    with open(config_path) as f: cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # 2) download & prepare data
    download_flickr8k(
        cfg['data']['kaggle_dataset'],
        cfg['data']['download_path'],
        kaggle_json_path=cfg['data'].get('kaggle_json_path')
    )

    vocab = Vocabulary(freq_threshold=cfg['data']['freq_threshold'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = Flickr8kDataset(
        cfg['data']['captions_file'],
        cfg['data']['images_dir'],
        vocabulary=vocab,
        transform=transform,
        max_length=cfg['data']['max_length']
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocab)} with frequency threshold {cfg['data']['freq_threshold']}")


    train_loader = get_loader(train_dataset, vocab=vocab, batch_size=60, shuffle=True, num_workers=2)
    val_loader = get_loader(val_dataset, vocab=vocab, batch_size=60, shuffle=False, num_workers=2)
    test_loader = get_loader(test_dataset, vocab=vocab, batch_size=60, shuffle=False, num_workers=2)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # for images, captions in dataloaders['train']:
    #     print(images.shape, captions.shape)
    #     break
    
    encoder = EncoderCNN(fine_tune_cnn=cfg['train']['encoder']['fine_tune']).to(device)
    decoder = DecoderRNN(
        embed_dim=cfg['train']['embed_dim'],
        decoder_dim=cfg['train']['decoder_dim'],
        vocab_size=len(vocab),
        dropout=cfg['train']['dropout'],
        use_attention=cfg['train']['use_attention']
    ).to(device)
    
    print_model_param_counts(encoder, name="Encoder (ResNet101)")
    print_model_param_counts(decoder, name="Decoder (LSTM)")

    # 3) train
    train_loss, val_loss, bleu1, bleu2 = train_model(
        encoder, decoder,
        dataloaders=dataloaders,
        criterion=nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"]),
        optimizer=torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=5e-5),
        vocab=vocab,
        device=device,
        teacher_forcing=False,
        early_stopping_metric='loss',
        decode_strategy=cfg['evaluate']['decode_strategy'],
        beam_size=cfg['evaluate']['beam_size'],
        patience=cfg['train']['patience'],
        max_epochs = cfg['train']['max_epochs']
    )

    plot_metrics(train_loss, val_loss, bleu1, bleu2)

    visualize_predictions(encoder, decoder, test_dataset, vocab, 
                          device=device, decode_strategy=cfg['evaluate']['decode_strategy'], 
                          num_samples=3)

    bleu1, bleu2 = evaluate_test_data(encoder, decoder, dataloaders['test'], vocab, 
                                      device, decode_strategy=cfg['evaluate']['decode_strategy'])

    
    if cfg['train']['use_attention']:
        sample_index = random.randint(0, len(test_dataset) - 1)
        sample_image, sample_caption_tensor = test_dataset[sample_index]

        visualize_attention_map(sample_image, encoder, decoder, vocab, device)

if __name__=="__main__":
    main()
