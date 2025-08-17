from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu_scores(references, hypotheses):
    """
    Compute BLEU-1 and BLEU-2 scores with smoothing.
    """
    smoothie = SmoothingFunction().method4
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0.0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5), smoothing_function=smoothie)
    return bleu1, bleu2


def print_model_param_counts(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"{name} Parameters:")
    print(f"Total:       {total_params:,}")
    print(f"Trainable:   {trainable_params:,}")
    print(f"Frozen:      {frozen_params:,}")
    print("-" * 40)
