import torch

def compute_loss(end_points, d_set, criterion):
    logits = end_points['logits']  # Shape: [Batch, NumClasses, NumPoints]
    logits = logits-logits[:, 1:, :]
    labels = end_points['labels']  # Shape: [Batch, NumPoints]

    # A função criterion, com ignore_index=255, já ignora os rótulos corretos
    loss = criterion(logits, labels).mean()

    # Adiciona as previsões (a classe com maior logit) para cada ponto
    # Isso será usado pelas funções de métrica
    end_points['preds'] = torch.argmax(logits, dim=1)

    return loss, end_points