import torch

def compute_loss(end_points, d_set, criterion):
    logits = end_points['logits']  # Shape: [Batch, NumClasses, NumPoints]
    labels = end_points['labels']  # Shape: [Batch, NumPoints]
    loss = criterion(logits, labels).mean()
    end_points['preds'] = torch.argmax(logits, dim=1)

    return loss, end_points