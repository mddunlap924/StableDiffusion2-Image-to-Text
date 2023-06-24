import numpy as np
from scipy import spatial
from torch import nn

def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred) 
        for y_true, y_pred in zip(y_trues, y_preds)
    ])


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output