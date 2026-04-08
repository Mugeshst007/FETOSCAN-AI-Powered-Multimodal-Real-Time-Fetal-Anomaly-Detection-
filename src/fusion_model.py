import numpy as np

IMAGE_WEIGHT = 0.6
CTG_WEIGHT = 0.4

def late_fusion(img_pred=None, ctg_pred=None):

    preds = []
    weights = []

    if img_pred is not None:
        preds.append(img_pred)
        weights.append(IMAGE_WEIGHT)

    if ctg_pred is not None:
        preds.append(ctg_pred)
        weights.append(CTG_WEIGHT)

    if len(preds) == 0:
        raise ValueError("No input provided")

    final_pred = np.average(preds, axis=0, weights=weights)

    return final_pred