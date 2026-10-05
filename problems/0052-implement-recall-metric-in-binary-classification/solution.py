import numpy as np

def recall(y_true, y_pred):
    tp = 0
    fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1:
            if pred == 1:
                tp += 1
            else:
                fn += 1

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)
