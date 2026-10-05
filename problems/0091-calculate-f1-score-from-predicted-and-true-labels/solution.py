def calculate_f1_score(y_true, y_pred):
    tp = fp = fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0

    if precision + recall == 0:
        return 0

    f1 = 2 * precision * recall / (precision + recall)
	return round(f1,3)