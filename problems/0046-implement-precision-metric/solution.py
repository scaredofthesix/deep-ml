import numpy as np
def precision(y_true, y_pred):
	tp = 0
	fp = 0
	for true, pred in zip(y_true, y_pred):
		if pred == 1:
			if true  == 1:
				tp += 1
			else:
				fp += 1
	if tp + fp == 0:
		return 0
	return tp/ (tp + fp)
