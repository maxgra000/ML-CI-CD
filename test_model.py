import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None

def test_predictions_length():
    preds, y_test = train_and_predict()
    assert len(preds) > 0
    assert len(preds) == len(y_test)

def test_predictions_value_range():
    preds, _ = train_and_predict()
    assert all(pred in [0, 1, 2] for pred in preds)

def test_model_accuracy():
    acc = get_accuracy()
    assert acc >= 0.7
