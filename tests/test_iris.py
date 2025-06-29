# tests/test_iris.py
from iris_pipeline.iris import load_data, train_model, predict, save_model, load_model
import os

def test_data_shape():
    X, y = load_data()
    assert X.shape[0] == len(y)
    assert X.shape[1] == 4

def test_model_accuracy():
    X, y = load_data()
    model, acc = train_model(X, y)
    assert acc > 0.7


