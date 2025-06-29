# tests/test_iris.py
from iris_pipeline.iris import load_data, train_model, predict

def test_data_shape():
    X, y = load_data()
    assert X.shape[0] == len(y)
    assert X.shape[1] == 4  # 4 features in IRIS

def test_model_accuracy():
    X, y = load_data()
    model, acc = train_model(X, y)
    assert acc > 0.7

def test_single_prediction():
    X, y = load_data()
    model, _ = train_model(X, y)
    sample = X[0]
    pred = predict(model, sample)
    assert isinstance(pred, int) or isinstance(pred, float)
    assert 0 <= pred <= 2  # IRIS has 3 classes (0, 1, 2)
