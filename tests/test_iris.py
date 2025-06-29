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

def test_single_prediction():
    X, y = load_data()
    model, _ = train_model(X, y)
    sample = X[0]
    pred = predict(model, sample)
    assert isinstance(pred, (int, float, np.integer, np.floating))
    assert 0 <= pred <= 2

def test_model_save_and_load():
    X, y = load_data()
    model, _ = train_model(X, y)
    save_model(model, 'temp_model.pkl')
    assert os.path.exists('temp_model.pkl')
    
    loaded_model = load_model('temp_model.pkl')
    sample = X[0]
    pred = predict(loaded_model, sample)
    assert 0 <= pred <= 2

    os.remove('temp_model.pkl')  # Clean up
