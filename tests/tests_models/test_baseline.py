import numpy as np
import tensorflow as tf
from src.trafficsigns.models.baseline import build_baseline_model

def test_build_baseline_model():
    """Teste la bonne construction du modèle de référence et la forme de sa sortie."""
    input_shape = (32, 32, 3)
    num_classes = 43
    
    model = build_baseline_model(input_shape, num_classes)
    
    assert isinstance(model, tf.keras.Model)
    
    batch_size = 2
    dummy_images = np.random.rand(batch_size, input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
    
    predictions = model.predict(dummy_images)

    assert predictions.shape == (batch_size, num_classes)
    
    for pred in predictions:
        assert np.isclose(np.sum(pred), 1.0, atol=1e-5)