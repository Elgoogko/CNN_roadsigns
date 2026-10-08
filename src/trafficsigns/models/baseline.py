import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

def build_baseline_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Construit un modèle de référence (Dense/MLP) pour la classification multiclasse.

    Args:
        input_shape (tuple[int, int, int]): Les dimensions des images en entrée (ex: (32, 32, 3)).
        num_classes (int): Le nombre total de catégories de panneaux (ex: 43 pour GTSRB).
        
    Returns:
        tf.keras.Model: Le modèle Keras compilé, prêt pour l'entraînement.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Exemple pour des images de 32x32 pixels en RGB (3 canaux) et K=43 classes (ex: GTSRB)
    INPUT_SHAPE = (32, 32, 3)
    NUM_CLASSES = 43 
    
    model = build_baseline_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()