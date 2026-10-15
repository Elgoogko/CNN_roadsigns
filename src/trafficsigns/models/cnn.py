import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Construit un réseau de neurones convolutif (cnn) pour la classification
    de panneaux de signalisation.
    
    Args:
        input_shape: Dimensions de l'image d'entrée (ex: (48, 48, 3))
        num_classes: Nombre de classes à prédire (ex: 43)
        
    Returns:
        tf.keras.Model: Le modèle Keras compilé.
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        # Le Dropout désactive aléatoirement 50% des neurones pendant l'entraînement 
        # pour forcer le réseau à ne pas apprendre par cœur (évite l'overfitting)
        Dropout(0.5), 
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_cnn_model(input_shape=(48, 48, 3), num_classes=43)
    model.summary()