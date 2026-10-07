import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

def build_baseline_model(input_shape, num_classes):
    """
    Construit un modèle de référence (Dense/MLP) pour la classification multiclasse.
    Architecture : Image -> Flatten -> Dense (ReLU) -> Dense (Softmax)
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