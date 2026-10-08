import os
import argparse
import numpy as np
import tensorflow as tf
from src.trafficsigns.models.baseline import build_baseline_model

def train_model(data_path: str, model_save_path: str, epochs: int, batch_size: int):
    """
    Charge les données .npz, entraîne le modèle et sauvegarde les poids.
    """
    print(f"--- 1. Chargement des données depuis {data_path} ---")
    data = np.load(data_path)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    print(f"Images d'entraînement : {X_train.shape}")
    print(f"Images de validation  : {X_val.shape}")

    print("\n--- 2. Construction du modèle ---")
    input_shape = X_train.shape[1:] 
    num_classes = len(np.unique(y_train)) 
    

    model = build_baseline_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    print("\n--- 3. Début de l'entraînement ---")
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    print(f"\n--- 4. Sauvegarde du modèle ---")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Modèle sauvegardé avec succès sous : {model_save_path}")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle")
    parser.add_argument("--data", type=str, default="data/processed/dataset.npz", help="Chemin vers le fichier de données")
    parser.add_argument("--model_out", type=str, default="models/baseline_model.keras", help="Chemin de sauvegarde du modèle")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille du batch")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_save_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size
    )