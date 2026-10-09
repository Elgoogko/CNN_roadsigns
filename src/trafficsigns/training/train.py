import os
import argparse
from pathlib import Path
import tensorflow as tf
from trafficsigns.data.preprocessing import load_dataset, normalize, NUM_CLASSES
from trafficsigns.data.split import load_split  # Correction de la coquille de Bari (split au lieu de splits)
from trafficsigns.models.baseline import build_baseline_model

def train_model(data_dir: str, model_save_path: str, epochs: int, batch_size: int):
    """
    Charge les données, normalise, entraîne le modèle et sauvegarde les poids.
    """
    print(f"--- 1. Chargement des données depuis le dossier {data_dir} ---")
    X, y, _ = load_dataset(data_dir, mmap=True)
    train_idx, val_idx = load_split(data_dir)
    
    print("--- 2. Prétraitement (Normalisation) ---")
    X_train, y_train = normalize(X[train_idx]), y[train_idx]
    X_val, y_val = normalize(X[val_idx]), y[val_idx]

    print(f"Images d'entraînement : {X_train.shape}")
    print(f"Images de validation  : {X_val.shape}")

    print("\n--- 3. Construction du modèle ---")
    input_shape = X_train.shape[1:] 
    num_classes = NUM_CLASSES
    
    model = build_baseline_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    print("\n--- 4. Début de l'entraînement ---")
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    print(f"\n--- 5. Sauvegarde du modèle ---")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Modèle sauvegardé avec succès sous : {model_save_path}")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle")
    parser.add_argument("--data_dir", type=str, default="data/processed/train", help="Dossier contenant img_mats, labels et split")
    parser.add_argument("--model_out", type=str, default="models/baseline_model.keras", help="Chemin de sauvegarde du modèle")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille du batch")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size
    )