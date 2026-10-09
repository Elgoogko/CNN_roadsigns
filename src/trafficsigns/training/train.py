import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from src.trafficsigns.models.baseline import build_baseline_model

def train_model(data_dir: str, model_save_path: str, epochs: int, batch_size: int):
    """
    Charge les données séparées (X, y, split), normalise, entraîne le modèle et sauvegarde les poids.
    """
    print(f"--- 1. Chargement des données depuis le dossier {data_dir} ---")
    data_dir = Path(data_dir)
    
    X_full = np.load(data_dir / "img_mats.npy")
    y_full = np.load(data_dir / "labels.npy")
    
    splits = np.load(data_dir / "split.npz")
    train_idx, val_idx = splits["train_idx"], splits["val_idx"]
    
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    
    print("--- 2. Prétraitement (Normalisation) ---")
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0

    print(f"Images d'entraînement : {X_train.shape}")
    print(f"Images de validation  : {X_val.shape}")

    print("\n--- 3. Construction du modèle ---")
    input_shape = X_train.shape[1:] 
    num_classes = len(np.unique(y_full)) 
    
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