import os
import argparse
from trafficsigns.data.preprocessing import normalize, NUM_CLASSES
from trafficsigns.models.baseline import build_baseline_model
from trafficsigns.utils.compute_parameters import get_mean_std
from trafficsigns.data.dataset import load_dataset, load_split
from pathlib import Path
import numpy as np

def train_model(data_dir: str, model_save_path: str, epochs: int, batch_size: int):
    """
    Charge les données, normalise, entraîne le modèle et sauvegarde les poids.
    """
    print(f"--- 1. Chargement des données depuis le dossier {data_dir} ---")
    x, y, _ = load_dataset(data_dir, mmap=True)
    train_idx, val_idx = load_split(data_dir)
    
    print("--- 2. Prétraitement (Normalisation) ---")
    mean, std = get_mean_std(data_dir)  # calculées sur le TRAIN uniquement
    x_train, y_train = normalize(x[train_idx], mean, std), y[train_idx]
    x_val, y_val = normalize(x[val_idx], mean, std), y[val_idx]  # mêmes stats

    print(f"Images d'entraînement : {x_train.shape}")
    print(f"Images de validation  : {x_val.shape}")

    print("\n--- 3. Construction du modèle ---")
    input_shape = x_train.shape[1:]
    num_classes = NUM_CLASSES
    
    model = build_baseline_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    print("\n--- 4. Début de l'entraînement ---")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    print(f"\n--- 5. Sauvegarde du modèle ---")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Modèle sauvegardé avec succès sous : {model_save_path}")
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path.parent / "norm_stats.npz", mean=mean, std=std)  # pour l'inférence

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