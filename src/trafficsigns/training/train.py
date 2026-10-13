import os
import argparse
from trafficsigns.data.preprocessing import normalize, NUM_CLASSES
from trafficsigns.config import load_config, ROOT, MODEL_REGISTRY
from trafficsigns.utils.compute_parameters import get_mean_std
from trafficsigns.data.dataset import load_dataset, load_split
from pathlib import Path
import numpy as np

def train_model(data_dir: str, model_save_path: str, epochs: int, batch_size: int, model_to_train: str):
    """
    Charge les données, normalise, entraîne le modèle et sauvegarde les poids.
    """
    assert model_to_train in MODEL_REGISTRY.keys()

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
    
    model = MODEL_REGISTRY[model_to_train](input_shape=input_shape, num_classes=num_classes)
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

    CFG = load_config(Path.joinpath(ROOT, "configs","base.yaml"))
    user_input = "n"
    model_to_train = "baseline"
    data_dir = "data/processed/train"
    model_out = "models/new_model.keras"
    epochs = 15
    batch_size = 32

    if CFG["train"]:
        while user_input not in ["n", "y"]:
            print("Configuration trouvé ! Utilisé les configurations de base ? [y/N]")
            user_input = input().lower()

    if user_input == "y":
        keys = CFG["data_dir"].keys()
        assert CFG["model"]["name"]

        model_to_train = CFG["model"]["name"]

        if "data_dir" in keys:
            data_dir = CFG["train"]["data_dir"]
        if "model_out" in keys:
            model_out = CFG["train"]["model_out"]
        if "epochs" in keys:
            epochs = CFG["train"]["epochs"]
        if batch_size in keys:
            batch_size = CFG["train"]["batch_size"]
    else:
        parser = argparse.ArgumentParser(description="Script d'entraînement du modèle")
        parser.add_argument("--data_dir", type=str, default=data_dir, help="Dossier contenant img_mats, labels et split")
        parser.add_argument("--model_out", type=str, default=model_out, help="Chemin de sauvegarde du modèle")
        parser.add_argument("--epochs", type=int, default=epochs, help="Nombre d'époques d'entraînement")
        parser.add_argument("--batch_size", type=int, default=batch_size, help="Taille du batch")

        args = parser.parse_args()

    train_model(
        data_dir=data_dir,
        model_save_path=model_out,
        epochs=epochs,
        batch_size=batch_size,
        model_to_train=model_to_train
    )